"""Stdlib-only web server for interactive fire mapping.

Uses http.server.ThreadingHTTPServer — no FastAPI, no uvicorn, no Jinja2.
SSE (Server-Sent Events) via fetch() for real-time console streaming.
"""

import datetime
import glob
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs

import numpy as np
from osgeo import gdal

from .state import AppState, FireInfo, FireStatus
from .preview import generate_all_previews

gdal.UseExceptions()

_HERE = os.path.dirname(os.path.abspath(__file__))

# Global state — set by init_app() before the server starts
state: AppState = None

# GPU lock — serialises heavy operations (only one at a time)
_gpu_lock = threading.Lock()
_gpu_queue = 0            # number of tasks waiting or running
_gpu_queue_lock = threading.Lock()   # protects the counter


def init_app(app_state: AppState):
    global state
    state = app_state


_SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days in seconds


def _save_sessions():
    """Persist sessions to disk."""
    if not state.session_file:
        return
    try:
        import yaml
        with open(state.session_file, 'w') as f:
            yaml.dump(dict(state.sessions), f,
                      default_flow_style=False, sort_keys=False)
    except Exception:
        pass


def _save_notes():
    """Persist all fire notes to notes.yaml."""
    try:
        import yaml
        notes_path = os.path.join(state.output_root, 'notes.yaml')
        notes_data = {}
        for fn, fire in state.fires.items():
            if fire.notes:
                notes_data[fn] = fire.notes
        tmp_path = notes_path + '.tmp'
        with open(tmp_path, 'w') as f:
            yaml.dump(notes_data, f,
                      default_flow_style=False, sort_keys=True)
        os.replace(tmp_path, notes_path)
    except Exception:
        pass


def _compute_ml_area(fire: 'FireInfo',
                     clf_path: str = None) -> float:
    """Compute ML burned area in hectares from a classified raster.

    Returns area in ha or -1 if computation fails.
    """
    if clf_path is None:
        clf_path = os.path.join(
            fire.cache_dir,
            f'{fire.fire_numbe}_crop.bin_classified.bin')
    if not os.path.isfile(clf_path):
        return -1.0
    try:
        gt = state.raster_gt
        pixel_area_m2 = abs(gt[1] * gt[5])
        ds = gdal.Open(clf_path, gdal.GA_ReadOnly)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        burned_px = int(np.nansum(arr > 0))
        ml_area_ha = burned_px * pixel_area_m2 / 10000.0
        return round(ml_area_ha, 2)
    except Exception:
        return -1.0


def _overlay_mask_on_post(fire: 'FireInfo', raster_path: str,
                          out_name: str, color: tuple):
    """Overlay a binary raster on the post-fire preview.

    *color* is (r, g, b) floats 0-1 for the tint.
    Produces a pixel-aligned PNG at the same dimensions as post.png.
    """
    try:
        post_path = os.path.join(fire.cache_dir, 'previews', 'post.png')
        if not os.path.isfile(post_path) or not os.path.isfile(raster_path):
            return

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.image import imread, imsave
        from scipy.ndimage import zoom as scipy_zoom

        post = imread(post_path)
        if post.ndim == 2:
            post = np.stack([post] * 3, axis=2)

        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        ph, pw = post.shape[:2]
        ah, aw = arr.shape
        if ah != ph or aw != pw:
            arr = scipy_zoom(
                arr.astype(np.float32),
                (ph / ah, pw / aw), order=0)

        mask = arr > 0
        result = post[:, :, :3].copy()
        r, g, b = color
        result[mask, 0] = np.clip(result[mask, 0] * 0.3 + r * 0.7, 0, 1)
        result[mask, 1] = np.clip(result[mask, 1] * 0.3 + g * 0.7, 0, 1)
        result[mask, 2] = np.clip(result[mask, 2] * 0.3 + b * 0.7, 0, 1)

        out_path = os.path.join(fire.cache_dir, 'previews', f'{out_name}.png')
        imsave(out_path, np.clip(result, 0, 1))

        if (out_name not in fire.available_views
                and not out_name.startswith('serial_')):
            fire.available_views.append(out_name)
    except Exception:
        pass


def _generate_result_preview(fire: 'FireInfo'):
    """Generate pixel-aligned overlay previews after mapping."""
    clf_path = os.path.join(
        fire.cache_dir,
        f'{fire.fire_numbe}_crop.bin_classified.bin')
    _overlay_mask_on_post(fire, clf_path, 'result', (0.9, 0.1, 0.0))

    # Also generate hint overlay if hint raster exists
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))


def _compute_agreement(fire: 'FireInfo') -> float:
    """Compute overlap % between ML classification and hint perimeter.

    Returns percentage (0-100) or -1 if computation fails.
    """
    try:
        clf_path = os.path.join(
            fire.cache_dir,
            f'{fire.fire_numbe}_crop.bin_classified.bin')
        hint_path = fire.hint_bin
        if not clf_path or not hint_path:
            return -1.0
        if not os.path.isfile(clf_path) or not os.path.isfile(hint_path):
            return -1.0

        ds_clf = gdal.Open(clf_path, gdal.GA_ReadOnly)
        ds_hint = gdal.Open(hint_path, gdal.GA_ReadOnly)
        if ds_clf is None or ds_hint is None:
            return -1.0

        clf = ds_clf.GetRasterBand(1).ReadAsArray()
        hint = ds_hint.GetRasterBand(1).ReadAsArray()
        ds_clf = ds_hint = None

        if clf.shape != hint.shape:
            return -1.0

        clf_mask = clf > 0
        hint_mask = hint > 0
        union = np.sum(clf_mask | hint_mask)
        if union == 0:
            return 0.0
        intersection = np.sum(clf_mask & hint_mask)
        return round(float(intersection / union) * 100, 1)
    except Exception:
        return -1.0


# =========================================================================
# Parameter ranking — learns from accepted_params.csv
# =========================================================================

_SIZE_BUCKETS = [
    (0, 10), (10, 50), (50, 100), (100, 500),
    (500, 1000), (1000, 5000), (5000, float('inf')),
]

# Parameters that matter for ranking (exclude metadata/display)
_RANKABLE_PARAMS = [
    'sample_rate', 'min_samples', 'max_samples', 'seed',
    'embed_bands', 'tsne_perplexity', 'tsne_learning_rate',
    'tsne_max_iter', 'tsne_init', 'tsne_n_components',
    'tsne_random_state',
    'controlled_ratio', 'hdbscan_min_samples',
    'rf_n_estimators', 'rf_max_depth', 'rf_max_features',
    'rf_random_state',
    'padding', 'contour_width',
]


def _parse_fire_context(fire_numbe: str) -> tuple:
    """Extract (region, zone) from fire number.

    E.g., 'C11659' → ('C', 'C1'), 'G80123' → ('G', 'G8').
    """
    if not fire_numbe:
        return ('', '')
    region = fire_numbe[0].upper() if fire_numbe[0].isalpha() else ''
    zone = ''
    if len(fire_numbe) >= 2 and region:
        zone = region + fire_numbe[1]
    return (region, zone)


def _size_bucket(ha: float) -> tuple:
    """Return (lo, hi) bucket for a fire size."""
    for lo, hi in _SIZE_BUCKETS:
        if lo <= ha < hi:
            return (lo, hi)
    return _SIZE_BUCKETS[-1]


def _load_accepted_params() -> list[dict]:
    """Load accepted_params.csv into a list of dicts."""
    csv_path = os.path.join(state.output_root, 'accepted_params.csv')
    if not os.path.isfile(csv_path):
        return []
    try:
        import csv
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def _rank_params_for_fire(fire_numbe: str, fire_size_ha: float,
                          n: int = 3) -> list[dict]:
    """Return top-N parameter sets for a fire, ranked by agreement_pct.

    Uses hierarchical context matching:
      1. Same zone + same size bucket (e.g., C1 fires 100-500 ha)
      2. Same region + same size bucket (e.g., all C fires 100-500 ha)
      3. Same size bucket, any region
      4. Fall back to recommended_settings.yaml
    """
    all_rows = _load_accepted_params()
    if not all_rows:
        # Cold start — use all recommended settings tiers
        seen = set()
        ranked = []
        # Size-matched tier first
        rec = _get_recommended_params(fire_size_ha)
        if rec:
            key = tuple(sorted(
                (k, str(v)) for k, v in rec.items() if v is not None))
            seen.add(key)
            ranked.append(rec)
        # Then remaining tiers
        for row in state.recommended_settings:
            if len(ranked) >= n:
                break
            tier_params = row.get('params', {})
            if not tier_params:
                continue
            key = tuple(sorted(
                (k, str(v)) for k, v in tier_params.items()
                if v is not None))
            if key not in seen:
                seen.add(key)
                ranked.append(dict(tier_params))
        return ranked[:n]

    region, zone = _parse_fire_context(fire_numbe)
    bucket = _size_bucket(fire_size_ha)

    def in_bucket(row):
        try:
            sz = float(row.get('fire_size_ha', 0))
            return bucket[0] <= sz < bucket[1]
        except (ValueError, TypeError):
            return False

    def row_context(row):
        fn = row.get('fire_numbe', '')
        r, z = _parse_fire_context(fn)
        return (r, z)

    def row_agreement(row):
        try:
            return float(row.get('agreement_pct', -1))
        except (ValueError, TypeError):
            return -1.0

    def extract_params(row):
        """Extract a clean params dict from a CSV row.

        Only includes keys that have actual values — omits None/empty
        so that downstream .get(key, default) falls back correctly.
        """
        params = {}
        for key in _RANKABLE_PARAMS:
            val = row.get(key)
            if val is not None and val != '':
                # Try to convert to number
                try:
                    if '.' in str(val):
                        params[key] = float(val)
                    else:
                        params[key] = int(val)
                except (ValueError, TypeError):
                    params[key] = val
        return params

    # Hierarchical matching
    candidates = None
    min_required = 3

    # Level 1: same zone + same size bucket
    if zone:
        level1 = [r for r in all_rows
                   if in_bucket(r) and row_context(r)[1] == zone]
        if len(level1) >= min_required:
            candidates = level1

    # Level 2: same region + same size bucket
    if candidates is None and region:
        level2 = [r for r in all_rows
                   if in_bucket(r) and row_context(r)[0] == region]
        if len(level2) >= min_required:
            candidates = level2

    # Level 3: same size bucket, any region
    if candidates is None:
        level3 = [r for r in all_rows if in_bucket(r)]
        if len(level3) >= 1:
            candidates = level3

    # Level 4: all accepted fires
    if candidates is None:
        candidates = all_rows

    if not candidates:
        params = _get_recommended_params(fire_size_ha)
        return [params] if params else []

    # Sort by agreement_pct descending
    candidates.sort(key=row_agreement, reverse=True)

    # Extract unique parameter sets (deduplicate)
    seen = set()
    ranked = []
    for row in candidates:
        params = extract_params(row)
        # Create a hashable key for deduplication
        key = tuple(sorted(
            (k, str(v)) for k, v in params.items() if v is not None))
        if key not in seen:
            seen.add(key)
            ranked.append(params)
        if len(ranked) >= n:
            break

    # If we don't have enough, pad with recommended settings.
    # Try the size-matched tier first, then all other tiers.
    if len(ranked) < n:
        # Size-matched tier first
        rec = _get_recommended_params(fire_size_ha)
        if rec:
            key = tuple(sorted(
                (k, str(v)) for k, v in rec.items() if v is not None))
            if key not in seen:
                seen.add(key)
                ranked.append(rec)
        # Then remaining tiers
        for row in state.recommended_settings:
            if len(ranked) >= n:
                break
            tier_params = row.get('params', {})
            if not tier_params:
                continue
            key = tuple(sorted(
                (k, str(v)) for k, v in tier_params.items()
                if v is not None))
            if key not in seen:
                seen.add(key)
                ranked.append(dict(tier_params))

    return ranked[:n]


def _get_recommended_params(fire_size_ha: float) -> dict:
    """Find recommended params for a fire of given size.

    Always returns a fresh copy so callers can never mutate the
    canonical settings stored in ``state.recommended_settings``.
    """
    for row in state.recommended_settings:
        if (fire_size_ha >= row.get('min_ha', 0)
                and (row.get('max_ha') is None
                     or fire_size_ha < row['max_ha'])):
            return dict(row.get('params', {}))
    return {}


_batch_thread = None


def _batch_map_worker(fire_numbes: list[str]):
    """Process fires sequentially with recommended settings."""
    import traceback

    state.batch_status = {
        'running': True,
        'total': len(fire_numbes),
        'completed': 0,
        'current_fire': '',
        'errors': [],
    }

    sys.stderr.write(
        f'[batch] Starting batch: {len(fire_numbes)} fire(s)\n')
    sys.stderr.flush()

    for fire_numbe in fire_numbes:
        fire = state.fires.get(fire_numbe)
        if not fire or fire.status in (
                FireStatus.ACCEPTED, FireStatus.MAPPING):
            state.batch_status['completed'] += 1
            continue

        state.batch_status['current_fire'] = fire_numbe
        sys.stderr.write(f'[batch] [{fire_numbe}] Starting...\n')
        sys.stderr.flush()

        params = _get_recommended_params(fire.fire_size_ha)
        if not params:
            params = {}

        padding = params.get('padding')
        if padding is None:
            padding = state.padding
        padding = float(padding)

        try:
            with _gpu_lock:
                # Prepare
                sys.stderr.write(
                    f'[batch] [{fire_numbe}] Preparing '
                    f'(padding={padding})...\n')
                sys.stderr.flush()
                _prepare_fire_sync(fire_numbe, padding)
                fire = state.fires[fire_numbe]
                if fire.status == FireStatus.ERROR:
                    sys.stderr.write(
                        f'[batch] [{fire_numbe}] Prepare FAILED: '
                        f'{fire.error_msg}\n')
                    sys.stderr.flush()
                    state.batch_status['errors'].append(fire_numbe)
                    state.batch_status['completed'] += 1
                    continue

                # Map
                fire.status = FireStatus.MAPPING
                fire.console_log = []
                fire.last_params = params
                state.current_job = {
                    'fire_numbe': fire_numbe,
                    'client_ip': 'batch',
                    'started_at': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                }

                cmd = _build_mapping_cmd(fire, params)
                sys.stderr.write(
                    f'[batch] [{fire_numbe}] Running CLI: '
                    f'{" ".join(os.path.basename(c) for c in cmd[:3])} '
                    f'...\n')
                sys.stderr.flush()

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=state.project_root,
                )

                for line in iter(proc.stdout.readline, b''):
                    text = line.decode(errors='replace').rstrip()
                    if text:
                        fire.console_log.append(text)
                        sys.stderr.write(
                            f'[batch] [{fire_numbe}] {text}\n')
                        sys.stderr.flush()

                rc = proc.wait()
                state.current_job = None

                if rc == 0:
                    comp = os.path.join(
                        fire.cache_dir,
                        f'{fire_numbe}_comparison.png')
                    fire.last_comparison = (
                        comp if os.path.exists(comp) else '')
                    fire.last_params = params
                    fire.agreement_pct = _compute_agreement(fire)
                    fire.ml_area_ha = _compute_ml_area(fire)
                    _generate_result_preview(fire)
                    fire.status = FireStatus.MAPPED
                    sys.stderr.write(
                        f'[batch] [{fire_numbe}] MAPPED OK '
                        f'(agreement={fire.agreement_pct}%)\n')
                else:
                    fire.status = FireStatus.ERROR
                    fire.error_msg = f'Exited with code {rc}'
                    state.batch_status['errors'].append(fire_numbe)
                    sys.stderr.write(
                        f'[batch] [{fire_numbe}] FAILED (rc={rc})\n')
                sys.stderr.flush()

        except Exception as exc:
            fire.status = FireStatus.ERROR
            fire.error_msg = str(exc)
            state.batch_status['errors'].append(fire_numbe)
            state.current_job = None
            sys.stderr.write(
                f'[batch] [{fire_numbe}] EXCEPTION:\n'
                f'{traceback.format_exc()}\n')
            sys.stderr.flush()

        state.batch_status['completed'] += 1

    state.batch_status['running'] = False
    state.batch_status['current_fire'] = ''
    sys.stderr.write(
        f'[batch] Complete: {state.batch_status["completed"]}'
        f'/{state.batch_status["total"]} fires, '
        f'{len(state.batch_status["errors"])} error(s)\n')
    sys.stderr.flush()


def _serial_map_worker(fire_numbe: str, param_sets: list[dict]):
    """Run N mappings for the same fire.

    Optimisation: the expensive deterministic part (t-SNE + RF) runs once
    on the first invocation and is cached.  Runs 2-N load the cached state
    and only re-run HDBSCAN, which is much faster.
    """
    import traceback

    fire = state.fires[fire_numbe]
    fire.serial_results = []
    fire.console_log = []
    n_total = len(param_sets)

    # Path for the shared intermediate state (.npz)
    state_file = os.path.join(
        fire.cache_dir, f'{fire_numbe}_serial_state.npz')

    sys.stderr.write(
        f'[serial] Starting {n_total} run(s) for {fire_numbe}\n')
    sys.stderr.flush()

    # Snapshot previously accepted results as Run 0
    if fire.previously_accepted:
        try:
            old_agreement = fire.agreement_pct
            old_ml_area = fire.ml_area_ha
            old_params = dict(fire.last_params) if fire.last_params else {}

            # Search both cache_dir and canonical output dir
            canon_dir = os.path.join(state.output_root, fire_numbe)
            search_dirs = [fire.cache_dir]
            if os.path.isdir(canon_dir):
                search_dirs.append(canon_dir)

            # Find existing classified raster
            old_clf = None
            for _dir in search_dirs:
                for _pat in (f'{fire_numbe}_crop.bin_classified.bin',
                             f'{fire_numbe}_crop_classified.bin',
                             f'{fire_numbe}_classified.bin'):
                    _cand = os.path.join(_dir, _pat)
                    if os.path.isfile(_cand):
                        old_clf = _cand
                        break
                if old_clf:
                    break
            if old_clf is None:
                for _dir in search_dirs:
                    for _cand in glob.glob(os.path.join(
                            _dir, '*classified*.bin')):
                        if 'serial_' not in os.path.basename(_cand):
                            old_clf = _cand
                            break
                    if old_clf:
                        break

            # Copy classified raster as serial_0
            s0_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_0_classified.bin')
            if old_clf and os.path.isfile(old_clf):
                shutil.copy2(old_clf, s0_clf)
                old_hdr = os.path.splitext(old_clf)[0] + '.hdr'
                if not os.path.isfile(old_hdr):
                    old_hdr = old_clf + '.hdr'
                if os.path.isfile(old_hdr):
                    shutil.copy2(
                        old_hdr,
                        os.path.splitext(s0_clf)[0] + '.hdr')

            # Copy comparison PNG as serial_0
            s0_comp = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_0.png')
            for _dir in search_dirs:
                old_comp = os.path.join(
                    _dir, f'{fire_numbe}_comparison.png')
                if os.path.isfile(old_comp):
                    shutil.copy2(old_comp, s0_comp)
                    break

            # Copy brush comparison as serial_0
            s0_brush = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_0_brush.png')
            for _dir in search_dirs:
                old_brush = os.path.join(
                    _dir, f'{fire_numbe}_brush_comparison.png')
                if os.path.isfile(old_brush):
                    shutil.copy2(old_brush, s0_brush)
                    break

            # Generate overlay for Run 0
            if os.path.isfile(s0_clf):
                _overlay_mask_on_post(
                    fire, s0_clf, 'serial_0', (0.9, 0.1, 0.0))

            fire.serial_results.append({
                'run_id': 0,
                'params': old_params,
                'agreement_pct': old_agreement,
                'ml_area_ha': old_ml_area,
                'comparison': s0_comp,
                'classified': s0_clf,
                'is_previous': True,
            })
            fire.console_log.append(
                '=== Run 0: Previously accepted result ===')
            fire.console_log.append(
                f'  Agreement: {old_agreement}%'
                f', ML area: {old_ml_area} ha')
        except Exception:
            sys.stderr.write(
                f'[serial] Warning: could not snapshot previous '
                f'result for {fire_numbe}\n')
            sys.stderr.flush()

    # Use the first param set's padding for any prepare needed
    base_params = param_sets[0]
    padding = base_params.get('padding')
    if padding is None:
        padding = state.padding
    padding = float(padding)

    for i, params in enumerate(param_sets):
        run_id = i + 1
        fire.console_log.append(
            f'=== Serial run {run_id}/{n_total} ===')

        # Log key params for this run (only the ones that vary)
        key_params = []
        for k in ('hdbscan_min_samples', 'controlled_ratio'):
            v = params.get(k)
            if v is not None and v != '':
                key_params.append(f'{k}={v}')
        if key_params:
            fire.console_log.append(
                f'  HDBSCAN params: {", ".join(key_params)}')
        if run_id == 1:
            # Log full params for the first (full pipeline) run
            full_keys = []
            for k in ('embed_bands', 'tsne_perplexity', 'sample_rate',
                       'padding'):
                v = params.get(k)
                if v is not None and v != '':
                    full_keys.append(f'{k}={v}')
            if full_keys:
                fire.console_log.append(
                    f'  Base params: {", ".join(full_keys)}')

        try:
            with _gpu_lock:
                # Re-prepare only when padding changed or files missing
                if run_id == 1:
                    needs_prepare = (
                        fire.padding_used != padding
                        or not fire.cache_dir
                        or not os.path.isdir(fire.cache_dir)
                        or not fire.crop_bin
                        or not os.path.isfile(fire.crop_bin)
                        or not fire.hint_bin
                        or not os.path.isfile(fire.hint_bin)
                    )
                    if needs_prepare:
                        fire.console_log.append(
                            f'  Re-preparing (padding '
                            f'{fire.padding_used} → {padding}) ...')
                        _prepare_fire_sync(fire_numbe, padding)
                        fire = state.fires[fire_numbe]
                        if fire.status == FireStatus.ERROR:
                            fire.serial_results.append({
                                'run_id': run_id,
                                'params': params,
                                'agreement_pct': -1,
                                'error': fire.error_msg,
                            })
                            break  # all runs share same prep — stop

                fire.status = FireStatus.MAPPING
                fire.last_params = params
                state.current_job = {
                    'fire_numbe': f'{fire_numbe} (run {run_id}/{n_total})',
                    'client_ip': 'serial',
                    'started_at': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                }

                # Run 1: full pipeline + save state
                # Runs 2-N: load cached state (HDBSCAN only)
                is_first = (run_id == 1)
                cmd = _build_mapping_cmd(
                    fire, params,
                    save_state=state_file if is_first else None,
                    load_state=state_file if not is_first else None,
                )

                if not is_first:
                    fire.console_log.append(
                        '  (resuming from cached t-SNE + RF)')

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=state.project_root,
                )

                for line in iter(proc.stdout.readline, b''):
                    text = line.decode(errors='replace').rstrip()
                    if text:
                        fire.console_log.append(text)
                        sys.stderr.write(
                            f'[serial] [{fire_numbe}#{run_id}] {text}\n')

                rc = proc.wait()
                state.current_job = None
                sys.stderr.flush()

                if rc == 0:
                    # Compute agreement
                    agr = _compute_agreement(fire)

                    # Save serial comparison image
                    src_comp = os.path.join(
                        fire.cache_dir,
                        f'{fire_numbe}_comparison.png')
                    serial_comp = os.path.join(
                        fire.cache_dir,
                        f'{fire_numbe}_serial_{run_id}.png')
                    if os.path.isfile(src_comp):
                        shutil.copy2(src_comp, serial_comp)

                    # Save serial brush comparison
                    src_brush = os.path.join(
                        fire.cache_dir,
                        f'{fire_numbe}_brush_comparison.png')
                    serial_brush = os.path.join(
                        fire.cache_dir,
                        f'{fire_numbe}_serial_{run_id}_brush.png')
                    if os.path.isfile(src_brush):
                        shutil.copy2(src_brush, serial_brush)

                    # Find the classified raster the CLI just wrote
                    src_clf = None
                    for _pat in (f'{fire_numbe}_crop.bin_classified.bin',
                                 f'{fire_numbe}_crop_classified.bin',
                                 f'{fire_numbe}_classified.bin'):
                        _cand = os.path.join(fire.cache_dir, _pat)
                        if os.path.isfile(_cand):
                            src_clf = _cand
                            break
                    if src_clf is None:
                        for _cand in glob.glob(os.path.join(
                                fire.cache_dir, '*classified*.bin')):
                            src_clf = _cand
                            break

                    # Save serial classified raster + ENVI header
                    serial_clf = os.path.join(
                        fire.cache_dir,
                        f'{fire_numbe}_serial_{run_id}_classified.bin')
                    if src_clf and os.path.isfile(src_clf):
                        shutil.copy2(src_clf, serial_clf)
                        src_hdr = (
                            os.path.splitext(src_clf)[0] + '.hdr')
                        if not os.path.isfile(src_hdr):
                            src_hdr = src_clf + '.hdr'
                        if os.path.isfile(src_hdr):
                            shutil.copy2(
                                src_hdr,
                                os.path.splitext(serial_clf)[0]
                                + '.hdr')

                    # Use whichever classified raster exists
                    clf_for_run = serial_clf if os.path.isfile(
                        serial_clf) else src_clf

                    # Compute ML area for this run
                    run_ml_area = _compute_ml_area(
                        fire, clf_for_run) if clf_for_run else -1.0

                    # Generate pixel-aligned overlay for this run
                    if clf_for_run:
                        _overlay_mask_on_post(
                            fire, clf_for_run, f'serial_{run_id}',
                            (0.9, 0.1, 0.0))

                    # Update the main 'result' overlay so the dropdown
                    # shows "ML classification" immediately — just copy
                    # the per-run overlay instead of re-rendering
                    serial_overlay = os.path.join(
                        fire.cache_dir, 'previews',
                        f'serial_{run_id}.png')
                    result_overlay = os.path.join(
                        fire.cache_dir, 'previews', 'result.png')
                    if os.path.isfile(serial_overlay):
                        shutil.copy2(serial_overlay, result_overlay)
                        if 'result' not in fire.available_views:
                            fire.available_views.append('result')
                    else:
                        _generate_result_preview(fire)

                    fire.serial_results.append({
                        'run_id': run_id,
                        'params': params,
                        'agreement_pct': agr,
                        'ml_area_ha': run_ml_area,
                        'comparison': serial_comp,
                        'classified': serial_clf,
                    })

                    fire.console_log.append(
                        f'Run {run_id} complete (agreement={agr}%'
                        f', ML area={run_ml_area} ha)')
                else:
                    fire.serial_results.append({
                        'run_id': run_id,
                        'params': params,
                        'agreement_pct': -1,
                        'error': f'Exited with code {rc}',
                    })
                    fire.console_log.append(
                        f'Run {run_id} FAILED (exit code {rc})')
                    # If run 1 fails, no state to resume from
                    if is_first:
                        fire.console_log.append(
                            'Stopping serial — no cached state.')
                        break

        except Exception as exc:
            state.current_job = None
            fire.serial_results.append({
                'run_id': run_id,
                'params': params,
                'agreement_pct': -1,
                'error': str(exc),
            })
            sys.stderr.write(
                f'[serial] [{fire_numbe}#{run_id}] EXCEPTION:\n'
                f'{traceback.format_exc()}\n')
            sys.stderr.flush()
            if run_id == 1:
                break  # no cached state to resume from

    # Pick best result as the "current" mapping (exclude Run 0 / previous)
    successful = [r for r in fire.serial_results
                  if r.get('agreement_pct', -1) >= 0
                  and not r.get('is_previous')]
    if successful:
        best = max(successful, key=lambda r: r['agreement_pct'])
        fire.agreement_pct = best['agreement_pct']
        fire.ml_area_ha = best.get('ml_area_ha', -1.0)
        fire.last_params = best['params']

        # Copy best result as the "main" comparison
        best_comp = best.get('comparison', '')
        if best_comp and os.path.isfile(best_comp):
            main_comp = os.path.join(
                fire.cache_dir, f'{fire_numbe}_comparison.png')
            shutil.copy2(best_comp, main_comp)
            fire.last_comparison = main_comp

        best_clf = best.get('classified', '')
        if best_clf and os.path.isfile(best_clf):
            main_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_crop.bin_classified.bin')
            shutil.copy2(best_clf, main_clf)
            best_hdr = (
                os.path.splitext(best_clf)[0] + '.hdr')
            if not os.path.isfile(best_hdr):
                best_hdr = best_clf + '.hdr'
            if os.path.isfile(best_hdr):
                shutil.copy2(
                    best_hdr,
                    os.path.splitext(main_clf)[0] + '.hdr')
            _generate_result_preview(fire)

        fire.status = FireStatus.MAPPED
        fire.console_log.append(
            f'Serial mapping complete. Best: run {best["run_id"]} '
            f'(agreement={best["agreement_pct"]}%)')
    else:
        fire.status = FireStatus.ERROR
        fire.error_msg = 'All serial runs failed'
        fire.console_log.append('All serial runs failed.')

    sys.stderr.write(
        f'[serial] {fire_numbe} done: {len(successful)}/{n_total} '
        f'successful\n')
    sys.stderr.flush()


def _save_ip_list():
    """Persist approved and blocked IPs to disk."""
    if not state.ip_file:
        return
    try:
        import yaml
        data = {
            'approved': dict(state.approved_ips),
            'blocked': dict(state.blocked_ips),
        }
        with open(state.ip_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except Exception:
        pass


# =========================================================================
# Simple template rendering  (replaces Jinja2)
# =========================================================================

def _html_escape(s: str) -> str:
    """Escape HTML special characters."""
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;')
             .replace("'", '&#39;'))


def render_template(name: str, context: dict) -> str:
    """Replace ``{{ key }}`` placeholders in a template file.

    Values are HTML-escaped by default.  Use ``{{{ key }}}`` for raw
    (unescaped) insertion when the value is known-safe.
    """
    path = os.path.join(_HERE, 'templates', name)
    with open(path) as f:
        html = f.read()
    for key, val in context.items():
        html = html.replace('{{{ ' + key + ' }}}', str(val))   # raw
        html = html.replace('{{ ' + key + ' }}', _html_escape(str(val)))
    return html


# =========================================================================
# Preparation — runs synchronously (called from request thread)
# =========================================================================

def _prepare_fire_sync(fire_numbe: str, padding: float | None = None):
    """Prepare a fire for mapping: crop, VIIRS accumulate, hint, previews."""
    from batch_fire_mapping.run_fire_mapping import (
        raster_native_extent, crop_raster, rasterize_polygon,
    )
    from shapely.geometry import box as shapely_box

    fire = state.fires[fire_numbe]
    fire.status = FireStatus.PREPARING
    fire.error_msg = ""

    pad = padding if padding is not None else state.padding

    try:
        row = state.gdf[
            state.gdf['FIRE_NUMBE'].astype(str) == fire_numbe
        ].iloc[0]
    except (IndexError, KeyError):
        fire.status = FireStatus.ERROR
        fire.error_msg = f"Fire {fire_numbe} not found in shapefile"
        return

    # -- Parse FIRE_DATE --
    raw = row.get('FIRE_DATE', '')
    try:
        if hasattr(raw, 'date'):
            fire_date = datetime.datetime(raw.year, raw.month, raw.day)
        else:
            fire_date = datetime.datetime.strptime(
                str(raw).split()[0], '%Y-%m-%d')
    except (ValueError, AttributeError):
        fire.status = FireStatus.ERROR
        fire.error_msg = f"Cannot parse FIRE_DATE: {raw!r}"
        return

    acc_start = fire_date - datetime.timedelta(days=5)

    # -- Clip polygon to raster, compute crop bounds --
    gt = state.raster_gt
    W, H = state.raster_W, state.raster_H
    rx1, ry1, rx2, ry2 = raster_native_extent(gt, W, H)
    raster_box = shapely_box(rx1, ry1, rx2, ry2)

    clipped = row.geometry.intersection(raster_box)
    if clipped.is_empty:
        fire.status = FireStatus.ERROR
        fire.error_msg = "Fire polygon does not overlap the raster"
        return

    bounds = clipped.bounds  # (minx, miny, maxx, maxy)
    px_lo = int((bounds[0] - gt[0]) / gt[1])
    px_hi = int((bounds[2] - gt[0]) / gt[1])
    py_lo = int((bounds[3] - gt[3]) / gt[5])  # maxy -> top row
    py_hi = int((bounds[1] - gt[3]) / gt[5])  # miny -> bottom row

    fire_max_dim = max(px_hi - px_lo, py_hi - py_lo)
    p = max(1, int(round(pad * fire_max_dim)))
    px_lo = max(0, px_lo - p)
    px_hi = min(W - 1, px_hi + p)
    py_lo = max(0, py_lo - p)
    py_hi = min(H - 1, py_hi + p)

    if px_lo >= px_hi or py_lo >= py_hi:
        fire.status = FireStatus.ERROR
        fire.error_msg = "Crop box has zero area after clipping"
        return

    crop_xmin = gt[0] + px_lo * gt[1]
    crop_xmax = gt[0] + px_hi * gt[1]
    crop_ymax = gt[3] + py_lo * gt[5]
    crop_ymin = gt[3] + py_hi * gt[5]

    crop_w = px_hi - px_lo
    crop_h = py_hi - py_lo
    fire.crop_w = crop_w
    fire.crop_h = crop_h
    fire.padding_used = pad

    sample_size = int(round(crop_w * crop_h * state.sample_rate))
    sample_size = max(state.min_samples, min(state.max_samples, sample_size))
    fire.sample_size = sample_size

    # -- Create / clear cache directory --
    cache_dir = os.path.join(state.output_root, '.web_cache', fire_numbe)
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    fire.cache_dir = cache_dir

    # -- Crop raster --
    crop_bin = os.path.join(cache_dir, f'{fire_numbe}_crop.bin')
    if not crop_raster(state.raster_path, crop_bin,
                       crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        fire.status = FireStatus.ERROR
        fire.error_msg = "GDAL crop failed"
        return
    fire.crop_bin = crop_bin

    crop_gt = (crop_xmin, gt[1], gt[2], crop_ymax, gt[4], gt[5])

    # -- Rasterize traditional perimeter --
    perim_bin = os.path.join(cache_dir, f'{fire_numbe}_perimeter.bin')
    try:
        rasterize_polygon(
            state.polygon_file, fire_numbe, state.raster_crs,
            crop_bin, perim_bin, geometry=clipped,
            crop_gt=crop_gt, crop_w=crop_w, crop_h=crop_h)
    except Exception:
        perim_bin = None
    fire.perim_bin = perim_bin or ''

    # -- VIIRS accumulation --
    viirs_bin = None
    acc_end = fire_date
    plot_start = acc_start.date()
    plot_end = fire_date.date()

    if (state.perimeter_mode == 'viirs'
            and state.viirs_gdf is not None
            and not state.viirs_gdf.empty):
        from viirs.utils.accumulate import accumulate
        from viirs.utils.rasterize import rasterize_shapefile

        inside = state.viirs_gdf[
            state.viirs_gdf.geometry.within(row.geometry)]

        if not inside.empty:
            acc_end = datetime.datetime.combine(
                inside['detection_date'].max(), datetime.time.min)
            inside_window = inside[
                inside['detection_datetime'] >= acc_start]
            if not inside_window.empty:
                plot_start = inside_window['detection_date'].min()
            plot_end = acc_end.date()

        try:
            acc_paths = accumulate(
                shp_dir=state.viirs_shp_dir,
                start_str=acc_start.strftime('%Y%m%d'),
                end_str=acc_end.strftime('%Y%m%d'),
                reference_raster=crop_bin,
                output_dir=cache_dir,
                final_only=True,
                bbox=(crop_xmin, crop_ymin, crop_xmax, crop_ymax),
            )
        except Exception:
            acc_paths = []

        if acc_paths:
            try:
                viirs_bin = rasterize_shapefile(
                    shp_path=acc_paths[-1],
                    ref_image=crop_bin,
                    output_dir=cache_dir,
                    buffer_m=375.0,
                )
                if viirs_bin:
                    ds = gdal.Open(viirs_bin, gdal.GA_ReadOnly)
                    arr = ds.GetRasterBand(1).ReadAsArray()
                    ds = None
                    if np.nansum(arr) == 0:
                        viirs_bin = None
            except Exception:
                viirs_bin = None

    # -- Select hint raster --
    if state.perimeter_mode == 'traditional':
        fire.hint_bin = perim_bin or ''
        fire.perimeter_type = 'traditional'
    elif viirs_bin:
        fire.hint_bin = viirs_bin
        fire.viirs_bin = viirs_bin
        fire.perimeter_type = 'viirs'
    elif perim_bin and os.path.exists(perim_bin):
        fire.hint_bin = perim_bin
        fire.perimeter_type = 'traditional'
    else:
        fire.status = FireStatus.ERROR
        fire.error_msg = "No classification hint available"
        return

    fire.acc_start = str(plot_start)
    fire.acc_end = str(plot_end)

    # -- Generate preview images --
    views = generate_all_previews(crop_bin, cache_dir, fire_numbe)
    fire.available_views = views

    # -- Copy results from canonical dir for previously accepted fires --
    canon_dir = os.path.join(state.output_root, fire_numbe)
    if os.path.isdir(canon_dir):
        copied = []
        for fname in os.listdir(canon_dir):
            src = os.path.join(canon_dir, fname)
            dst = os.path.join(cache_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(fname)
        if copied:
            sys.stderr.write(
                f'[prepare] [{fire_numbe}] Restored {len(copied)} '
                f'file(s) from accepted dir\n')
            sys.stderr.flush()

    # -- Find classified raster (try multiple naming patterns) --
    clf_path = None
    for pattern in (f'{fire_numbe}_crop.bin_classified.bin',
                    f'{fire_numbe}_crop_classified.bin',
                    f'{fire_numbe}_classified.bin'):
        candidate = os.path.join(cache_dir, pattern)
        if os.path.isfile(candidate):
            clf_path = candidate
            break
    if clf_path is None:
        # Last resort: any *classified*.bin
        for candidate in glob.glob(
                os.path.join(cache_dir, '*classified*.bin')):
            clf_path = candidate
            break

    # -- Generate overlay previews (always try both) --
    if clf_path and os.path.isfile(clf_path):
        # Point fire at the classified raster for overlay generation
        _overlay_mask_on_post(fire, clf_path, 'result', (0.9, 0.1, 0.0))
        if 'result' not in fire.available_views:
            fire.available_views.append('result')
        sys.stderr.write(
            f'[prepare] [{fire_numbe}] Generated ML classification '
            f'overlay from {os.path.basename(clf_path)}\n')
        sys.stderr.flush()
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))

    fire.status = FireStatus.READY


# =========================================================================
# Accept — runs synchronously
# =========================================================================

def _accept_fire_sync(fire_numbe: str) -> str:
    """Copy results from cache to canonical dir, write params. Returns path."""
    fire = state.fires[fire_numbe]
    cache_dir = fire.cache_dir
    fire_dir = os.path.join(state.output_root, fire_numbe)

    if os.path.isdir(fire_dir):
        shutil.rmtree(fire_dir)
    os.makedirs(fire_dir)

    for pattern in ('*.bin', '*.hdr', '*.png', '*.shp', '*.dbf',
                     '*.shx', '*.prj', '*.cpg'):
        for f in glob.glob(os.path.join(cache_dir, pattern)):
            shutil.copy2(f, fire_dir)

    # Compute ML area from the accepted dir
    clf_bin = os.path.join(
        fire_dir, f'{fire_numbe}_crop.bin_classified.bin')
    ml_area_val = _compute_ml_area(fire, clf_bin)
    ml_area_ha = ml_area_val if ml_area_val >= 0 else None
    ml_area_m2 = (ml_area_ha * 10000.0) if ml_area_ha is not None else None
    fire.ml_area_ha = ml_area_val

    # Write params YAML
    try:
        import yaml
        params_dict = {
            'fire': {
                'fire_numbe': fire_numbe,
                'fire_date': fire.fire_date,
                'fire_size_ha': fire.fire_size_ha,
                'ml_area_ha': ml_area_ha,
                'ml_area_m2': ml_area_m2,
                'agreement_pct': fire.agreement_pct,
                'notes': fire.notes or '',
            },
            'run': {
                'timestamp': datetime.datetime.now().isoformat(
                    timespec='seconds'),
                'source': 'web',
            },
            'inputs': {
                'raster': state.raster_path,
                'perimeter_type': fire.perimeter_type,
            },
            'crop': {
                'padding': fire.padding_used,
                'width_px': fire.crop_w,
                'height_px': fire.crop_h,
                'total_px': fire.crop_w * fire.crop_h,
            },
            'sampling': {
                'sample_rate': state.sample_rate,
                'actual_sample_size': fire.sample_size,
            },
            'accumulation': {
                'start_date': fire.acc_start,
                'end_date': fire.acc_end,
            },
        }
        if fire.last_params:
            for section in ('tsne', 'hdbscan', 'random_forest'):
                if section in fire.last_params:
                    params_dict[section] = fire.last_params[section]

        path = os.path.join(fire_dir, f'{fire_numbe}_params.yaml')
        with open(path, 'w') as f:
            yaml.dump(params_dict, f,
                      default_flow_style=False, sort_keys=False)
    except ImportError:
        pass

    # Update fire_status.yaml
    try:
        import yaml
        status_path = os.path.join(state.output_root, 'fire_status.yaml')
        idx = {}
        if os.path.exists(status_path):
            with open(status_path) as f:
                idx = yaml.safe_load(f) or {}
        idx[fire_numbe] = {
            'status': 'accepted',
            'timestamp': datetime.datetime.now().isoformat(
                timespec='seconds'),
            'fire_dir': fire_dir,
            'source': 'web',
        }
        with open(status_path, 'w') as f:
            yaml.dump(idx, f, default_flow_style=False, sort_keys=True)
    except Exception:
        pass

    # Clean up XML artefacts
    for xml in glob.glob(os.path.join(fire_dir, '*.xml')):
        try:
            os.remove(xml)
        except Exception:
            pass

    # Append to accepted_params.csv for parameter learning
    try:
        import csv
        csv_path = os.path.join(state.output_root, 'accepted_params.csv')
        write_header = not os.path.isfile(csv_path)
        row_data = {
            'fire_numbe': fire_numbe,
            'fire_size_ha': fire.fire_size_ha,
            'agreement_pct': fire.agreement_pct,
            'padding': fire.padding_used,
            'timestamp': datetime.datetime.now().isoformat(
                timespec='seconds'),
        }
        if fire.last_params:
            for k, v in fire.last_params.items():
                row_data[k] = v
        with open(csv_path, 'a', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=row_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row_data)
    except Exception:
        pass

    fire.status = FireStatus.ACCEPTED
    fire.previously_accepted = False
    return fire_dir


# =========================================================================
# Build fire_mapping_cli.py command
# =========================================================================

def _build_mapping_cmd(fire: FireInfo, params: dict,
                       save_state: str = None,
                       load_state: str = None) -> list[str]:
    """Build the subprocess command for fire_mapping_cli.py."""
    rate = params.get('sample_rate')
    rate = float(rate) if rate is not None else state.sample_rate
    min_s = params.get('min_samples')
    min_s = int(min_s) if min_s is not None else state.min_samples
    max_s = params.get('max_samples')
    max_s = int(max_s) if max_s is not None else state.max_samples
    sample_size = int(round(fire.crop_w * fire.crop_h * rate))
    sample_size = max(min_s, min(max_s, sample_size))

    cmd = [
        sys.executable,
        state.cli_script,
        '--sample_size', str(sample_size),
        fire.crop_bin,
        fire.hint_bin,
        '--fire_numbe', fire.fire_numbe,
        '--start_date', fire.acc_start,
        '--end_date', fire.acc_end,
    ]

    if fire.perim_bin and os.path.exists(fire.perim_bin):
        cmd += ['--perimeter', fire.perim_bin]

    if save_state:
        cmd += ['--save_state', save_state]
    if load_state:
        cmd += ['--load_state', load_state]

    flag_map = {
        'seed': '--seed',
        'rf_n_estimators': '--rf_n_estimators',
        'rf_max_depth': '--rf_max_depth',
        'rf_max_features': '--rf_max_features',
        'rf_random_state': '--rf_random_state',
        'controlled_ratio': '--controlled_ratio',
        'hdbscan_min_samples': '--hdbscan_min_samples',
        'tsne_perplexity': '--tsne_perplexity',
        'tsne_learning_rate': '--tsne_learning_rate',
        'tsne_max_iter': '--tsne_max_iter',
        'tsne_init': '--tsne_init',
        'tsne_n_components': '--tsne_n_components',
        'tsne_random_state': '--tsne_random_state',
        'contour_width': '--contour_width',
    }

    for key, flag in flag_map.items():
        val = params.get(key)
        if val is not None and str(val).strip():
            # Argparse int args choke on "15.0" — normalise whole floats
            if isinstance(val, float) and val == int(val):
                val = int(val)
            cmd += [flag, str(val)]

    eb = params.get('embed_bands')
    if eb and str(eb).strip():
        cmd += ['--embed_bands', str(eb)]

    return cmd


# =========================================================================
# HTTP request handler
# =========================================================================

class FireHandler(BaseHTTPRequestHandler):
    """Routes all HTTP requests for the fire mapping web interface."""

    # -- Routing tables (compiled once) --
    ROUTES_GET = [
        (re.compile(r'^/$'), 'handle_fire_list'),
        (re.compile(r'^/login$'), 'handle_login_page'),
        (re.compile(r'^/logout$'), 'handle_logout'),
        (re.compile(r'^/admin$'), 'handle_admin_page'),
        (re.compile(r'^/fire/(?P<fire_numbe>[^/]+)$'), 'handle_fire_page'),
        (re.compile(r'^/api/fires$'), 'handle_api_fires'),
        (re.compile(r'^/api/settings$'), 'handle_api_settings_get'),
        (re.compile(r'^/api/access/status$'), 'handle_api_access_status'),
        (re.compile(r'^/api/batch/status$'), 'handle_api_batch_status'),
        (re.compile(r'^/api/admin/ips$'), 'handle_api_admin_ips'),
        (re.compile(r'^/api/admin/queue$'), 'handle_api_admin_queue'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/preview/(?P<view>[^/]+)$'),
         'handle_api_preview'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/comparison$'),
         'handle_api_comparison'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/brush_comparison$'),
         'handle_api_brush_comparison'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/status$'),
         'handle_api_status'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/console$'),
         'handle_api_console'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/ranked_params$'),
         'handle_api_ranked_params'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial_results$'),
         'handle_api_serial_results'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial/(?P<run_id>[0-9]+)/image$'),
         'handle_api_serial_image'),
        (re.compile(r'^/api/report$'), 'handle_api_report'),
        (re.compile(r'^/static/(?P<path>.+)$'), 'handle_static'),
    ]
    ROUTES_POST = [
        (re.compile(r'^/login$'), 'handle_login_post'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/prepare$'),
         'handle_api_prepare'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/accept$'),
         'handle_api_accept'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/map$'),
         'handle_api_map'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/remove$'),
         'handle_api_remove'),
        (re.compile(r'^/api/settings$'), 'handle_api_settings_post'),
        (re.compile(r'^/api/batch/map$'), 'handle_api_batch_map'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/notes$'),
         'handle_api_notes'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial_map$'),
         'handle_api_serial_map'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial/(?P<run_id>[0-9]+)/accept$'),
         'handle_api_serial_accept'),
        (re.compile(r'^/api/admin/ip/(?P<action>approve|block|revoke|unblock)$'),
         'handle_api_admin_ip_action'),
    ]

    # Paths that bypass IP checks (pending page needs CSS/logo + status poll)
    _IP_EXEMPT = {'/api/access/status', '/static/style.css',
                  '/static/BC-Wildfire-Service-logo.png'}
    # Paths that bypass ALL auth (login page, static assets for login)
    _NO_SESSION = {'/login', '/static/style.css',
                   '/static/BC-Wildfire-Service-logo.png'}

    def _route(self, routes):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        for pattern, handler_name in routes:
            m = pattern.match(path)
            if m:
                getattr(self, handler_name)(**m.groupdict())
                return True
        return False

    # ================================================================
    # Authentication & IP access control (cookie-based sessions)
    # ================================================================

    def _get_cookie(self, name: str) -> str:
        """Extract a cookie value from the Cookie header."""
        raw = self.headers.get('Cookie', '')
        for part in raw.split(';'):
            part = part.strip()
            if part.startswith(name + '='):
                return part[len(name) + 1:]
        return ''

    def _check_session(self) -> str | None:
        """Check session cookie. Returns role or None."""
        self._username = ''
        self._role = ''

        # No passwords configured → everyone is admin
        if not state.admin_password and not state.user_password:
            self._role = 'admin'
            return 'admin'

        token = self._get_cookie('session')
        if not token or token not in state.sessions:
            return None

        session = state.sessions[token]

        # Check expiry
        try:
            created = datetime.datetime.fromisoformat(session['created_at'])
            age = (datetime.datetime.now() - created).total_seconds()
            if age > _SESSION_MAX_AGE:
                del state.sessions[token]
                _save_sessions()
                return None
        except (KeyError, ValueError):
            del state.sessions[token]
            return None

        self._username = session.get('username', '')
        self._role = session.get('role', 'user')
        return self._role

    def _check_ip(self, role: str) -> bool:
        """Check IP access. Admins auto-approve. Returns True if allowed."""
        ip = self.client_address[0]
        username = getattr(self, '_username', '')

        if role == 'admin':
            # Auto-approve admin IPs
            if ip not in state.approved_ips:
                state.approved_ips[ip] = {
                    'username': username,
                    'role': 'admin',
                    'approved_by': 'auto (admin)',
                    'timestamp': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                }
                _save_ip_list()
            else:
                # Update username on each visit
                state.approved_ips[ip]['username'] = username
                state.approved_ips[ip]['role'] = 'admin'
            # Remove from pending if was there
            state.pending_ips.pop(ip, None)
            return True

        # Regular user — check IP
        if ip in state.blocked_ips:
            self._send_html(render_template('pending.html', {
                'ip': ip,
                'title': 'Access Denied',
                'message': 'Your IP address has been blocked by an '
                           'administrator.',
                'auto_refresh': 'false',
            }), 403)
            return False

        if ip in state.approved_ips:
            # Update username on each visit
            state.approved_ips[ip]['username'] = username
            state.approved_ips[ip]['role'] = 'user'
            return True

        # Unknown IP → pending
        now = datetime.datetime.now().isoformat(timespec='seconds')
        if ip not in state.pending_ips:
            state.pending_ips[ip] = {
                'username': username,
                'first_seen': now,
                'last_seen': now,
            }
        else:
            state.pending_ips[ip]['last_seen'] = now
            state.pending_ips[ip]['username'] = username

        self._send_html(render_template('pending.html', {
            'ip': ip,
            'title': 'Access Pending',
            'message': 'Your IP address has been registered. '
                       'An administrator will review your access request.',
            'auto_refresh': 'true',
        }))
        return False

    def _redirect(self, url, status=302):
        self.send_response(status)
        self.send_header('Location', url)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def _gate(self) -> str | None:
        """Full auth + IP gate. Returns role or None."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # Login page and static assets — no session needed
        if path in self._NO_SESSION or path.startswith('/static/'):
            self._role = ''
            self._username = ''
            return 'none'

        # Check session cookie
        role = self._check_session()
        if role is None:
            self._redirect('/login')
            return None

        # IP-exempt paths (access-status polling)
        if path in self._IP_EXEMPT:
            return role

        # IP access control
        if not self._check_ip(role):
            return None

        return role

    def do_GET(self):
        if self._gate() is None:
            return
        if not self._route(self.ROUTES_GET):
            self.send_error(404)

    def do_POST(self):
        if self._gate() is None:
            return
        # CSRF protection — exempt login form, require header on API calls
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != '/login':
            origin = self.headers.get('Origin', '')
            x_req = self.headers.get('X-Requested-With', '')
            if origin:
                host = self.headers.get('Host', '')
                allowed = {f'http://{host}', f'https://{host}'}
                if origin not in allowed:
                    self.send_error(403, 'Cross-origin request blocked')
                    return
            elif not x_req:
                self.send_error(403, 'Missing origin header')
                return
        if not self._route(self.ROUTES_POST):
            self.send_error(404)

    # -- Response helpers --

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath, media_type=None):
        if not os.path.isfile(filepath):
            self.send_error(404)
            return
        if media_type is None:
            media_type = (mimetypes.guess_type(filepath)[0]
                          or 'application/octet-stream')
        with open(filepath, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', media_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    _MAX_BODY = 1_000_000  # 1 MB

    def _read_body(self) -> dict:
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        if length > self._MAX_BODY:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- Page handlers --

    def handle_fire_list(self):
        admin_link = ('<a href="/admin" class="btn" '
                      'style="font-size:11px;padding:3px 10px">'
                      'Admin</a>'
                      if getattr(self, '_role', '') == 'admin' else '')
        html = render_template('fire_list.html', {
            'raster_name': os.path.basename(state.raster_path),
            'polygon_name': os.path.basename(state.polygon_file),
            'n_fires': str(len(state.fires)),
            'admin_link': admin_link,
        })
        self._send_html(html)

    def handle_login_page(self):
        # Already logged in? Redirect to home.
        token = self._get_cookie('session')
        if token and token in state.sessions:
            self._redirect('/')
            return
        html = render_template('login.html', {'error_msg': ''})
        self._send_html(html)

    def handle_login_post(self):
        import hmac
        import secrets
        # Parse form body (application/x-www-form-urlencoded)
        length = int(self.headers.get('Content-Length', 0))
        if length > 10000:
            self.send_error(400)
            return
        raw = self.rfile.read(length).decode(errors='replace')
        from urllib.parse import parse_qs
        form = parse_qs(raw)
        username = form.get('username', [''])[0].strip()
        password = form.get('password', [''])[0]

        role = None
        if (state.admin_password
                and hmac.compare_digest(password, state.admin_password)):
            role = 'admin'
        elif (state.user_password
              and hmac.compare_digest(password, state.user_password)):
            role = 'user'

        if role is None:
            html = render_template('login.html', {
                'error_msg': '<div class="error-msg" style="display:block">'
                             'Invalid password.</div>',
            })
            self._send_html(html, 401)
            return

        # Create session
        token = secrets.token_hex(32)
        state.sessions[token] = {
            'role': role,
            'username': username,
            'ip': self.client_address[0],
            'created_at': datetime.datetime.now().isoformat(
                timespec='seconds'),
        }
        _save_sessions()

        # Set cookie and redirect to home
        cookie = (f'session={token}; HttpOnly; SameSite=Strict; '
                  f'Path=/; Max-Age={_SESSION_MAX_AGE}')
        self.send_response(302)
        self.send_header('Location', '/')
        self.send_header('Set-Cookie', cookie)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def handle_logout(self):
        # Clear session
        token = self._get_cookie('session')
        if token and token in state.sessions:
            del state.sessions[token]
            _save_sessions()
        # Clear cookie and redirect to login
        self.send_response(302)
        self.send_header('Location', '/login')
        self.send_header('Set-Cookie',
                         'session=; HttpOnly; SameSite=Strict; '
                         'Path=/; Max-Age=0')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def handle_admin_page(self):
        if getattr(self, '_role', '') != 'admin':
            self.send_error(403, 'Admin access required')
            return
        html = render_template('admin.html', {})
        self._send_html(html)

    def handle_fire_page(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_html('Fire not found', 404)
            return
        fire = state.fires[fire_numbe]
        html = render_template('fire_mapping.html', {
            'fire_numbe': fire_numbe,
            'fire_numbe_json': json.dumps(fire_numbe),
            'fire_date': fire.fire_date,
            'fire_year': str(fire.fire_year),
            'fire_size_ha': str(fire.fire_size_ha),
            'fire_status': fire.status.value,
            'padding': str(state.padding),
            'sample_rate': str(state.sample_rate),
            'min_samples': str(state.min_samples),
            'max_samples': str(state.max_samples),
        })
        self._send_html(html)

    def handle_static(self, path):
        filepath = os.path.join(_HERE, 'static', path)
        if not os.path.abspath(filepath).startswith(
                os.path.join(_HERE, 'static')):
            self.send_error(403)
            return
        self._send_file(filepath)

    # -- API handlers --

    def handle_api_fires(self):
        fires = [
            {
                'fire_numbe': f.fire_numbe,
                'fire_date': f.fire_date,
                'fire_year': f.fire_year,
                'fire_size_ha': f.fire_size_ha,
                'status': f.status.value,
                'previously_accepted': f.previously_accepted,
                'agreement_pct': f.agreement_pct,
                'ml_area_ha': f.ml_area_ha,
                'notes': f.notes,
            }
            for f in state.fires.values()
            if not f.hidden
        ]
        self._send_json(fires)

    def handle_api_settings_get(self):
        self._send_json(state.recommended_settings)

    def handle_api_settings_post(self):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        body = self._read_body()
        settings = body.get('settings', [])
        # Sanity-check: embed_bands must be a string, not numeric
        for row in settings:
            p = row.get('params', {})
            eb = p.get('embed_bands')
            if eb is not None and isinstance(eb, (int, float)):
                sys.stderr.write(
                    f'[settings] WARNING: embed_bands was numeric '
                    f'({eb!r}) — converting to empty string\n')
                p['embed_bands'] = ''
        state.recommended_settings = settings
        self._send_json({'status': 'saved'})

    # -- Batch mapping API --

    def handle_api_batch_map(self):
        global _batch_thread
        if (state.batch_status
                and state.batch_status.get('running')):
            self._send_json(
                {'error': 'A batch is already running'}, 400)
            return
        body = self._read_body()
        fire_numbes = body.get('fire_numbes', [])
        # Filter out accepted fires
        fire_numbes = [
            fn for fn in fire_numbes
            if fn in state.fires
            and state.fires[fn].status != FireStatus.ACCEPTED
        ]
        if not fire_numbes:
            self._send_json({'error': 'No eligible fires selected'}, 400)
            return
        _batch_thread = threading.Thread(
            target=_batch_map_worker,
            args=(fire_numbes,),
            daemon=True)
        _batch_thread.start()
        self._send_json({
            'status': 'started',
            'total': len(fire_numbes),
        })

    def handle_api_batch_status(self):
        self._send_json(
            state.batch_status or {'running': False})

    # -- Access control & admin API --

    def handle_api_access_status(self):
        """Called by the pending page to check if IP was approved."""
        ip = self.client_address[0]
        if ip in state.approved_ips:
            self._send_json({'status': 'approved'})
        elif ip in state.blocked_ips:
            self._send_json({'status': 'blocked'})
        else:
            self._send_json({'status': 'pending'})

    def handle_api_admin_ips(self):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        self._send_json({
            'approved': state.approved_ips,
            'blocked': state.blocked_ips,
            'pending': state.pending_ips,
        })

    def handle_api_admin_queue(self):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        self._send_json({
            'current': state.current_job,
            'waiting': list(state.waiting_jobs),
        })

    def handle_api_admin_ip_action(self, action):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        body = self._read_body()
        ip = body.get('ip', '').strip()
        if not ip:
            self._send_json({'error': 'No IP provided'}, 400)
            return

        now = datetime.datetime.now().isoformat(timespec='seconds')

        if action == 'approve':
            # Preserve username from pending entry
            pending_info = state.pending_ips.get(ip, {})
            state.approved_ips[ip] = {
                'username': pending_info.get('username', ''),
                'role': 'user',
                'approved_by': self.client_address[0],
                'timestamp': now,
            }
            state.pending_ips.pop(ip, None)
            state.blocked_ips.pop(ip, None)

        elif action == 'block':
            pending_info = state.pending_ips.get(ip, {})
            approved_info = state.approved_ips.get(ip, {})
            state.blocked_ips[ip] = {
                'username': (pending_info.get('username', '')
                             or approved_info.get('username', '')),
                'blocked_by': self.client_address[0],
                'timestamp': now,
            }
            state.approved_ips.pop(ip, None)
            state.pending_ips.pop(ip, None)

        elif action == 'revoke':
            state.approved_ips.pop(ip, None)

        elif action == 'unblock':
            state.blocked_ips.pop(ip, None)

        _save_ip_list()
        self._send_json({'status': 'ok'})

    # -- Fire API --

    def handle_api_prepare(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return

        fire = state.fires[fire_numbe]
        body = self._read_body()
        padding = body.get('padding')

        prev_status = fire.status

        needs_prepare = fire.status in (
            FireStatus.PENDING, FireStatus.ERROR)
        # Re-prepare accepted/mapped fires with no previews yet
        if not fire.available_views and fire.status in (
                FireStatus.ACCEPTED, FireStatus.MAPPED):
            needs_prepare = True
        padding_changed = (padding is not None
                           and fire.padding_used != float(padding))
        if padding_changed:
            needs_prepare = True
        if fire.status == FireStatus.PREPARING:
            self._send_json({'status': 'preparing'})
            return

        if needs_prepare:
            with _gpu_lock:
                _prepare_fire_sync(fire_numbe, padding)
            fire = state.fires[fire_numbe]
            if fire.status == FireStatus.READY and prev_status in (
                    FireStatus.ACCEPTED, FireStatus.MAPPED):
                if padding_changed:
                    # User changed params — treat as fresh start
                    fire.previously_accepted = (
                        prev_status == FireStatus.ACCEPTED)
                    fire.last_comparison = ''
                else:
                    # Initial page load — just regenerating previews
                    fire.status = prev_status

        fire = state.fires[fire_numbe]
        if fire.status == FireStatus.ERROR:
            self._send_json(
                {'status': 'error', 'error': fire.error_msg}, 500)
            return

        # Check for comparison images (cache dir only for re-prepared
        # fires; cache + canonical for untouched accepted fires)
        has_comparison = False
        has_brush = False
        canonical = os.path.join(state.output_root, fire_numbe)

        if fire.last_comparison and os.path.isfile(fire.last_comparison):
            has_comparison = True
        elif not fire.previously_accepted:
            for d in (fire.cache_dir, canonical):
                if not d:
                    continue
                comp = os.path.join(d, f'{fire_numbe}_comparison.png')
                if os.path.isfile(comp):
                    fire.last_comparison = comp
                    has_comparison = True
                    break

        if not fire.previously_accepted:
            for d in (fire.cache_dir, canonical):
                if not d:
                    continue
                brush = os.path.join(
                    d, f'{fire_numbe}_brush_comparison.png')
                if os.path.isfile(brush):
                    has_brush = True
                    break

        self._send_json({
            'status': fire.status.value,
            'views': fire.available_views,
            'crop_w': fire.crop_w,
            'crop_h': fire.crop_h,
            'sample_size': fire.sample_size,
            'perimeter_type': fire.perimeter_type,
            'acc_start': fire.acc_start,
            'acc_end': fire.acc_end,
            'has_comparison': has_comparison,
            'has_brush_comparison': has_brush,
            'previously_accepted': fire.previously_accepted,
            'ml_area_ha': fire.ml_area_ha,
        })

    def handle_api_preview(self, fire_numbe, view):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        # Validate view name — alphanumeric/underscore/hyphen only
        if not re.fullmatch(r'[A-Za-z0-9_-]+', view):
            self.send_error(400, 'Invalid view name')
            return
        fire = state.fires[fire_numbe]
        png = os.path.join(fire.cache_dir, 'previews', f'{view}.png')

        # On-the-fly generation for serial overlays
        if not os.path.exists(png):
            m = re.match(r'^serial_(\d+)$', view)
            if m:
                rid = m.group(1)
                # Try multiple naming patterns for the classified raster
                for _pat in (f'{fire_numbe}_serial_{rid}_classified.bin',
                             f'{fire_numbe}_crop.bin_classified.bin'):
                    _cand = os.path.join(fire.cache_dir, _pat)
                    if os.path.isfile(_cand):
                        _overlay_mask_on_post(
                            fire, _cand, view, (0.9, 0.1, 0.0))
                        if os.path.exists(png):
                            break

        if not os.path.exists(png):
            self._send_json(
                {'error': f"Preview '{view}' not available"}, 404)
            return
        self._send_file(png, 'image/png')

    def handle_api_comparison(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        path = None
        if fire.last_comparison and os.path.isfile(fire.last_comparison):
            path = fire.last_comparison
        else:
            for d in (fire.cache_dir,
                      os.path.join(state.output_root, fire_numbe)):
                if not d:
                    continue
                candidate = os.path.join(
                    d, f'{fire_numbe}_comparison.png')
                if os.path.isfile(candidate):
                    path = candidate
                    break
        if not path:
            self._send_json({'error': 'No comparison available'}, 404)
            return
        self._send_file(path, 'image/png')

    def handle_api_brush_comparison(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        path = None
        for d in (fire.cache_dir,
                  os.path.join(state.output_root, fire_numbe)):
            if not d:
                continue
            candidate = os.path.join(
                d, f'{fire_numbe}_brush_comparison.png')
            if os.path.isfile(candidate):
                path = candidate
                break
        if not path:
            self._send_json(
                {'error': 'No brush comparison available'}, 404)
            return
        self._send_file(path, 'image/png')

    def handle_api_accept(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if fire.status != FireStatus.MAPPED:
            self._send_json(
                {'error': f'Cannot accept: status is {fire.status.value}'},
                400)
            return
        _accept_fire_sync(fire_numbe)
        self._send_json({'status': 'accepted'})

    def handle_api_status(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        f = state.fires[fire_numbe]
        self._send_json({'status': f.status.value, 'error': f.error_msg})

    def handle_api_console(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        f = state.fires[fire_numbe]

        # Check for comparison images
        has_comparison = bool(
            f.last_comparison and os.path.isfile(f.last_comparison))
        has_brush = False
        if not has_comparison:
            canonical = os.path.join(state.output_root, fire_numbe)
            for d in (f.cache_dir, canonical):
                if not d:
                    continue
                if os.path.isfile(
                        os.path.join(d, f'{fire_numbe}_comparison.png')):
                    has_comparison = True
                    break
        for d in (f.cache_dir,
                  os.path.join(state.output_root, fire_numbe)):
            if not d:
                continue
            if os.path.isfile(
                    os.path.join(
                        d, f'{fire_numbe}_brush_comparison.png')):
                has_brush = True
                break

        # Clean serial results for JSON
        serial_results = []
        for r in f.serial_results:
            serial_results.append({
                'run_id': r.get('run_id'),
                'agreement_pct': r.get('agreement_pct', -1),
                'ml_area_ha': r.get('ml_area_ha', -1),
                'error': r.get('error', ''),
                'params': r.get('params', {}),
                'is_previous': r.get('is_previous', False),
            })

        self._send_json({
            'status': f.status.value,
            'previously_accepted': f.previously_accepted,
            'lines': f.console_log,
            'last_params': f.last_params,
            'agreement_pct': f.agreement_pct,
            'ml_area_ha': f.ml_area_ha,
            'available_views': list(f.available_views),
            'serial_results': serial_results,
            'has_comparison': has_comparison,
            'has_brush_comparison': has_brush,
        })

    # -- Serial mapping & parameter ranking API --

    def handle_api_ranked_params(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        parsed = urlparse(self.path)
        from urllib.parse import parse_qs
        qs = parse_qs(parsed.query)
        n = int(qs.get('n', ['3'])[0])
        n = max(1, min(10, n))

        ranked = _rank_params_for_fire(
            fire_numbe, fire.fire_size_ha, n)

        region, zone = _parse_fire_context(fire_numbe)
        bucket = _size_bucket(fire.fire_size_ha)
        csv_rows = _load_accepted_params()
        total_accepted = len(csv_rows)

        self._send_json({
            'ranked': ranked,
            'context': {
                'region': region,
                'zone': zone,
                'size_bucket': list(bucket),
                'total_accepted': total_accepted,
            },
        })

    def handle_api_serial_map(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if fire.status == FireStatus.MAPPING:
            self._send_json({'error': 'Already mapping'}, 400)
            return

        body = self._read_body()
        n = int(body.get('n', 3))
        n = max(1, min(10, n))
        user_base = body.get('base_params')  # from "Map Fire" with N>1

        if user_base:
            # User supplied explicit params — use as run 1,
            # vary HDBSCAN for runs 2-N
            base = dict(user_base)
        else:
            # No explicit params — use ranked/recommended as base
            ranked = _rank_params_for_fire(
                fire_numbe, fire.fire_size_ha, 1)
            if not ranked:
                self._send_json(
                    {'error': 'No parameter sets available'}, 400)
                return

            # Validate key params to catch corruption early
            base_check = ranked[0]
            eb = base_check.get('embed_bands')
            tp = base_check.get('tsne_perplexity')
            if eb is not None and isinstance(eb, (int, float)):
                sys.stderr.write(
                    f'[serial] WARNING: embed_bands is numeric '
                    f'({eb!r}), expected comma-separated string '
                    f'— reloading from YAML\n')
                sys.stderr.flush()
                fresh = _get_recommended_params(fire.fire_size_ha)
                if fresh:
                    ranked = [fresh]
            if (tp is not None and isinstance(tp, str)
                    and ',' in str(tp)):
                sys.stderr.write(
                    f'[serial] WARNING: tsne_perplexity is string '
                    f'({tp!r}), expected number '
                    f'— reloading from YAML\n')
                sys.stderr.flush()
                fresh = _get_recommended_params(fire.fire_size_ha)
                if fresh:
                    ranked = [fresh]
            base = dict(ranked[0])

        # Generate N param sets: same base, vary HDBSCAN params.
        # Run 1 = base params.  Runs 2-N get varied
        # hdbscan_min_samples values.
        base_hms = int(base.get('hdbscan_min_samples', 20))
        param_sets = [base]

        # Generate hdbscan_min_samples variations around base
        variations = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        variations = sorted(
            [v for v in variations if v != base_hms],
            key=lambda v: abs(v - base_hms))
        for v in variations:
            if len(param_sets) >= n:
                break
            variant = dict(base)
            variant['hdbscan_min_samples'] = v
            param_sets.append(variant)

        param_sets = param_sets[:n]

        # Set status BEFORE starting thread to avoid race
        if fire.status == FireStatus.ACCEPTED:
            fire.previously_accepted = True
        fire.status = FireStatus.MAPPING
        fire.serial_results = []
        fire.console_log = []

        thread = threading.Thread(
            target=_serial_map_worker,
            args=(fire_numbe, param_sets),
            daemon=True)
        thread.start()

        self._send_json({
            'status': 'started',
            'n': len(param_sets),
        })

    def handle_api_serial_results(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        results = []
        for r in fire.serial_results:
            results.append({
                'run_id': r.get('run_id'),
                'agreement_pct': r.get('agreement_pct', -1),
                'ml_area_ha': r.get('ml_area_ha', -1),
                'error': r.get('error', ''),
                'params': r.get('params', {}),
                'is_previous': r.get('is_previous', False),
            })
        self._send_json({
            'status': fire.status.value,
            'results': results,
        })

    def handle_api_serial_image(self, fire_numbe, run_id):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]

        # Check requested image type
        qs = parse_qs(urlparse(self.path).query)
        img_type = (qs.get('type', ['']) or [''])[0]

        if img_type == 'comparison':
            comp_path = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_{run_id}.png')
            if os.path.isfile(comp_path):
                self._send_file(comp_path, 'image/png')
                return
            self._send_json(
                {'error': 'Comparison not found for this run'}, 404)
            return

        if img_type == 'brush':
            brush_path = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_{run_id}_brush.png')
            if os.path.isfile(brush_path):
                self._send_file(brush_path, 'image/png')
                return
            self._send_json(
                {'error': 'Brush comparison not found for this run'},
                404)
            return

        # Default: serve pixel-aligned overlay
        overlay_path = os.path.join(
            fire.cache_dir, 'previews', f'serial_{run_id}.png')
        if not os.path.isfile(overlay_path):
            serial_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_{run_id}_classified.bin')
            if os.path.isfile(serial_clf):
                _overlay_mask_on_post(
                    fire, serial_clf, f'serial_{run_id}',
                    (0.9, 0.1, 0.0))
        if os.path.isfile(overlay_path):
            self._send_file(overlay_path, 'image/png')
            return
        # Fall back to comparison figure
        comp_path = os.path.join(
            fire.cache_dir, f'{fire_numbe}_serial_{run_id}.png')
        if os.path.isfile(comp_path):
            self._send_file(comp_path, 'image/png')
            return
        self._send_json({'error': 'Image not found'}, 404)

    def handle_api_serial_accept(self, fire_numbe, run_id):
        fire_numbe = unquote(fire_numbe)
        run_id = int(run_id)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        result = next(
            (r for r in fire.serial_results
             if r.get('run_id') == run_id), None)
        if not result:
            self._send_json({'error': 'Run not found'}, 404)
            return

        # Copy the selected run's results as the main results
        serial_clf = result.get('classified', '')
        serial_comp = result.get('comparison', '')
        if serial_clf and os.path.isfile(serial_clf):
            main_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_crop.bin_classified.bin')
            shutil.copy2(serial_clf, main_clf)
            ser_hdr = (
                os.path.splitext(serial_clf)[0] + '.hdr')
            if not os.path.isfile(ser_hdr):
                ser_hdr = serial_clf + '.hdr'
            if os.path.isfile(ser_hdr):
                shutil.copy2(
                    ser_hdr,
                    os.path.splitext(main_clf)[0] + '.hdr')
        if serial_comp and os.path.isfile(serial_comp):
            main_comp = os.path.join(
                fire.cache_dir, f'{fire_numbe}_comparison.png')
            shutil.copy2(serial_comp, main_comp)
            fire.last_comparison = main_comp

        fire.agreement_pct = result.get('agreement_pct', -1)
        fire.last_params = result.get('params', {})
        fire.status = FireStatus.MAPPED

        # Now accept via the normal flow
        _accept_fire_sync(fire_numbe)
        self._send_json({'status': 'accepted'})

    def handle_api_notes(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        body = self._read_body()
        state.fires[fire_numbe].notes = body.get('notes', '')
        _save_notes()
        self._send_json({'status': 'saved'})

    def handle_api_report(self):
        """Generate PDF report of selected accepted fires."""
        from batch_fire_mapping.generate_report import generate_report
        import tempfile

        # Get selected fire numbers from query params
        parsed = urlparse(self.path)
        from urllib.parse import parse_qs
        qs = parse_qs(parsed.query)
        fire_list = qs.get('fire', [])

        if not fire_list:
            self._send_json(
                {'error': 'No fires specified'}, 400)
            return

        # Create temp dir with symlinks to selected fire dirs
        tmp_dir = tempfile.mkdtemp(prefix='fire_report_sel_')
        try:
            for fn in fire_list:
                src = os.path.join(state.output_root, fn)
                if os.path.isdir(src):
                    dst = os.path.join(tmp_dir, fn)
                    os.symlink(os.path.abspath(src), dst)

            tmp_pdf = os.path.join(tmp_dir, 'report.pdf')
            pdf = generate_report(tmp_dir, tmp_pdf)
            if pdf is None or not os.path.isfile(pdf):
                self._send_json(
                    {'error': 'Report generation failed'}, 500)
                return

            with open(pdf, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Content-Disposition',
                             'attachment; filename="fire_report.pdf"')
            self.end_headers()
            self.wfile.write(data)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def handle_api_remove(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        state.fires[fire_numbe].hidden = True
        self._send_json({'status': 'removed'})

    # -- Mapping with SSE streaming --

    def handle_api_map(self, fire_numbe):
        """Run fire_mapping_cli.py and stream output as SSE."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return

        fire = state.fires[fire_numbe]
        body = self._read_body()
        params = body.get('params', {})

        if fire.status not in (
                FireStatus.READY, FireStatus.MAPPED, FireStatus.ACCEPTED):
            self._send_json({
                'error': f'Fire not ready (status: {fire.status.value})',
            }, 400)
            return

        # Begin SSE response (chunked — no Content-Length)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()

        # Clear console log for fresh mapping
        fire.console_log = []

        def sse(event_type, data):
            payload = json.dumps({'type': event_type, **data})
            # Buffer for reconnection
            if event_type == 'log':
                fire.console_log.append(data.get('message', ''))
            try:
                self.wfile.write(f'data: {payload}\n\n'.encode())
                self.wfile.flush()
            except BrokenPipeError:
                pass

        # GPU serialisation with queue tracking
        client_ip = self.client_address[0]
        job_entry = {
            'fire_numbe': fire_numbe,
            'client_ip': client_ip,
            'queued_at': datetime.datetime.now().isoformat(
                timespec='seconds'),
        }

        with _gpu_queue_lock:
            queue_pos = _gpu_queue
            globals()['_gpu_queue'] = _gpu_queue + 1
            state.waiting_jobs.append(job_entry)

        if _gpu_lock.locked():
            sse('log', {
                'message': f'Queued — {queue_pos} job(s) ahead. '
                           f'Waiting for GPU...',
            })

        try:
            with _gpu_lock:
                # Move from waiting to current
                if job_entry in state.waiting_jobs:
                    state.waiting_jobs.remove(job_entry)
                state.current_job = {
                    **job_entry,
                    'started_at': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                }
                if fire.status == FireStatus.ACCEPTED:
                    fire.previously_accepted = True
                fire.status = FireStatus.MAPPING
                cmd = _build_mapping_cmd(fire, params)
                short_cmd = ' '.join(
                    os.path.basename(c) if i < 3 else c
                    for i, c in enumerate(cmd))
                sse('log', {'message': f'$ {short_cmd}'})

                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=state.project_root,
                    )

                    for raw_line in iter(proc.stdout.readline, b''):
                        text = raw_line.decode(errors='replace').rstrip()
                        if text:
                            sse('log', {'message': text})

                    rc = proc.wait()

                except Exception as exc:
                    fire.status = FireStatus.READY
                    sse('error', {
                        'message': f'Failed to start subprocess: {exc}',
                    })
                    return

                if rc == 0:
                    comp = os.path.join(
                        fire.cache_dir, f'{fire_numbe}_comparison.png')
                    fire.last_comparison = (
                        comp if os.path.exists(comp) else '')
                    fire.last_params = params
                    fire.agreement_pct = _compute_agreement(fire)
                    fire.ml_area_ha = _compute_ml_area(fire)
                    _generate_result_preview(fire)
                    fire.status = FireStatus.MAPPED
                    sse('complete', {
                        'comparison_url': (
                            f'/api/fire/{fire_numbe}/comparison'
                            f'?t={int(time.time())}'),
                        'agreement_pct': fire.agreement_pct,
                        'ml_area_ha': fire.ml_area_ha,
                    })
                else:
                    fire.status = FireStatus.READY
                    sse('error', {
                        'message': (
                            f'fire_mapping_cli.py exited with code {rc}'),
                    })
        finally:
            with _gpu_queue_lock:
                globals()['_gpu_queue'] = max(0, _gpu_queue - 1)
            state.current_job = None
            if job_entry in state.waiting_jobs:
                state.waiting_jobs.remove(job_entry)

    # -- Logging --

    def log_message(self, format, *args):
        # Print only non-static requests to keep terminal clean
        msg = format % args
        if '/static/' not in msg:
            sys.stderr.write(
                f'[{self.log_date_time_string()}] {msg}\n')


# =========================================================================
# Server factory
# =========================================================================

def create_server(host: str = '0.0.0.0',
                  port: int = 8765) -> ThreadingHTTPServer:
    """Create and return the threaded HTTP server."""
    server = ThreadingHTTPServer((host, port), FireHandler)
    return server
