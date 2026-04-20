"""Admin-only parameter analyzer web UI.

Phase A: skeleton. Registers routes via monkey-patching FireHandler so
that batch_fire_mapping_web/app.py remains untouched. All analyzer state
lives in AnalyzerState (analyzer_state.py) and is attached to AppState
as ``app_state.analyzer``.

Output layout (all under <out_dir>/analyzing_parameters/):

    analyzer_config.yaml         admin's current config (N sets, M runs, fires)
    analyzer_accepted.csv        master CSV — one row per accepted run
    <FIRE>/                      per-fire canonical dir (only exists after accept)
        <FIRE>_crop_max.bin/.hdr biggest-padding crop backdrop
        manifest.yaml            {saved_padding, accepts: [run_0001, ...]}
        run_0001/                one accepted run's full outputs
            params.yaml
            <FIRE>_classified.bin/.hdr
            <FIRE>_comparison.png
            thumb.png
    .analyzer_cache/<FIRE>/      per-fire working cache (all N*M runs)
        tsne_rf_<hash>.npz       cached t-SNE+RF intermediate state
        set_{S}_run_{R}/
            params.yaml
            agreement.json
            thumb.png
"""

import os
import re
import sys
from urllib.parse import unquote

from .analyzer_state import AnalyzerState, AnalyzerFireInfo, AnalyzerStatus


# Module-level analyzer state — set by init_analyzer()
_astate: AnalyzerState = None


# =========================================================================
# Initialization
# =========================================================================

def init_analyzer(app_state):
    """Create analyzer state, wire paths, load config + scan disk.

    Called from __main__.py after AppState is populated. Attaches the
    resulting AnalyzerState to ``app_state.analyzer``.
    """
    global _astate
    _astate = AnalyzerState()

    root = os.path.join(app_state.output_root, 'analyzing_parameters')
    _astate.analyzer_root = root
    _astate.cache_root = os.path.join(root, '.analyzer_cache')
    _astate.config_file = os.path.join(root, 'analyzer_config.yaml')
    _astate.csv_file = os.path.join(root, 'analyzer_accepted.csv')

    os.makedirs(_astate.analyzer_root, exist_ok=True)
    os.makedirs(_astate.cache_root, exist_ok=True)

    _load_config()
    _scan_canonical_dir(app_state)

    app_state.analyzer = _astate

    sys.stderr.write(
        f'[analyzer] Initialized. root={_astate.analyzer_root} '
        f'fires_with_runs={len(_astate.fires)} '
        f'param_sets={len(_astate.config.param_sets)} '
        f'selected_fires={len(_astate.config.selected_fires)}\n')
    sys.stderr.flush()


def _load_config():
    """Load analyzer_config.yaml into the AnalyzerConfig."""
    if not os.path.isfile(_astate.config_file):
        return
    try:
        import yaml
        with open(_astate.config_file) as f:
            data = yaml.safe_load(f) or {}
        _astate.config.param_sets = list(data.get('param_sets', []))
        _astate.config.m_runs_per_set = int(data.get('m_runs_per_set', 3))
        _astate.config.m_run_jitter = int(data.get('m_run_jitter', 1))
        _astate.config.selected_fires = list(data.get('selected_fires', []))
        _astate.config.description = str(data.get('description', ''))
    except Exception as exc:
        sys.stderr.write(
            f'[analyzer] WARNING: Failed to load config: {exc}\n')
        sys.stderr.flush()


def _save_config():
    """Persist AnalyzerConfig atomically."""
    try:
        from .app import _atomic_yaml_dump
        data = {
            'param_sets': _astate.config.param_sets,
            'm_runs_per_set': _astate.config.m_runs_per_set,
            'm_run_jitter': _astate.config.m_run_jitter,
            'selected_fires': _astate.config.selected_fires,
            'description': _astate.config.description,
        }
        _atomic_yaml_dump(_astate.config_file, data, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[analyzer] WARNING: Failed to save config: {exc}\n')
        sys.stderr.flush()


def _scan_canonical_dir(app_state):
    """Rebuild analyzer fire state from analyzing_parameters/<FIRE>/ dirs.

    For each fire directory found, read its manifest.yaml and populate
    an AnalyzerFireInfo with the accepted runs. Also reads each
    run_####/params.yaml to reconstruct the AnalyzerRun entries.
    """
    if not os.path.isdir(_astate.analyzer_root):
        return

    import yaml
    from .analyzer_state import AnalyzerRun

    for fn in os.listdir(_astate.analyzer_root):
        fire_dir = os.path.join(_astate.analyzer_root, fn)
        if not os.path.isdir(fire_dir):
            continue
        # Skip dotfiles (.analyzer_cache) and non-fire metadata files.
        if fn.startswith('.') or fn.startswith('analyzer_'):
            continue
        if fn not in app_state.fires:
            # Fire not in the loaded shapefile — skip but leave dir intact.
            continue

        info = AnalyzerFireInfo(
            fire_numbe=fn,
            canonical_dir=fire_dir,
            cache_dir=os.path.join(_astate.cache_root, fn),
        )

        # Load manifest (biggest-padding backdrop + accept list).
        manifest_path = os.path.join(fire_dir, 'manifest.yaml')
        accept_ids = []
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path) as f:
                    m = yaml.safe_load(f) or {}
                info.saved_max_padding = float(m.get('saved_padding', 0.0))
                info.saved_max_crop_rel = str(m.get('saved_crop', ''))
                # Only accept values matching the canonical 'run_NNNN'
                # shape — a corrupted or crafted manifest cannot inject
                # path segments like '../../etc' into the joins below.
                raw_ids = list(m.get('accepts', []))
                accept_ids = [str(a) for a in raw_ids
                              if isinstance(a, str)
                              and re.fullmatch(r'run_\d{4,}', a)]
                if len(accept_ids) != len(raw_ids):
                    sys.stderr.write(
                        f'[analyzer] WARNING: dropped '
                        f'{len(raw_ids) - len(accept_ids)} malformed '
                        f'accept id(s) from {manifest_path}\n')
            except Exception as exc:
                sys.stderr.write(
                    f'[analyzer] WARNING: Bad manifest for {fn}: {exc}\n')

        # Reconstruct runs from each run_#### subfolder's params.yaml.
        for aid in accept_ids:
            sub = os.path.join(fire_dir, aid)
            params_path = os.path.join(sub, 'params.yaml')
            if not os.path.isfile(params_path):
                continue
            try:
                with open(params_path) as f:
                    pdata = yaml.safe_load(f) or {}
            except Exception:
                continue
            grid = pdata.get('grid', {})
            params = pdata.get('params', {})
            outcome = pdata.get('outcome', {})
            run = AnalyzerRun(
                set_idx=int(grid.get('set_idx', -1)),
                run_idx=int(grid.get('run_idx', -1)),
                params=dict(params),
                padding_used=float(params.get('padding', 0.0)),
                agreement_pct=float(
                    outcome.get('agreement_pct', -1.0) or -1.0),
                ml_area_ha=float(
                    outcome.get('ml_area_ha', -1.0) or -1.0),
                status='done',
                accepted=True,
                accept_id=aid,
                accepted_at=str(pdata.get('accepted_at', '')),
            )
            info.runs.append(run)

        if info.runs:
            info.status = AnalyzerStatus.ANALYZED

        _astate.fires[fn] = info

    # Also scan .analyzer_cache for un-accepted runs (partial work). This
    # merges them into the runs list of each AnalyzerFireInfo so the
    # gallery can show completed runs even after a cancel/crash.
    from .analyzer_worker import scan_cache_for_fire
    if os.path.isdir(_astate.cache_root):
        for fn in os.listdir(_astate.cache_root):
            full = os.path.join(_astate.cache_root, fn)
            if not os.path.isdir(full) or fn not in app_state.fires:
                continue
            cached = scan_cache_for_fire(fn, _astate)
            if not cached:
                continue
            info = _astate.fires.get(fn)
            if info is None:
                info = AnalyzerFireInfo(
                    fire_numbe=fn,
                    cache_dir=full,
                    canonical_dir=os.path.join(_astate.analyzer_root, fn),
                )
                _astate.fires[fn] = info
            # De-dup: skip cached runs that already have an accepted twin
            # at the same (set_idx, run_idx).
            existing = {(r.set_idx, r.run_idx) for r in info.runs}
            for r in cached:
                if (r.set_idx, r.run_idx) not in existing:
                    info.runs.append(r)
            if info.status == AnalyzerStatus.PENDING and info.runs:
                info.status = AnalyzerStatus.PARTIAL


# =========================================================================
# HTML helpers (reuse existing template engine from app.py)
# =========================================================================

def _require_admin(handler):
    """Return True if request is from an admin; send 403 otherwise."""
    if getattr(handler, '_role', '') != 'admin':
        handler.send_error(403, 'Admin only')
        return False
    return True


# =========================================================================
# Page handlers (monkey-patched onto FireHandler via register_routes)
# =========================================================================

def handle_analyzer_page(self):
    """Render the main analyzer page (admin-only)."""
    if not _require_admin(self):
        return
    from .app import render_template, state as app_state
    html = render_template('analyzer.html', {
        'raster_name': os.path.basename(app_state.raster_path),
        'polygon_name': os.path.basename(app_state.polygon_file),
        'analyzer_root': _astate.analyzer_root,
    })
    self._send_html(html)


# =========================================================================
# API handlers
# =========================================================================

def handle_api_analyzer_status(self):
    """Return overall analyzer state snapshot for the UI.

    Deliberately small: the per-fire details are fetched via a
    separate fires endpoint. This one is cheap to poll.
    """
    if not _require_admin(self):
        return
    with _astate.lock:
        running = _astate.running
        batch = dict(_astate.batch_status) if _astate.batch_status else None
    self._send_json({
        'running': running,
        'batch': batch,
        'root': _astate.analyzer_root,
        'm_runs_per_set': _astate.config.m_runs_per_set,
        'n_param_sets': len(_astate.config.param_sets),
        'n_selected_fires': len(_astate.config.selected_fires),
        'n_fires_with_data': len(_astate.fires),
    })


# =========================================================================
# Config API (read/write analyzer_config.yaml)
# =========================================================================

# Full list of pipeline parameters we expose for analyzer param sets.
# Order matters — it controls rendering order in the UI.
_PARAM_KEYS = (
    'padding',
    'sample_rate', 'min_samples', 'max_samples', 'seed',
    'embed_bands',
    'tsne_perplexity', 'tsne_learning_rate', 'tsne_max_iter',
    'tsne_init', 'tsne_n_components', 'tsne_random_state',
    'rf_n_estimators', 'rf_max_depth', 'rf_max_features',
    'rf_random_state',
    'controlled_ratio', 'hdbscan_min_samples',
    'contour_width',
)


def _validate_param_set(raw: dict) -> tuple:
    """Validate a single param set. Returns (cleaned_dict, error_or_None)."""
    from .app import _validate_param, _validate_embed_bands
    cleaned: dict = {}
    # Name is optional — used purely for UI grouping.
    if raw.get('name'):
        cleaned['name'] = str(raw['name'])[:80]

    for key in _PARAM_KEYS:
        val = raw.get(key)
        if val is None or val == '':
            continue
        try:
            if key == 'embed_bands':
                eb = _validate_embed_bands(val)
                if eb:
                    cleaned[key] = eb
            elif key == 'padding':
                v = float(val)
                if not (0.0 <= v <= 5.0):
                    return (None, f'padding out of range [0, 5]: {val}')
                cleaned[key] = v
            else:
                cleaned[key] = _validate_param(key, val)
        except (ValueError, TypeError) as exc:
            return (None, f'{key}: {exc}')
    return (cleaned, None)


def handle_api_analyzer_config_get(self):
    """Return the current analyzer config + a seed template for new sets."""
    if not _require_admin(self):
        return
    from .app import state as app_state
    # Seed: the recommended settings the admin can copy from.
    seeds = []
    for tier in app_state.recommended_settings:
        seeds.append({
            'label': str(tier.get('label', '') or 'setting'),
            'params': dict(tier.get('params', {})),
        })
    self._send_json({
        'config': {
            'param_sets': list(_astate.config.param_sets),
            'm_runs_per_set': _astate.config.m_runs_per_set,
            'm_run_jitter': _astate.config.m_run_jitter,
            'selected_fires': list(_astate.config.selected_fires),
            'description': _astate.config.description,
        },
        'seeds': seeds,
        'param_keys': list(_PARAM_KEYS),
    })


def handle_api_analyzer_config_post(self):
    """Persist the admin's analyzer config after validation."""
    if not _require_admin(self):
        return
    body = self._read_body()
    if body is None:
        return

    # --- m_runs_per_set ---
    try:
        m = int(body.get('m_runs_per_set', 3))
    except (ValueError, TypeError):
        self._send_json({'error': 'm_runs_per_set must be an integer'}, 400)
        return
    if not (1 <= m <= 20):
        self._send_json(
            {'error': 'm_runs_per_set must be between 1 and 20'}, 400)
        return

    # --- m_run_jitter ---
    try:
        jitter = int(body.get('m_run_jitter', 1))
    except (ValueError, TypeError):
        self._send_json({'error': 'm_run_jitter must be an integer'}, 400)
        return
    if not (0 <= jitter <= 50):
        self._send_json(
            {'error': 'm_run_jitter must be between 0 and 50'}, 400)
        return

    # --- param_sets ---
    raw_sets = body.get('param_sets', [])
    if not isinstance(raw_sets, list):
        self._send_json({'error': 'param_sets must be a list'}, 400)
        return
    cleaned_sets = []
    for idx, raw in enumerate(raw_sets):
        if not isinstance(raw, dict):
            self._send_json(
                {'error': f'Param set #{idx + 1} is not an object'}, 400)
            return
        cleaned, err = _validate_param_set(raw)
        if err:
            self._send_json(
                {'error': f'Param set #{idx + 1}: {err}'}, 400)
            return
        cleaned_sets.append(cleaned)

    # --- selected_fires ---
    from .app import state as app_state
    raw_fires = body.get('selected_fires', [])
    if not isinstance(raw_fires, list):
        self._send_json({'error': 'selected_fires must be a list'}, 400)
        return
    valid_fires = [
        fn for fn in raw_fires
        if isinstance(fn, str) and fn in app_state.fires
    ]

    description = str(body.get('description', ''))[:500]

    with _astate.lock:
        _astate.config.param_sets = cleaned_sets
        _astate.config.m_runs_per_set = m
        _astate.config.m_run_jitter = jitter
        _astate.config.selected_fires = valid_fires
        _astate.config.description = description
    _save_config()
    self._send_json({
        'status': 'saved',
        'n_param_sets': len(cleaned_sets),
        'm_runs_per_set': m,
        'n_selected_fires': len(valid_fires),
    })


# =========================================================================
# Fire selection API (reuses the main app's fire list, admin-only)
# =========================================================================

def handle_api_analyzer_fires(self):
    """Return fire list suitable for analyzer fire selection.

    Includes a subset of fields plus analyzer-side status (how many
    accepted runs are already on disk) so the admin can prioritise.
    """
    if not _require_admin(self):
        return
    from .app import state as app_state
    result = []
    for f in app_state.fires.values():
        if f.hidden:
            continue
        afi = _astate.fires.get(f.fire_numbe)
        n_accepted = len(afi.runs) if afi else 0
        analyzer_status = afi.status.value if afi else 'pending'
        result.append({
            'fire_numbe': f.fire_numbe,
            'fire_date': f.fire_date,
            'fire_year': f.fire_year,
            'fire_size_ha': f.fire_size_ha,
            'status': f.status.value,
            'analyzer_status': analyzer_status,
            'n_accepted_runs': n_accepted,
        })
    self._send_json(result)


# =========================================================================
# Worker trigger (stub in Phase B — real execution comes in Phase C)
# =========================================================================

def handle_api_analyzer_start(self):
    """Start the analyzer worker with the current config."""
    if not _require_admin(self):
        return
    if not _astate.config.param_sets:
        self._send_json({'error': 'No parameter sets configured'}, 400)
        return
    if not _astate.config.selected_fires:
        self._send_json({'error': 'No fires selected'}, 400)
        return

    from .app import state as app_state
    from .analyzer_worker import start_worker

    started = start_worker(app_state, _astate)
    if not started:
        self._send_json({'error': 'Analyzer already running'}, 400)
        return
    self._send_json({
        'status': 'started',
        'n_param_sets': len(_astate.config.param_sets),
        'm_runs_per_set': _astate.config.m_runs_per_set,
        'n_selected_fires': len(_astate.config.selected_fires),
        'total_runs': (len(_astate.config.param_sets)
                        * _astate.config.m_runs_per_set
                        * len(_astate.config.selected_fires)),
    })


def handle_api_analyzer_cancel(self):
    """Signal the worker to stop after the current fire."""
    if not _require_admin(self):
        return
    if not _astate.running:
        self._send_json({'error': 'Analyzer is not running'}, 400)
        return
    _astate.cancel_event.set()
    self._send_json({'status': 'cancelling'})


# =========================================================================
# Per-fire page + gallery API
# =========================================================================

_FIRE_NUMBE_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_. -]*$')


def _safe_fire_numbe(fire_numbe: str) -> str:
    """Decode + validate a fire_numbe from a URL to avoid path traversal."""
    fire_numbe = unquote(fire_numbe)
    if not _FIRE_NUMBE_RE.fullmatch(fire_numbe):
        raise ValueError('invalid fire_numbe')
    return fire_numbe


def handle_analyzer_fire_page(self, fire_numbe):
    """Render the per-fire gallery page."""
    if not _require_admin(self):
        return
    from .app import render_template, state as app_state
    try:
        fire_numbe = _safe_fire_numbe(fire_numbe)
    except ValueError:
        self.send_error(400, 'Invalid fire number')
        return
    if fire_numbe not in app_state.fires:
        self.send_error(404, 'Unknown fire')
        return
    import json as _json
    f = app_state.fires[fire_numbe]
    html = render_template('analyzer_fire.html', {
        'fire_numbe': fire_numbe,
        'fire_numbe_json': _json.dumps(fire_numbe),
        'fire_date': f.fire_date or '',
        'fire_year': str(f.fire_year),
        'fire_size_ha': str(f.fire_size_ha),
        'raster_name': os.path.basename(app_state.raster_path),
    })
    self._send_html(html)


def _run_to_json(run):
    return {
        'set_idx': run.set_idx,
        'run_idx': run.run_idx,
        'params': dict(run.params),
        'padding_used': run.padding_used,
        'agreement_pct': run.agreement_pct,
        'ml_area_ha': run.ml_area_ha,
        'status': run.status,
        'error_msg': run.error_msg,
        'accepted': run.accepted,
        'accept_id': run.accept_id,
        'accepted_at': run.accepted_at,
        'has_thumb': bool(run.thumb_rel),
        'has_comparison': bool(run.comparison_rel),
        'timestamp': run.timestamp,
    }


def handle_api_analyzer_fire_status(self, fire_numbe):
    """Return all per-fire data the gallery needs."""
    if not _require_admin(self):
        return
    try:
        fire_numbe = _safe_fire_numbe(fire_numbe)
    except ValueError:
        self._send_json({'error': 'invalid fire_numbe'}, 400)
        return
    info = _astate.fires.get(fire_numbe)
    if info is None:
        self._send_json({
            'fire_numbe': fire_numbe,
            'status': 'pending',
            'runs': [],
            'console': [],
            'param_sets': list(_astate.config.param_sets),
            'm_runs_per_set': _astate.config.m_runs_per_set,
            'saved_max_padding': 0.0,
            'saved_max_crop': '',
        })
        return
    with _astate.lock:
        runs = [_run_to_json(r) for r in info.runs]
        console = list(info.console_log)
    self._send_json({
        'fire_numbe': fire_numbe,
        'status': info.status.value,
        'error_msg': info.error_msg,
        'runs': runs,
        'console': console,
        'param_sets': list(_astate.config.param_sets),
        'm_runs_per_set': _astate.config.m_runs_per_set,
        'saved_max_padding': info.saved_max_padding,
        'saved_max_crop': info.saved_max_crop_rel,
        'batch': dict(_astate.batch_status) if _astate.batch_status else None,
        'running': _astate.running,
    })


def _safe_join(root: str, *parts: str) -> str:
    """Join and assert the result stays inside root (path traversal guard)."""
    root = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root, *parts))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError('path escapes root')
    return full


def handle_api_analyzer_run_image(self, fire_numbe, set_idx, run_idx):
    """Serve a run's thumbnail / comparison PNG from its cache subdir."""
    if not _require_admin(self):
        return
    try:
        fire_numbe = _safe_fire_numbe(fire_numbe)
        s = int(set_idx)
        r = int(run_idx)
    except (ValueError, TypeError):
        self.send_error(400, 'invalid parameters')
        return

    info = _astate.fires.get(fire_numbe)
    if info is None:
        self.send_error(404, 'fire not found')
        return
    run = next((x for x in info.runs
                if x.set_idx == s and x.run_idx == r), None)
    if run is None:
        self.send_error(404, 'run not found')
        return

    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(self.path).query)
    kind = qs.get('type', ['thumb'])[0]

    from .analyzer_worker import _padding_key, _run_dir_path
    snap_dir = os.path.join(
        _astate.cache_root, fire_numbe, _padding_key(run.padding_used))
    run_dir = _run_dir_path(snap_dir, s, r)

    name_map = {
        'thumb': 'thumb.png',
        'comparison': 'comparison.png',
        'brush': 'brush_comparison.png',
    }
    name = name_map.get(kind)
    if not name:
        self.send_error(400, 'invalid type')
        return

    # Fallback: if the run was accepted, its files may have moved to
    # canonical dir. Prefer canonical if present (more permanent).
    candidate_paths = []
    if run.accepted and run.accept_id:
        candidate_paths.append(
            os.path.join(_astate.analyzer_root, fire_numbe,
                         run.accept_id, name))
    candidate_paths.append(os.path.join(run_dir, name))

    for p in candidate_paths:
        try:
            p = _safe_join(_astate.analyzer_root, os.path.relpath(p, _astate.analyzer_root))
        except ValueError:
            continue
        if os.path.isfile(p):
            self._send_file(p, 'image/png')
            return
    self.send_error(404, 'image not found')


def handle_api_analyzer_accept(self, fire_numbe):
    if not _require_admin(self):
        return
    try:
        fire_numbe = _safe_fire_numbe(fire_numbe)
    except ValueError:
        self._send_json({'error': 'invalid fire_numbe'}, 400)
        return
    body = self._read_body()
    if body is None:
        return
    try:
        s = int(body.get('set_idx'))
        r = int(body.get('run_idx'))
    except (TypeError, ValueError):
        self._send_json({'error': 'set_idx/run_idx required'}, 400)
        return

    from .app import state as app_state
    from .analyzer_accept import accept_run
    try:
        result = accept_run(fire_numbe, s, r, _astate, app_state)
    except ValueError as exc:
        self._send_json({'error': str(exc)}, 400)
        return
    except FileNotFoundError as exc:
        self._send_json({'error': str(exc)}, 404)
        return

    from .analyzer_worker import _save_fire_runs
    info = _astate.fires.get(fire_numbe)
    if info:
        _save_fire_runs(info, _astate)

    self._send_json({'status': 'accepted', **result})


def handle_api_analyzer_unaccept(self, fire_numbe):
    if not _require_admin(self):
        return
    try:
        fire_numbe = _safe_fire_numbe(fire_numbe)
    except ValueError:
        self._send_json({'error': 'invalid fire_numbe'}, 400)
        return
    body = self._read_body()
    if body is None:
        return
    accept_id = body.get('accept_id', '')
    if not accept_id or not re.fullmatch(r'run_\d{4,}', accept_id):
        self._send_json({'error': 'accept_id required'}, 400)
        return

    from .app import state as app_state
    from .analyzer_accept import unaccept_run
    try:
        result = unaccept_run(fire_numbe, accept_id, _astate, app_state)
    except ValueError as exc:
        self._send_json({'error': str(exc)}, 400)
        return

    from .analyzer_worker import _save_fire_runs
    info = _astate.fires.get(fire_numbe)
    if info:
        _save_fire_runs(info, _astate)

    self._send_json({'status': 'unaccepted', **result})


def _build_composite_overlay_png(fire_numbe: str, app_state) -> str:
    """Compose all accepted-run perimeters onto the biggest-padded backdrop.

    Each accepted run gets its own colour (cycled through a palette).
    Returns the absolute path to the generated PNG, or '' on failure.

    The overlay is rebuilt on every call to stay in sync with the current
    accept set. Cheap enough: one post.png read, N classification reads,
    one scipy zoom per mask, one imsave.
    """
    import numpy as np
    from osgeo import gdal
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.image import imread, imsave
    from scipy.ndimage import binary_erosion, zoom as scipy_zoom

    info = _astate.fires.get(fire_numbe)
    if info is None:
        return ''
    canon_dir = os.path.join(_astate.analyzer_root, fire_numbe)
    if not os.path.isdir(canon_dir):
        return ''

    # Backdrop: <FIRE>_post_max.png (written at accept time alongside
    # the crop). If it's missing we can't draw anything useful.
    backdrop_path = os.path.join(canon_dir, f'{fire_numbe}_post_max.png')
    crop_max_path = os.path.join(canon_dir, f'{fire_numbe}_crop_max.bin')
    if not os.path.isfile(backdrop_path) or not os.path.isfile(crop_max_path):
        return ''

    backdrop = imread(backdrop_path)
    if backdrop.ndim == 2:
        backdrop = np.stack([backdrop] * 3, axis=2)
    backdrop = backdrop[:, :, :3].astype(np.float32)
    if backdrop.max() > 1.5:
        backdrop /= 255.0
    ph, pw = backdrop.shape[:2]

    # Geotransform of the backdrop crop
    ds_bd = gdal.Open(crop_max_path, gdal.GA_ReadOnly)
    bd_gt = ds_bd.GetGeoTransform()
    bd_w, bd_h = ds_bd.RasterXSize, ds_bd.RasterYSize
    ds_bd = None

    # Preview might be scaled down from the raw crop — compute scaling
    # factors to map raster-pixel coords onto PNG-pixel coords.
    sx = pw / bd_w
    sy = ph / bd_h

    # Colour palette (R, G, B) in 0..1 — high-contrast, distinguishable
    palette = [
        (1.00, 0.15, 0.15),
        (0.15, 0.75, 0.15),
        (0.15, 0.55, 1.00),
        (1.00, 0.75, 0.00),
        (0.85, 0.25, 0.85),
        (0.00, 0.80, 0.80),
        (1.00, 0.45, 0.00),
        (0.55, 0.35, 0.00),
    ]

    accepted = [r for r in info.runs if r.accepted and r.accept_id]
    if not accepted:
        # Nothing to overlay — save the backdrop unchanged.
        out_path = os.path.join(canon_dir, f'{fire_numbe}_overlay.png')
        imsave(out_path, np.clip(backdrop, 0, 1))
        return out_path

    composite = backdrop.copy()
    legend_entries = []

    for idx, run in enumerate(accepted):
        color = palette[idx % len(palette)]
        clf_path = os.path.join(
            canon_dir, run.accept_id, 'classified.bin')
        if not os.path.isfile(clf_path):
            continue

        try:
            ds = gdal.Open(clf_path, gdal.GA_ReadOnly)
            arr = ds.GetRasterBand(1).ReadAsArray()
            arr_gt = ds.GetGeoTransform()
            aw = ds.RasterXSize
            ah = ds.RasterYSize
            ds = None
        except Exception as exc:
            sys.stderr.write(
                f'[analyzer overlay] WARNING: failed to read '
                f'{clf_path}: {exc}\n')
            continue

        # Align classified onto the backdrop's crop grid.
        if (abs(arr_gt[1] - bd_gt[1]) < 1e-6
                and abs(arr_gt[5] - bd_gt[5]) < 1e-6):
            off_x = int(round((arr_gt[0] - bd_gt[0]) / bd_gt[1]))
            off_y = int(round((arr_gt[3] - bd_gt[3]) / bd_gt[5]))
            aligned = np.zeros((bd_h, bd_w), dtype=np.uint8)
            src_y0 = max(0, -off_y)
            src_x0 = max(0, -off_x)
            dst_y0 = max(0, off_y)
            dst_x0 = max(0, off_x)
            copy_h = min(ah - src_y0, bd_h - dst_y0)
            copy_w = min(aw - src_x0, bd_w - dst_x0)
            if copy_h <= 0 or copy_w <= 0:
                continue
            aligned[dst_y0:dst_y0 + copy_h,
                    dst_x0:dst_x0 + copy_w] = (
                (arr[src_y0:src_y0 + copy_h,
                     src_x0:src_x0 + copy_w] > 0)).astype(np.uint8)
        else:
            # Different pixel sizes — fall back to naive resize
            aligned = (scipy_zoom(
                (arr > 0).astype(np.float32),
                (bd_h / ah, bd_w / aw), order=0) > 0.5).astype(np.uint8)

        # Perimeter = mask minus its 1-px erosion, dilated a bit for visibility.
        from scipy.ndimage import binary_dilation
        eroded = binary_erosion(aligned.astype(bool), iterations=1)
        outline = aligned.astype(bool) & (~eroded)
        outline = binary_dilation(outline, iterations=1)

        # Scale outline to PNG pixel dims
        if bd_h != ph or bd_w != pw:
            outline_scaled = scipy_zoom(
                outline.astype(np.float32),
                (sy, sx), order=0) > 0.5
        else:
            outline_scaled = outline

        r, g, b = color
        mask = outline_scaled
        composite[mask, 0] = r
        composite[mask, 1] = g
        composite[mask, 2] = b

        legend_entries.append({
            'accept_id': run.accept_id,
            'color_rgb': [int(r * 255), int(g * 255), int(b * 255)],
            'agreement_pct': run.agreement_pct,
            'padding_used': run.padding_used,
            'set_idx': run.set_idx,
            'run_idx': run.run_idx,
        })

    out_path = os.path.join(canon_dir, f'{fire_numbe}_overlay.png')
    imsave(out_path, np.clip(composite, 0, 1))

    # Stash the legend in a sidecar JSON so the UI can render it.
    import json as _json
    legend_path = os.path.join(canon_dir, f'{fire_numbe}_overlay_legend.json')
    with open(legend_path, 'w') as f:
        _json.dump(legend_entries, f, indent=2)

    return out_path


def handle_api_analyzer_overlay(self, fire_numbe):
    """Serve (rebuilding on demand) the composite overlay PNG."""
    if not _require_admin(self):
        return
    try:
        fire_numbe = _safe_fire_numbe(fire_numbe)
    except ValueError:
        self.send_error(400, 'invalid fire_numbe')
        return
    from .app import state as app_state
    try:
        path = _build_composite_overlay_png(fire_numbe, app_state)
    except Exception as exc:
        sys.stderr.write(
            f'[analyzer overlay] ERROR building overlay for '
            f'{fire_numbe}: {exc}\n')
        self._send_json({'error': str(exc)}, 500)
        return
    if not path:
        self._send_json(
            {'error': 'no backdrop available (no accepted runs yet?)'},
            404)
        return
    self._send_file(path, 'image/png')


def handle_api_analyzer_overlay_legend(self, fire_numbe):
    """Return the legend (colour per accept_id) for the composite overlay."""
    if not _require_admin(self):
        return
    try:
        fire_numbe = _safe_fire_numbe(fire_numbe)
    except ValueError:
        self._send_json({'error': 'invalid fire_numbe'}, 400)
        return
    canon_dir = os.path.join(_astate.analyzer_root, fire_numbe)
    legend_path = os.path.join(
        canon_dir, f'{fire_numbe}_overlay_legend.json')
    if not os.path.isfile(legend_path):
        self._send_json([])
        return
    import json as _json
    try:
        with open(legend_path) as f:
            data = _json.load(f)
    except Exception:
        data = []
    self._send_json(data)


def handle_api_analyzer_csv(self):
    """Download analyzer_accepted.csv for offline analysis."""
    if not _require_admin(self):
        return
    if not os.path.isfile(_astate.csv_file):
        self._send_json({'error': 'CSV not yet populated'}, 404)
        return
    with open(_astate.csv_file, 'rb') as f:
        body = f.read()
    self.send_response(200)
    self.send_header('Content-Type', 'text/csv; charset=utf-8')
    self.send_header('Content-Length', str(len(body)))
    self.send_header('Content-Disposition',
                     'attachment; filename="analyzer_accepted.csv"')
    self.end_headers()
    self.wfile.write(body)


def handle_api_analyzer_csv_preview(self):
    """Return analyzer_accepted.csv rows as JSON plus summary stats.

    The UI renders this as a table on the main analyzer page so the
    admin can scan accepted runs at a glance without downloading the
    raw CSV.
    """
    if not _require_admin(self):
        return
    from .analyzer_state import ANALYZER_CSV_FIELDNAMES

    rows = []
    if os.path.isfile(_astate.csv_file):
        import csv
        try:
            with open(_astate.csv_file, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as exc:
            sys.stderr.write(
                f'[analyzer] WARNING: csv preview read failed: {exc}\n')
            sys.stderr.flush()

    # Summary statistics: totals + per-fire / per-region / per-size-bucket
    unique_fires = set()
    per_region: dict = {}
    per_year: dict = {}
    per_bucket: dict = {}
    agreements: list = []
    for r in rows:
        fn = r.get('fire_numbe', '')
        if fn:
            unique_fires.add(fn)
        reg = r.get('fire_region', '') or '?'
        per_region[reg] = per_region.get(reg, 0) + 1
        yr = r.get('fire_year', '') or '?'
        per_year[yr] = per_year.get(yr, 0) + 1
        lo = r.get('size_bucket_lo', '') or '0'
        hi = r.get('size_bucket_hi', '') or ''
        bkt = f'{lo}-{hi}' if hi and hi != '0' else f'{lo}+'
        per_bucket[bkt] = per_bucket.get(bkt, 0) + 1
        try:
            v = float(r.get('agreement_pct') or -1)
            if v >= 0:
                agreements.append(v)
        except (TypeError, ValueError):
            pass

    # Count in-memory fires (analyzer_state) for "fires touched" metric
    fires_with_any_run = 0
    fires_analyzed = 0
    with _astate.lock:
        for info in _astate.fires.values():
            if info.runs:
                fires_with_any_run += 1
            if info.status.value in ('analyzed', 'partial'):
                fires_analyzed += 1

    agr_stats = {}
    if agreements:
        agreements.sort()
        n = len(agreements)
        agr_stats = {
            'min': round(agreements[0], 1),
            'max': round(agreements[-1], 1),
            'median': round(agreements[n // 2], 1),
            'mean': round(sum(agreements) / n, 1),
        }

    self._send_json({
        'rows': rows,
        'fieldnames': list(ANALYZER_CSV_FIELDNAMES),
        'summary': {
            'total_accepted': len(rows),
            'unique_fires': len(unique_fires),
            'fires_with_any_run': fires_with_any_run,
            'fires_analyzed': fires_analyzed,
            'per_region': per_region,
            'per_year': per_year,
            'per_size_bucket': per_bucket,
            'agreement_stats': agr_stats,
        },
    })


# =========================================================================
# Route registration — called once at startup from __main__.py
# =========================================================================

def register_routes():
    """Extend FireHandler's route tables with analyzer routes.

    This is the only integration point with app.py. We append — we do
    not replace — so existing routes keep behaving exactly as before.
    """
    from .app import FireHandler

    # Attach handler methods (must be named as referenced in routes).
    FireHandler.handle_analyzer_page = handle_analyzer_page
    FireHandler.handle_analyzer_fire_page = handle_analyzer_fire_page
    FireHandler.handle_api_analyzer_status = handle_api_analyzer_status
    FireHandler.handle_api_analyzer_config_get = handle_api_analyzer_config_get
    FireHandler.handle_api_analyzer_config_post = handle_api_analyzer_config_post
    FireHandler.handle_api_analyzer_fires = handle_api_analyzer_fires
    FireHandler.handle_api_analyzer_start = handle_api_analyzer_start
    FireHandler.handle_api_analyzer_cancel = handle_api_analyzer_cancel
    FireHandler.handle_api_analyzer_fire_status = handle_api_analyzer_fire_status
    FireHandler.handle_api_analyzer_run_image = handle_api_analyzer_run_image
    FireHandler.handle_api_analyzer_accept = handle_api_analyzer_accept
    FireHandler.handle_api_analyzer_unaccept = handle_api_analyzer_unaccept
    FireHandler.handle_api_analyzer_csv = handle_api_analyzer_csv
    FireHandler.handle_api_analyzer_csv_preview = handle_api_analyzer_csv_preview
    FireHandler.handle_api_analyzer_overlay = handle_api_analyzer_overlay
    FireHandler.handle_api_analyzer_overlay_legend = (
        handle_api_analyzer_overlay_legend)

    FireHandler.ROUTES_GET.extend([
        (re.compile(r'^/analyzer$'), 'handle_analyzer_page'),
        (re.compile(r'^/analyzer/fire/(?P<fire_numbe>[^/]+)$'),
         'handle_analyzer_fire_page'),
        (re.compile(r'^/api/analyzer/status$'),
         'handle_api_analyzer_status'),
        (re.compile(r'^/api/analyzer/config$'),
         'handle_api_analyzer_config_get'),
        (re.compile(r'^/api/analyzer/fires$'),
         'handle_api_analyzer_fires'),
        (re.compile(r'^/api/analyzer/csv$'),
         'handle_api_analyzer_csv'),
        (re.compile(r'^/api/analyzer/csv/preview$'),
         'handle_api_analyzer_csv_preview'),
        (re.compile(
            r'^/api/analyzer/fire/(?P<fire_numbe>[^/]+)/status$'),
         'handle_api_analyzer_fire_status'),
        (re.compile(
            r'^/api/analyzer/fire/(?P<fire_numbe>[^/]+)/run/'
            r'(?P<set_idx>\d+)/(?P<run_idx>\d+)/image$'),
         'handle_api_analyzer_run_image'),
        (re.compile(
            r'^/api/analyzer/fire/(?P<fire_numbe>[^/]+)/overlay$'),
         'handle_api_analyzer_overlay'),
        (re.compile(
            r'^/api/analyzer/fire/(?P<fire_numbe>[^/]+)/overlay/legend$'),
         'handle_api_analyzer_overlay_legend'),
    ])
    FireHandler.ROUTES_POST.extend([
        (re.compile(r'^/api/analyzer/config$'),
         'handle_api_analyzer_config_post'),
        (re.compile(r'^/api/analyzer/start$'),
         'handle_api_analyzer_start'),
        (re.compile(r'^/api/analyzer/cancel$'),
         'handle_api_analyzer_cancel'),
        (re.compile(
            r'^/api/analyzer/fire/(?P<fire_numbe>[^/]+)/accept$'),
         'handle_api_analyzer_accept'),
        (re.compile(
            r'^/api/analyzer/fire/(?P<fire_numbe>[^/]+)/unaccept$'),
         'handle_api_analyzer_unaccept'),
    ])
