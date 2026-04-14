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
from urllib.parse import urlparse, unquote

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

        if out_name not in fire.available_views:
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


def _get_recommended_params(fire_size_ha: float) -> dict:
    """Find recommended params for a fire of given size."""
    for row in state.recommended_settings:
        if (fire_size_ha >= row.get('min_ha', 0)
                and (row.get('max_ha') is None
                     or fire_size_ha < row['max_ha'])):
            return row.get('params', {})
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

        padding = params.get('padding', state.padding)

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

    # -- Generate overlay previews if classified raster exists --
    clf_path = os.path.join(cache_dir,
                            f'{fire_numbe}_crop.bin_classified.bin')
    if os.path.isfile(clf_path):
        _generate_result_preview(fire)
    elif fire.hint_bin and os.path.isfile(fire.hint_bin):
        # At least generate the hint overlay
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

    # Compute ML area
    gt = state.raster_gt
    pixel_area_m2 = abs(gt[1] * gt[5])
    ml_area_ha = None
    ml_area_m2 = None
    crop_name = f'{fire_numbe}_crop'
    clf_bin = os.path.join(fire_dir, f'{crop_name}.bin_classified.bin')
    if os.path.isfile(clf_bin):
        try:
            ds = gdal.Open(clf_bin, gdal.GA_ReadOnly)
            arr = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            burned_px = int(np.nansum(arr > 0))
            ml_area_m2 = burned_px * pixel_area_m2
            ml_area_ha = ml_area_m2 / 10000.0
        except Exception:
            pass

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

def _build_mapping_cmd(fire: FireInfo, params: dict) -> list[str]:
    """Build the subprocess command for fire_mapping_cli.py."""
    rate = float(params.get('sample_rate', state.sample_rate))
    min_s = int(params.get('min_samples', state.min_samples))
    max_s = int(params.get('max_samples', state.max_samples))
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

    def handle_api_notes(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        body = self._read_body()
        state.fires[fire_numbe].notes = body.get('notes', '')
        self._send_json({'status': 'saved'})

    def handle_api_report(self):
        """Generate PDF report of accepted fires and send as download."""
        from batch_fire_mapping.generate_report import generate_report
        import tempfile
        tmp = tempfile.mktemp(suffix='.pdf')
        pdf = generate_report(state.output_root, tmp)
        if pdf is None or not os.path.isfile(pdf):
            self._send_json({'error': 'Report generation failed'}, 500)
            return
        try:
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
            try:
                os.remove(pdf)
            except Exception:
                pass

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

        def sse(event_type, data):
            payload = json.dumps({'type': event_type, **data})
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
                    _generate_result_preview(fire)
                    fire.status = FireStatus.MAPPED
                    sse('complete', {
                        'comparison_url': (
                            f'/api/fire/{fire_numbe}/comparison'
                            f'?t={int(time.time())}'),
                        'agreement_pct': fire.agreement_pct,
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
