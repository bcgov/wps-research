"""Per-fire preview / status / abort routes (prepare, console, progress).

This is one slice of FireHandler. Methods reference module-level
helpers from ``app`` via top-of-file imports; ``state`` is rebound
in :func:`init` so it tracks the live :class:`AppState` instance
created by ``app.init_app``.
"""

import datetime
import glob
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse, unquote, parse_qs

import numpy as np
from osgeo import gdal

from ..state import AppState, FireInfo, FireStatus
from ..auth import (
    _hash_token, _normalize_ip, _check_login_rate, _record_failed_login,
    _sweep_expired_sessions, _SESSION_MAX_AGE,
)
from ..notifications import (
    _save_notifications, _load_notifications, _prune_notifications_unlocked,
    _push_notification, _pop_notifications,
)
from ..cache_retention import (
    _save_cache_retention, _load_cache_retention, _dir_bytes_and_mtime,
    _cache_scan, _cache_sweep, _cache_sweep_loop, _cache_sweep_lock,
)
from ..progress import (
    _STAGE_MARKERS, _STAGE_ORDER_FULL, _STAGE_ORDER_RESUME, _STAGE_LABELS,
    _STAGE_TIMINGS_MAX_SAMPLES, _STAGE_FALLBACK,
    _detect_stage, _save_stage_timings, _load_stage_timings,
    _record_stage_duration, _stage_median, _estimate_full_run_seconds,
    _ProgressTracker, _progress_snapshot, _ETA_FUDGE, _ETA_FLOOR_S,
)
from ..mapping import (
    _compute_ml_area, _overlay_mask_on_post, _generate_result_preview,
    _compute_agreement,
)
from ..persistence import (
    _save_sessions, _save_settings, _save_notes, _save_ip_list,
    _save_fire_state, _load_fire_state,
    _save_active_year, _switch_year,
)
from ..brush import (
    _class_brush_exe, _read_envi_mask, _write_envi_mask_like,
    _run_class_brush_only, _align_mask_to_crop_frame,
    _render_comparison_png, _render_ml_classification_png,
    _render_brush_comparison_png,
)
from ..templates import _html_escape, render_template
from ..validation import _PARAM_SPEC, _validate_param, _validate_embed_bands
from ..mapping_cmd import _build_mapping_cmd
from ..io_utils import _atomic_yaml_dump
from ..preview import generate_all_previews
from ..prepare import switch_hint_mode, switch_post_source

# Late-bound to avoid a circular-import: app imports the mixins, then
# app.init_app calls each mixin's ``init`` which re-assigns ``state`` and
# the inter-handler helpers/registries that live in ``app.py``.
state: AppState = None
_HERE = None
_gpu_lock = None
_gpu_queue_lock = None
_gpu_queue = None
_batch_thread = None
_SUBPROCESS_SILENCE_TIMEOUT = None
_batch_cancel = None
_serial_procs = None
_serial_procs_lock = None
_rebrush_procs = None
_rebrush_procs_lock = None
_accept_in_progress = None
_accept_in_progress_lock = None
_accept_file_lock = None
_set_fire_status = None
_terminate_serial_proc = None
_stream_subprocess = None
_get_recommended_settings = None
_clone_setting = None
_batch_map_worker = None
_serial_map_worker = None
_jitter_hdbscan = None
_prepare_fire_sync = None
_accept_fire_sync = None
_ensure_brush_comparison_in_cache = None
# These two stay in app.py because they need ``global`` rebinding.
# They are referenced through ``import_app_globals`` only as needed.


def init(app_state, helpers):
    """Bind shared helpers and the live AppState into this mixin module.

    ``helpers`` is the namespace dict published by ``app.init_app``;
    we copy each name into our module globals so unmodified method
    bodies (which reference bare names like ``state`` or ``_gpu_lock``)
    look them up here at call time.
    """
    g = globals()
    g['state'] = app_state
    for name, value in helpers.items():
        g[name] = value


class FireRoutes:
    """Per-fire preview / status / abort routes (prepare, console, progress)."""


    # -- Fire API --

    def handle_api_prepare(self, fire_numbe):
        # Timed end to end. The client reports its own round trip; the
        # difference between the two is network. If this number is
        # small and the client's is large, the wait is transfer, not
        # server work -- which is the distinction that has been hard to
        # make from the outside.
        _t0 = time.time()
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return

        fire = state.fires[fire_numbe]
        body = self._read_body()
        if body is None:
            return
        padding = body.get('padding')

        prev_status = fire.status

        needs_prepare = fire.status in (
            FireStatus.PENDING, FireStatus.ERROR)
        # Re-prepare accepted/mapped fires with no previews yet, or
        # whose previews/post.png went missing on disk (e.g. operator
        # manually wiped .web_cache while the server was running). The
        # in-memory available_views list isn't enough — _load_fire_state
        # filters it on startup, but a wipe after startup would leave
        # the in-memory list stale.
        if fire.status in (FireStatus.ACCEPTED, FireStatus.MAPPED):
            post_png = (os.path.join(fire.cache_dir, 'previews', 'post.png')
                        if fire.cache_dir else '')
            if (not fire.available_views
                    or not post_png
                    or not os.path.isfile(post_png)):
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
                    if (fire.previously_accepted
                            and fire.agreement_pct >= 0):
                        fire.previously_accepted_agreement_pct = (
                            fire.agreement_pct)
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

        sys.stderr.write(
            f'[perf] /prepare {fire_numbe}: '
            f'{(time.time() - _t0) * 1000:.0f} ms server-side '
            f'(status={fire.status.value}, '
            f'views={len(fire.available_views)})\n')
        sys.stderr.flush()
        self._send_json({
            'status': fire.status.value,
            'views': fire.available_views,
            'crop_w': fire.crop_w,
            'crop_h': fire.crop_h,
            'sample_size': fire.sample_size,
            'perimeter_type': fire.perimeter_type,
            'hint_mode': fire.hint_mode,
            # Report what the USER is on, not the transient value the
            # background prebuild may currently be sitting at -- that
            # race made a new fire open on MRAP instead of L2.
            'post_source': (getattr(fire, 'user_post_source', '')
                            or getattr(fire, 'post_source', 'l2')
                            or 'l2'),
            'acc_start': fire.acc_start,
            'acc_end': fire.acc_end,
            'has_comparison': has_comparison,
            'has_brush_comparison': has_brush,
            'previously_accepted': fire.previously_accepted,
            'ml_area_ha': fire.ml_area_ha,
        })

    def handle_api_hint_mode(self, fire_numbe):
        """Switch the hint mask between viirs / red-wins (post) / red-wins (diff)."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        body = self._read_body()
        if body is None:
            return
        mode = body.get('mode', 'viirs')
        result = switch_hint_mode(fire, mode)
        if not result.get('ok'):
            self._send_json({'error': result.get('error', 'unknown')}, 400)
            return
        self._send_json({
            'ok': True,
            'perimeter_type': fire.perimeter_type,
            'hint_mode': fire.hint_mode,
        })

    def handle_api_post_source(self, fire_numbe):
        """Switch between L2-recent and MRAP post imagery."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        body = self._read_body()
        if body is None:
            return
        source = body.get('source', 'l2')
        # An explicit user choice, so it becomes the stable value too.
        result = switch_post_source(fire, source)
        if result.get('ok'):
            try:
                from ..prepare import set_user_post_source
                set_user_post_source(fire, source)
            except Exception:
                fire.user_post_source = source
        if not result.get('ok'):
            self._send_json({'error': result.get('error', 'unknown')}, 400)
            return
        self._send_json(result)

    def handle_api_fire_overlays(self, fire_numbe):
        """Vector overlays (S2 tile grid + BCWS) in crop pixel coords."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        try:
            from ..fire_overlays import build_fire_overlays
            self._send_json(build_fire_overlays(state, fire))
        except Exception as exc:
            self._send_json({'error': str(exc)}, 500)

    def handle_api_fire_date_plot(self, fire_numbe):
        """Per-acquisition coverage polygons for the L2-recent buffer.

        Read straight back from the ramdisk sidecar written alongside
        the stack, so closing and reopening a fire (or flipping post
        source) recalls exactly the polygons matching the L2 buffer
        currently on disk. Returns an empty set for MRAP, which is a
        single composite and has no per-date structure.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if getattr(fire, 'post_source', 'l2') != 'l2':
            self._send_json({'dates': [], 'width': 0, 'height': 0,
                             'reason': 'not applicable to MRAP'})
            return
        try:
            from ..l2_recent import date_polygons_path
            path = date_polygons_path(fire.crop_bin)
            if not path or not os.path.isfile(path):
                self._send_json({'dates': [], 'width': 0, 'height': 0,
                                 'reason': 'not generated yet'})
                return
            with open(path, encoding='utf-8') as f:
                self._send_json(json.loads(f.read()))
        except Exception as exc:
            self._send_json({'error': str(exc)}, 500)

    def handle_api_date_plot_rebuild(self, fire_numbe):
        """Regenerate the per-acquisition coverage sidecar.

        The sidecar is a by-product of BUILDING the L2 buffer, so a
        fire whose stack was already cached (created before the feature
        existed, or reused across a restart) has none. Recomputing the
        coverage means re-extracting the same zips, so this simply
        forces a stack rebuild -- run in the background because that is
        tens of seconds of work, with progress visible in the fire
        console like any other build.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if getattr(fire, 'post_source', 'l2') != 'l2':
            self._send_json(
                {'error': 'Coverage applies to the L2 source only.'}, 400)
            return

        def _work():
            try:
                from ..aoi_stack import ensure_aoi_stack
                fire.console_log.append(
                    '  Rebuilding L2 buffer to generate per-acquisition '
                    'coverage (this re-reads the source zips) ...')
                ensure_aoi_stack(
                    fire.fire_numbe, fire.bbox_native, force=True,
                    instance_key=getattr(state, 'shared_root', '') or '',
                    post_source='l2',
                    ref_raster=(state.rasters_by_year.get(fire.fire_year)
                                or state.raster_path),
                    log_cb=lambda m: fire.console_log.append(m.rstrip()))
                fire.console_log.append(
                    '  Coverage generated -- the date plot will appear '
                    'shortly.')
            except Exception as exc:
                fire.console_log.append(
                    f'  Coverage generation failed: {exc}')

        threading.Thread(target=_work, daemon=True).start()
        self._send_json({'ok': True, 'started': True})

    def _geo_headers(self, fire, png_path, view_key=None):
        """Georeferencing headers describing the PNG being served.

        The client pairs these with the exact bytes it displays, so a
        preview can never be matched against another raster's
        geotransform. Name-based lookup kept getting this wrong
        because 'result' and 'serial_N' are copies whose provenance is
        not recoverable from the filename.

        Order of authority:
          1. the geo.json entry recorded when the PNG was rendered,
          2. the run's own classified raster (same grid as the crop it
             was mapped against),
          3. the current crop.
        """
        try:
            import json as _json
            from osgeo import gdal

            entry = None
            gj = os.path.join(fire.cache_dir, 'previews', 'geo.json')
            base = os.path.splitext(os.path.basename(png_path))[0]
            if os.path.isfile(gj):
                try:
                    with open(gj, encoding='utf-8') as f:
                        entry = (_json.load(f) or {}).get(base)
                except (OSError, ValueError):
                    entry = None

            if entry is None:
                # Fall back to a raster with the same grid.
                cand = None
                m = re.match(r'^serial_(\d+)$', base)
                if m:
                    cand = os.path.join(
                        fire.cache_dir,
                        f'{fire.fire_numbe}_serial_{m.group(1)}'
                        f'_classified.bin')
                if not cand or not os.path.isfile(cand):
                    cand = fire.crop_bin
                if cand and os.path.isfile(cand):
                    ds = gdal.Open(cand, gdal.GA_ReadOnly)
                    if ds is not None:
                        entry = {'gt': [float(v) for v in
                                        ds.GetGeoTransform()],
                                 'rw': ds.RasterXSize,
                                 'rh': ds.RasterYSize}
                        ds = None

            if not entry:
                return {}

            # The PNG's own dimensions must come from the PNG being
            # served, not from whatever was recorded earlier -- a
            # re-render at a different size would otherwise desync.
            pw = ph = 0
            try:
                from matplotlib.image import imread
                a = imread(png_path)
                ph, pw = int(a.shape[0]), int(a.shape[1])
            except Exception:
                pw, ph = entry.get('w', 0), entry.get('h', 0)

            # Self-check. A preview that is a COPY (result.png) can
            # carry an entry left over from an earlier render at a
            # different padding. The PNG's real dimensions are the
            # honest witness: if they disagree with the entry, find the
            # entry whose dimensions DO match and use that instead.
            # This resolves provenance from the bytes rather than
            # trusting a name, and repairs fires whose geo.json
            # predates the copy fix.
            if pw and ph and entry.get('w') and entry.get('h') and (
                    int(entry['w']) != pw or int(entry['h']) != ph):
                replacement = None
                try:
                    with open(gj, encoding='utf-8') as f:
                        allents = _json.load(f) or {}
                    for k, v in allents.items():
                        if (int(v.get('w', -1)) == pw
                                and int(v.get('h', -1)) == ph):
                            replacement = (k, v)
                            break
                except Exception:
                    replacement = None
                if replacement:
                    sys.stderr.write(
                        f'[geo] {base}: recorded size '
                        f'{entry.get("w")}x{entry.get("h")} != actual '
                        f'{pw}x{ph}; using entry '
                        f'"{replacement[0]}" instead\n')
                    entry = replacement[1]
                    base = f'{base}~{replacement[0]}'
                else:
                    sys.stderr.write(
                        f'[geo] {base}: recorded size '
                        f'{entry.get("w")}x{entry.get("h")} != actual '
                        f'{pw}x{ph} and no matching entry found -- '
                        f'split sync may be misaligned for this view\n')

            return {
                'X-Geo-GT': ','.join(f'{v:.10g}' for v in entry['gt']),
                'X-Geo-Raster': f"{entry.get('rw', 0)},"
                                f"{entry.get('rh', 0)}",
                'X-Geo-Png': f'{pw},{ph}',
                'X-Geo-Source': base,
                'Access-Control-Expose-Headers':
                    'X-Geo-GT,X-Geo-Raster,X-Geo-Png,X-Geo-Source',
            }
        except Exception as exc:
            sys.stderr.write(f'[geo] header build failed: {exc}\n')
            return {}

    def handle_api_fire_geo(self, fire_numbe):
        """Georeferencing for every raster a pane can display.

        Split-view sync needs to put the SAME GROUND under both panes.
        Matching by fraction-of-image only works when the two rasters
        cover the same extent -- mapping results are padded relative to
        the crop, so they cover MORE ground and fraction matching is
        systematically wrong (visibly so once zoomed in).

        Returning each raster's geotransform lets the client convert
        pane pixels -> native CRS -> the other pane's pixels, which is
        correct regardless of differing extents, resolutions or
        padding.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]

        def _geo(path):
            try:
                from osgeo import gdal
                ds = gdal.Open(path, gdal.GA_ReadOnly)
                if ds is None:
                    return None
                gt = ds.GetGeoTransform()
                out = {'gt': [float(v) for v in gt],
                       'w': ds.RasterXSize, 'h': ds.RasterYSize}
                ds = None
                return out
            except Exception:
                return None

        # Authoritative source: the sidecar written when each preview
        # was rendered. It knows the extent each PNG actually came
        # from, which the finished PNG itself cannot express.
        def _png_dims(path):
            try:
                from matplotlib.image import imread
                a = imread(path)
                return int(a.shape[1]), int(a.shape[0])
            except Exception:
                return 0, 0

        out = {'crop': None, 'runs': {}, 'views': {}}
        gj = os.path.join(fire.cache_dir, 'previews', 'geo.json')
        if os.path.isfile(gj):
            try:
                with open(gj, encoding='utf-8') as f:
                    out['views'] = json.load(f)
            except (OSError, ValueError):
                pass
        if fire.crop_bin and os.path.isfile(fire.crop_bin):
            out['crop'] = _geo(fire.crop_bin)

        # Backfill: a run mapped before geo.json existed has no entry,
        # so its result would fall back to the CURRENT crop's extent --
        # wrong whenever the sweep changed padding, which is the whole
        # problem. The run's own classified raster shares the grid of
        # the crop it was mapped against, so it reconstructs the
        # correct extent exactly.
        try:
            prev_dir = os.path.join(fire.cache_dir, 'previews')
            newest_run, newest_geo = None, None
            for f in sorted(os.listdir(fire.cache_dir)):
                m = re.match(
                    rf'^{re.escape(fire_numbe)}_serial_(\d+)'
                    rf'_classified\.bin$', f)
                if not m:
                    continue
                rid = m.group(1)
                key = f'serial_{rid}'
                if key in out['views']:
                    continue
                g = _geo(os.path.join(fire.cache_dir, f))
                if not g:
                    continue
                pw, ph = _png_dims(os.path.join(prev_dir, f'{key}.png'))
                entry = {'gt': g['gt'], 'rw': g['w'], 'rh': g['h'],
                         'w': pw or g['w'], 'h': ph or g['h']}
                out['views'][key] = entry
                if newest_run is None or int(rid) >= int(newest_run):
                    newest_run, newest_geo = rid, entry
            # previews/result.png is a copy of the most recent run's
            # overlay, so it shares that run's extent.
            if 'result' not in out['views'] and newest_geo:
                out['views']['result'] = dict(newest_geo)
        except OSError:
            pass

        # Serial runs keep their own classified raster, which may have
        # been produced at a different padding than the current crop.
        try:
            for f in sorted(os.listdir(fire.cache_dir)):
                m = re.match(
                    rf'^{re.escape(fire_numbe)}_serial_(\d+)'
                    rf'_classified\.bin$', f)
                if not m:
                    continue
                g = _geo(os.path.join(fire.cache_dir, f))
                if g:
                    out['runs'][m.group(1)] = g
        except OSError:
            pass
        self._send_json(out)

    def handle_api_fire_diagnose(self, fire_numbe):
        """Everything needed to explain a 0% / 0 ha outcome.

        Zero agreement with zero mapped area has several distinct
        causes that look identical from the UI: an empty hint, an empty
        classification, a hint and mask that do not overlap because
        they were produced at different paddings, an all-nodata stack,
        or a mask written with unexpected values. Guessing between them
        from the console log alone is slow, so this reports the raw
        facts for each raster involved.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        out = {'fire': fire_numbe,
               'post_source': getattr(fire, 'post_source', ''),
               'hint_mode': getattr(fire, 'hint_mode', ''),
               'status': str(getattr(fire.status, 'value', fire.status)),
               'padding_used': getattr(fire, 'padding_used', None),
               'sample_size': getattr(fire, 'sample_size', None),
               'agreement_pct': getattr(fire, 'agreement_pct', None),
               'ml_area_ha': getattr(fire, 'ml_area_ha', None),
               'rasters': {}, 'notes': []}

        def _stats(label, path):
            """Shape, geotransform and value distribution of a mask."""
            info = {'path': path, 'exists': bool(
                path and os.path.isfile(path))}
            if not info['exists']:
                out['notes'].append(f'{label}: MISSING at {path!r}')
                out['rasters'][label] = info
                return info
            try:
                from osgeo import gdal
                import numpy as np
                ds = gdal.Open(path, gdal.GA_ReadOnly)
                if ds is None:
                    info['error'] = 'gdal.Open returned None'
                    out['rasters'][label] = info
                    return info
                info['w'] = ds.RasterXSize
                info['h'] = ds.RasterYSize
                info['bands'] = ds.RasterCount
                info['gt'] = [float(v) for v in ds.GetGeoTransform()]
                a = ds.GetRasterBand(1).ReadAsArray()
                ds = None
                info['dtype'] = str(a.dtype)
                finite = np.isfinite(a)
                info['nan_px'] = int((~finite).sum())
                vals, counts = np.unique(a[finite], return_counts=True)
                # Cap: an unexpectedly continuous mask would otherwise
                # dump thousands of entries.
                if vals.size <= 12:
                    info['values'] = {str(v): int(c)
                                      for v, c in zip(vals, counts)}
                else:
                    info['values'] = f'{vals.size} distinct values'
                    info['min'] = float(vals.min())
                    info['max'] = float(vals.max())
                info['nonzero_px'] = int((a[finite] != 0).sum())
                info['total_px'] = int(a.size)
                if info['nonzero_px'] == 0:
                    out['notes'].append(
                        f'{label}: EMPTY -- no non-zero pixels')
            except Exception as exc:
                info['error'] = str(exc)
            out['rasters'][label] = info
            return info

        crop = _stats('crop_stack', fire.crop_bin)
        hint = _stats('hint', fire.hint_bin)

        from ..state import find_classified
        clf_path = find_classified(
            fire, [fire.cache_dir, os.path.dirname(fire.crop_bin or '')])
        clf = _stats('classified', clf_path or '(not found)')

        # Overlap is the usual culprit: a hint and a mask can each be
        # non-empty yet share no pixels if their extents differ.
        try:
            if (hint.get('exists') and clf.get('exists')
                    and 'gt' in hint and 'gt' in clf):
                same_grid = (
                    abs(hint['gt'][0] - clf['gt'][0]) < 1e-6
                    and abs(hint['gt'][3] - clf['gt'][3]) < 1e-6
                    and hint['w'] == clf['w'] and hint['h'] == clf['h'])
                out['hint_clf_same_grid'] = same_grid
                if not same_grid:
                    out['notes'].append(
                        'hint and classified are on DIFFERENT grids -- '
                        'agreement is computed over their overlap, and '
                        'a padding change between them is the usual '
                        'reason for 0%')
                else:
                    from osgeo import gdal
                    import numpy as np
                    a = gdal.Open(fire.hint_bin).ReadAsArray()
                    b = gdal.Open(clf_path).ReadAsArray()
                    am, bm = (a != 0), (b != 0)
                    inter = int((am & bm).sum())
                    union = int((am | bm).sum())
                    out['overlap'] = {
                        'hint_px': int(am.sum()),
                        'clf_px': int(bm.sum()),
                        'intersection_px': inter,
                        'union_px': union,
                        'iou_pct': (100.0 * inter / union) if union else 0.0,
                    }
                    if inter == 0 and am.sum() and bm.sum():
                        out['notes'].append(
                            'hint and classified are both non-empty but '
                            'do not intersect at all')
        except Exception as exc:
            out['overlap_error'] = str(exc)

        # Per-run summary: distinguishes "every run failed" from "one
        # run failed".
        try:
            runs = []
            for r in (getattr(fire, 'serial_results', None) or []):
                runs.append({k: r.get(k) for k in
                             ('run_id', 'agreement_pct', 'ml_area_ha',
                              'setting_name', 'padding')
                             if isinstance(r, dict)})
            out['serial_results'] = runs
        except Exception:
            pass

        self._send_json(out)

    def handle_api_acq_plans_status(self):
        """Progress of the acquisition-plan download."""
        try:
            from ..acq_plans import status
            self._send_json(status())
        except Exception as exc:
            self._send_json({'state': 'error', 'message': str(exc)})

    def handle_api_acq_plans_refresh(self):
        """Kick a refresh now (used by the Retry button)."""
        try:
            from ..acq_plans import refresh, status
            threading.Thread(
                target=lambda: refresh(force=True), daemon=True).start()
            self._send_json({'ok': True, 'status': status()})
        except Exception as exc:
            self._send_json({'ok': False, 'error': str(exc)})

    def handle_api_next_coverage(self, fire_numbe):
        """Expected soonest Sentinel-2 coverage of this AOI, by pass.

        An AOI is often not re-imaged all at once, so a single "next
        pass" date would be misleading. This returns the same shape as
        the L2 coverage plot -- geometry per acquisition in crop pixel
        coordinates -- so the two can be read the same way: one shows
        which part of the AOI arrived when, the other which part
        arrives next.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
            self._send_json({'error': 'AOI stack not built yet',
                             'passes': []})
            return
        try:
            from osgeo import gdal
            from ..acq_plans import next_coverage, load_cache

            if not load_cache():
                from ..acq_plans import status as _acq_status
                self._send_json({
                    'error': 'Acquisition plans have not been '
                             'downloaded yet.',
                    'status': _acq_status(),
                    'passes': []})
                return

            ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
            if ds is None:
                self._send_json({'error': 'cannot open AOI stack',
                                 'passes': []})
                return
            gt = ds.GetGeoTransform()
            w, h = ds.RasterXSize, ds.RasterYSize
            wkt = ds.GetProjection()
            ds = None

            # AOI corners in native CRS, from the stack's own
            # geotransform so it matches the previews exactly.
            x0, y0 = gt[0], gt[3]
            x1 = gt[0] + w * gt[1] + h * gt[2]
            y1 = gt[3] + w * gt[4] + h * gt[5]
            ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

            out = next_coverage(ring, wkt, gt, w, h)
            self._send_json(out)
        except Exception as exc:
            sys.stderr.write(f'[acq] next_coverage failed: {exc}\n')
            self._send_json({'error': str(exc), 'passes': []})

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

        # Honour ?src=<post source>. previews/ only ever holds the
        # CURRENTLY selected source, but each source's images are also
        # stashed in previews_<src>/. Without this the parameter would
        # be a mere cache-buster and a request for the other source
        # would be answered with the current source's image -- which
        # the client would then cache under the wrong key, showing the
        # wrong imagery after a switch.
        from urllib.parse import urlparse, parse_qs
        _q = parse_qs(urlparse(self.path).query)
        _src = (_q.get('src') or [''])[0]
        _stash_dir = None
        if re.fullmatch(r'[A-Za-z0-9_-]+', _src or ''):
            cand = os.path.join(fire.cache_dir, f'previews_{_src}')
            if os.path.isdir(cand):
                _stash_dir = cand
                cand_png = os.path.join(cand, f'{view}.png')
                if os.path.isfile(cand_png):
                    png = cand_png

        # The hint overlay depends on BOTH the post source and the hint
        # mode, so it is stored per mode as hint_<mode>.png. Previously
        # the ?hint= parameter was only a cache-buster and this handler
        # always returned previews/hint.png -- whichever mode happened
        # to have been rendered last -- which is why the two red-wins
        # masks kept looking identical.
        if view == 'hint':
            mode = (_q.get('hint') or [''])[0]
            if re.fullmatch(r'[A-Za-z0-9_-]+', mode or ''):
                # Prefer the requested source's stash; fall back to the
                # live previews dir for the current source.
                per_mode = None
                if _stash_dir:
                    c = os.path.join(_stash_dir, f'hint_{mode}.png')
                    if os.path.isfile(c):
                        per_mode = c
                if per_mode is None:
                    c = os.path.join(fire.cache_dir, 'previews',
                                     f'hint_{mode}.png')
                    if os.path.isfile(c):
                        per_mode = c
                if per_mode:
                    png = per_mode
                elif not _src or _src == getattr(
                        fire, 'post_source', 'l2'):
                    # Not rendered yet: build it now so the client gets
                    # the mode it actually asked for rather than a
                    # silently wrong image.
                    try:
                        from ..prepare import render_hint_for_mode
                        if render_hint_for_mode(fire, mode):
                            # render_hint_for_mode writes into the live
                            # previews dir for the current source.
                            png = os.path.join(
                                fire.cache_dir, 'previews',
                                f'hint_{mode}.png')
                    except Exception as exc:
                        sys.stderr.write(
                            f'[fire] hint render for {mode} failed: '
                            f'{exc}\n')

        # Timed from here: everything above is path resolution.
        # Before serving the ML result, make sure it is rendered in the
        # CURRENT AOI grid. A sweep re-preps at different paddings, so
        # an overlay written during an earlier run can describe a
        # different extent than post.png -- the misalignment. Rendering
        # it into today's grid makes the two identical by construction.
        if view == 'result':
            try:
                from ..mapping import ensure_overlay_current
                from ..state import find_classified
                _clf = find_classified(
                    fire, [fire.cache_dir,
                           os.path.dirname(fire.crop_bin or '')])
                if _clf:
                    ensure_overlay_current(fire, 'result', _clf)
            except Exception as _exc:
                sys.stderr.write(
                    f'[geo] result re-render skipped: {_exc}\n')

        _t_prev = time.time()

        # Last line of defence. Every view in a fire should sit on the
        # current AOI grid (prepare/switch/sweep-end all re-render to
        # it). If a run overlay is somehow still on an older grid --
        # a crash mid-sweep, a file restored from elsewhere -- repair
        # it here rather than serving an image that will misalign.
        if view in ('result',) or re.match(r'^serial_\d+$', view or ''):
            try:
                from osgeo import gdal
                from ..mapping import rerender_run_overlays
                ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly) \
                    if fire.crop_bin else None
                if ds is not None and os.path.isfile(png):
                    cw, ch = ds.RasterXSize, ds.RasterYSize
                    ds = None
                    from matplotlib.image import imread
                    a = imread(png)
                    # Previews are downsampled, so compare aspect
                    # rather than absolute size.
                    # Compare SIZE, not just aspect. A padded render
                    # keeps the aspect but changes the dimensions, so
                    # an aspect-only test misses exactly the case that
                    # causes the phantom zoom. Expected preview size is
                    # the crop scaled down to MAX_PREVIEW_DIM.
                    from ..preview import MAX_PREVIEW_DIM as _MPD
                    _m = min(_MPD / max(cw, ch), 1.0)
                    exp_w, exp_h = round(cw * _m), round(ch * _m)
                    got_w, got_h = int(a.shape[1]), int(a.shape[0])
                    ar_png = got_w / max(1, got_h)
                    ar_crop = cw / max(1, ch)
                    if (abs(got_w - exp_w) > 1 or abs(got_h - exp_h) > 1
                            or abs(ar_png - ar_crop) > 0.01 * max(
                                ar_png, ar_crop)):
                        sys.stderr.write(
                            f'[geo] {view}: preview is {got_w}x{got_h} '
                            f'but the current AOI implies '
                            f'{exp_w}x{exp_h} -- re-rendering\n')
                        rerender_run_overlays(fire)
            except Exception as _hexc:
                sys.stderr.write(
                    f'[geo] {view}: grid check skipped: {_hexc}\n')

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

        # Cache may have been evicted after accept. Fall back to the
        # canonical accepted dir, which now mirrors previews/ from
        # cache_dir at accept time (see prepare._accept_fire_sync).
        if not os.path.exists(png):
            canon_png = os.path.join(
                state.output_root, fire_numbe, 'previews', f'{view}.png')
            if os.path.isfile(canon_png):
                png = canon_png

        if not os.path.exists(png):
            self._send_json(
                {'error': f"Preview '{view}' not available"}, 404)
            return
        # Instrumentation must never be able to break the response.
        # A NameError in this block previously aborted the handler
        # before _send_file ran, which the browser saw as
        # ERR_EMPTY_RESPONSE and the UI reported as
        # 'View "Post-fire" not available'. Diagnostics are not worth
        # a broken endpoint.
        try:
            _sz = os.path.getsize(png)
            sys.stderr.write(
                f'[perf] /preview/{view} {fire_numbe}: '
                f'{(time.time() - _t_prev) * 1000:.0f} ms server-side, '
                f'{_sz / 1e6:.2f} MB, src={_src or "current"}, '
                f'file={os.path.basename(png)}\n')
            sys.stderr.flush()
        except Exception as exc:
            sys.stderr.write(f'[perf] preview log failed: {exc}\n')
        # Cache aggressively. Every preview URL carries ?t=<stamp>,
        # and the stamp changes whenever the server regenerates these
        # images, so the URL is content-keyed: the browser can reuse a
        # cached copy forever and will still pick up new renders,
        # because those arrive under a new URL.
        #
        # Without this a 4+ MB PNG was re-fetched on every fire open
        # and every pane change -- at the ~600 kB/s this link sustains,
        # that is ~7 s of pure re-transfer for bytes already held.
        self._send_file(png, 'image/png', cache_seconds=86400,
                        extra_headers=self._geo_headers(fire, png, view))

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

    def handle_api_status(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        f = state.fires[fire_numbe]
        with state.lock:
            payload = {'status': f.status.value, 'error': f.error_msg}
        self._send_json(payload)

    def handle_api_download(self, fire_numbe):
        """Zip the fire's canonical accepted-result directory and stream
        it back as a download. The fire must have been accepted at
        least once — this serves ``<output_root>/<fire_numbe>/``, not
        the ephemeral .web_cache working dir.

        The zip is (re)built fresh on every request rather than cached
        on disk, so it's always in sync with whatever is currently in
        the canonical dir (e.g. after a re-brush + re-accept). Built in
        a tmp file next to the canonical dir, streamed, then removed —
        nothing is left behind in <output_root> after the response.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        if not state.output_root:
            self._send_json({'error': 'output_root not configured'}, 500)
            return

        result_dir = os.path.join(state.output_root, fire_numbe)
        if not os.path.isdir(result_dir):
            self._send_json(
                {'error': 'No accepted result on disk for this fire yet. '
                          'Accept the fire first.'}, 404)
            return

        # Build the zip into a tmp path beside the canonical dir (same
        # filesystem -> the rename inside make_archive's caller-visible
        # contract is irrelevant here; we just clean up after streaming).
        # shutil.make_archive wants the destination *without* the
        # extension it appends, and writes under <output_root>/, so the
        # tmp name is collision-safe via the pid/thread suffix.
        tmp_base = os.path.join(
            state.output_root,
            f'.{fire_numbe}.{os.getpid()}.{threading.get_ident()}.tmp')
        try:
            tmp_zip = shutil.make_archive(
                tmp_base, 'zip', root_dir=state.output_root,
                base_dir=fire_numbe)
        except Exception as exc:
            self._send_json({'error': f'Zip failed: {exc}'}, 500)
            return

        try:
            size = os.path.getsize(tmp_zip)
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(size))
            self.send_header(
                'Content-Disposition',
                f'attachment; filename="{fire_numbe}.zip"')
            self.end_headers()
            with open(tmp_zip, 'rb') as f:
                shutil.copyfileobj(f, self.wfile)
        finally:
            try:
                os.remove(tmp_zip)
            except OSError:
                pass

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

        # Snapshot mutable lists under lock to avoid iteration-during-mutation
        with state.lock:
            console_lines = list(f.console_log)
            raw_serial = list(f.serial_results)
            settings_used = [
                {'label': str(s.get('label', '')),
                 'params': dict(s.get('params', {}))}
                for s in f.serial_settings
            ]

        # Clean serial results for JSON
        serial_results = []
        for r in raw_serial:
            serial_results.append({
                'run_id': r.get('run_id'),
                'setting_idx': r.get('setting_idx', 0),
                'run_idx': r.get('run_idx', 0),
                'setting_label': r.get('setting_label', ''),
                'agreement_pct': r.get('agreement_pct', -1),
                'ml_area_ha': r.get('ml_area_ha', -1),
                'error': r.get('error', ''),
                'params': r.get('params', {}),
                'is_previous': r.get('is_previous', False),
            })

        # Is class_brush.exe currently running for this fire? The
        # frontend uses this to re-adopt a rebrush that started before
        # a page refresh (or from a different browser tab).
        with _rebrush_procs_lock:
            rebrush_running = fire_numbe in _rebrush_procs

        self._send_json({
            'status': f.status.value,
            'previously_accepted': f.previously_accepted,
            'previously_accepted_agreement_pct': (
                f.previously_accepted_agreement_pct),
            'lines': console_lines,
            'last_params': f.last_params,
            'agreement_pct': f.agreement_pct,
            'ml_area_ha': f.ml_area_ha,
            'available_views': list(f.available_views),
            'serial_results': serial_results,
            'settings_used': settings_used,
            'k_runs_per_setting': int(state.k_runs_per_setting),
            'has_comparison': has_comparison,
            'has_brush_comparison': has_brush,
            'rebrush_running': rebrush_running,
        })

    # -- Progress / queue / notifications / cache / abort --

    def handle_api_progress(self, fire_numbe):
        """Live progress snapshot for a fire currently being mapped.

        Empty object when the fire is not in a running state. The poller
        in the UI uses this to render the stage-aware progress bar.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        snap = _progress_snapshot(fire)
        # Always include status so UI can decide whether to hide the bar.
        snap['status'] = fire.status.value
        # Also expose queue context — "you are waiting behind N".
        with state.lock:
            current = (dict(state.current_job)
                       if state.current_job else None)
            waiting = [dict(w) for w in state.waiting_jobs]
        snap['queue_current'] = current
        snap['queue_waiting'] = len(waiting)
        self._send_json(snap)

    def handle_api_fire_abort(self, fire_numbe):
        """Unified cancel — signals whichever job is currently active.

        Routes to rebrush cancel if a class_brush.exe is running, else
        falls through to the serial-mapping cancel semantics. Returns a
        structured summary the UI can trust without knowing the job
        type. Also records ``fire.last_cancel_reason`` for audit.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        body = self._read_body() or {}
        reason = str(body.get('reason', '') or '').strip()[:500]
        user = getattr(self, '_username', '') or ''
        with state.lock:
            fire.last_cancel_reason = (
                f'{datetime.datetime.now().isoformat(timespec="seconds")}'
                f'|{user}|{reason}' if reason else '')
        _save_fire_state()

        actions = []
        with _rebrush_procs_lock:
            proc = _rebrush_procs.get(fire_numbe)
        if proc is not None:
            try:
                proc.terminate()
                actions.append('rebrush_cancel_requested')
            except Exception:
                pass
        if fire.status == FireStatus.MAPPING:
            fire.serial_canceled = True
            # SIGTERM the CLI so _gpu_lock releases in seconds, not
            # minutes. Worker's cleanup waits on serial_accept_event
            # only when serial_accept_promoted is True (which /abort
            # does not set), so the user-cancel path here does not
            # stall on the event.
            _terminate_serial_proc(fire_numbe)
            actions.append('mapping_cancel_requested')

        if not actions:
            self._send_json(
                {'status': 'idle', 'actions': [],
                 'message': 'No running job on this fire.'}, 200)
            return
        self._send_json({'status': 'cancelling', 'actions': actions})
