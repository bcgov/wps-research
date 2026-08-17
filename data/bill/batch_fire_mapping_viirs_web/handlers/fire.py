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
import tempfile
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
        # READY was missing from this list, which is how a fire could
        # sit at "ready", open fine, and then report
        # 'View "Post-fire" not available' with no way to recover:
        # nothing ever re-checked its files.
        #
        # Rather than forcing a full re-prepare for any discrepancy,
        # verify_and_repair_fire() fixes what it can cheaply -- an
        # empty view list is re-derived from the directory, missing
        # previews are regenerated from the existing stack -- and only
        # asks for a full rebuild when the stack itself is gone (which
        # is expected after a reboot, since stacks live on /ram).
        if fire.status in (FireStatus.READY, FireStatus.ACCEPTED,
                           FireStatus.MAPPED):
            try:
                from ..prepare import verify_and_repair_fire
                rep = verify_and_repair_fire(fire)
                if rep.get('needs_full_rebuild'):
                    needs_prepare = True
                if rep.get('actions'):
                    sys.stderr.write(
                        f'[verify] {fire_numbe}: '
                        f'{"; ".join(rep["actions"])}\n')
            except Exception as exc:
                sys.stderr.write(
                    f'[verify] {fire_numbe}: check failed '
                    f'({type(exc).__name__}: {exc}); falling back to '
                    f'the previous file test\n')
                post_png = (os.path.join(fire.cache_dir, 'previews',
                                         'post.png')
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
                payload = json.loads(f.read())
            # Recover the platform for sidecars written before it was
            # recorded, and persist so the work is done once.
            payload, changed = self._backfill_date_sats(fire, payload)
            if changed:
                try:
                    tmp = path + '.tmp'
                    with open(tmp, 'w', encoding='utf-8') as f:
                        json.dump(payload, f)
                    os.replace(tmp, path)
                    sys.stderr.write(
                        f'[date_plot] backfilled satellites for '
                        f'{fire_numbe}\n')
                except OSError:
                    pass
            self._send_json(payload)
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

    def _backfill_date_sats(self, fire, payload):
        """Fill in per-date satellites for sidecars built before they
        were recorded.

        The platform is in every SAFE filename, so it can be recovered
        by listing the zips for this AOI's tiles -- no re-extraction,
        no rebuild of the fire. Without this the prefix would only
        appear on AOIs created after the change, which is a poor reason
        to make someone recreate work.
        """
        try:
            dates = payload.get('dates') or []
            if not dates or all(d.get('sats') for d in dates):
                return payload, False
            from ..l2_recent import (tiles_intersecting_bbox,
                                     zips_for_tile)
            from osgeo import gdal
            ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
            if ds is None:
                return payload, False
            gt, w, h = ds.GetGeoTransform(), ds.RasterXSize, \
                ds.RasterYSize
            wkt = ds.GetProjection()
            ds = None
            bbox = (gt[0], gt[3] + h * gt[5],
                    gt[0] + w * gt[1], gt[3])
            tiles = tiles_intersecting_bbox(bbox, wkt)

            by_date = {}
            for t in tiles:
                for entry in zips_for_tile(t):
                    # (sortkey, acq_yyyymmdd, tile, path)
                    acq8 = entry[1]
                    path = entry[-1]
                    base = os.path.basename(str(path))
                    sat = base[:3].upper()
                    if sat.startswith('S2'):
                        by_date.setdefault(acq8, set()).add(sat)

            changed = False
            for d in dates:
                if d.get('sats'):
                    continue
                sats = sorted(by_date.get(str(d.get('date')), []))
                if sats:
                    d['sats'] = sats
                    changed = True
            return payload, changed
        except Exception as exc:
            sys.stderr.write(
                f'[date_plot] satellite backfill skipped: {exc}\n')
            return payload, False

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
            'exclude_b8': bool(getattr(fire, 'exclude_b8', True)),
            'exclude_pre_fire': bool(
                getattr(fire, 'exclude_pre_fire', True)),
            'exclude_diff': bool(getattr(fire, 'exclude_diff', True)),
            'diff_only': bool(getattr(fire, 'diff_only', False)),
            'clip_to_bcws': bool(getattr(fire, 'clip_to_bcws', False)),
            # Everything the UI must restore when a fire is re-opened,
            # so the page comes back configured as it was left rather
            # than at defaults that do not match the stored result.
            'restrict_hint_bcws': bool(
                getattr(fire, 'restrict_hint_bcws', False)),
            'scaling': dict(getattr(fire, 'scaling', None) or {}),
            'band_override': list(
                getattr(fire, 'band_override', None) or []),
            'kgc_params': dict(getattr(fire, 'kgc_params', None) or {}),
            'ui_state': dict(getattr(fire, 'ui_state', None) or {}),
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

    def handle_api_acq_plans_diag(self):
        """Full acquisition-plan diagnostics (why is a satellite
        missing?)."""
        try:
            from ..acq_plans import diagnostics
            self._send_json(diagnostics())
        except Exception as exc:
            self._send_json({'error': str(exc)})

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
        # Produce the pre-brush layer the first time it is asked for.
        # Cheaper than rendering it eagerly for every fire, and it means
        # the layer works for results produced before it existed.
        if view == 'result_prebrush':
            _pb = os.path.join(fire.cache_dir, 'previews',
                               'result_prebrush.png')
            if not os.path.isfile(_pb):
                try:
                    from ..erase import render_prebrush_overlay
                    render_prebrush_overlay(fire)
                except Exception as _pexc:
                    sys.stderr.write(
                        f'[prebrush] on-demand render failed: '
                        f'{_pexc}\n')

        _stash_dir = None
        _cur_src = getattr(fire, 'post_source', 'l2') or 'l2'
        if re.fullmatch(r'[A-Za-z0-9_-]+', _src or ''):
            cand = os.path.join(fire.cache_dir, f'previews_{_src}')
            if os.path.isdir(cand):
                _stash_dir = cand
                cand_png = os.path.join(cand, f'{view}.png')
                if os.path.isfile(cand_png):
                    png = cand_png
            elif _src != _cur_src:
                # A source was explicitly requested, it is NOT the one
                # currently loaded, and no stash exists for it.
                #
                # Falling through here served the CURRENT source's
                # previews instead -- identical pixels under the other
                # source's label, which looks like two sources that
                # happen to agree rather than a missing stash. That is
                # the worst possible failure for a comparison view.
                #
                # Build it on demand: switch the fire to that source
                # (which stashes the current previews and renders the
                # other set), then switch back so nothing else is
                # disturbed. Costs one stack build the first time and
                # nothing afterwards.
                sys.stderr.write(
                    f'[preview] {fire_numbe}: no previews_{_src} '
                    f'stash; generating it on demand\n')
                try:
                    from ..prepare import switch_post_source
                    # switch_post_source() repoints fire.post_source
                    # and fire.crop_bin. Mark the fire as prebuilding
                    # and pin user_post_source so a concurrent
                    # /prepare reports the source the USER is on rather
                    # than the transient one -- the same race that made
                    # new fires open on MRAP.
                    fire.prebuilding = True
                    fire.user_post_source = _cur_src
                    r1 = switch_post_source(fire, _src)
                    if not r1.get('ok'):
                        raise RuntimeError(
                            r1.get('error', 'switch failed'))
                    r2 = switch_post_source(fire, _cur_src)
                    fire.prebuilding = False
                    if not r2.get('ok'):
                        sys.stderr.write(
                            f'[preview] {fire_numbe}: could not switch '
                            f'back to {_cur_src}: '
                            f'{r2.get("error")}\n')
                    cand = os.path.join(fire.cache_dir,
                                        f'previews_{_src}')
                    if os.path.isdir(cand):
                        _stash_dir = cand
                        cand_png = os.path.join(cand, f'{view}.png')
                        if os.path.isfile(cand_png):
                            png = cand_png
                            sys.stderr.write(
                                f'[preview] {fire_numbe}: '
                                f'previews_{_src} built\n')
                except Exception as exc:
                    # Always clear the flag, or the fire would look
                    # permanently mid-prebuild after one failure.
                    try:
                        fire.prebuilding = False
                        if getattr(fire, 'post_source', '') != _cur_src:
                            switch_post_source(fire, _cur_src)
                    except Exception:
                        pass
                    # Report rather than serve the wrong source: a
                    # visibly missing image is recoverable, a silently
                    # wrong one is not.
                    sys.stderr.write(
                        f'[preview] {fire_numbe}: could not build '
                        f'previews_{_src}: {type(exc).__name__}: '
                        f'{exc}\n')
                    self._send_json(
                        {'error': f'No {_src.upper()} imagery is '
                                  f'available for this fire yet '
                                  f'({exc}). Switch the left pane to '
                                  f'{_src.upper()} once to build it.'},
                        409)
                    return

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
        # Prefer the JPEG twin for continuous-tone views: same picture,
        # roughly an order of magnitude fewer bytes. Falls back to the
        # PNG whenever the twin is missing (older fires, mask views, or
        # a failed JPEG encode), so this can never break a view.
        serve_path, serve_type, fmt = png, 'image/png', 'png'

        # ?lowres=1 asks for the small proxy: same scene, tens of kB,
        # used for the first paint while the full image downloads.
        if (_q.get('lowres', [''])[0] or '') in ('1', 'true', 'yes'):
            _low = os.path.splitext(png)[0] + '.low.jpg'
            if os.path.isfile(_low):
                try:
                    _lhdrs = self._geo_headers(fire, png, view)
                except Exception:
                    _lhdrs = {}
                _lhdrs['X-Preview-Format'] = 'low'
                self._send_file(_low, 'image/jpeg', cache_seconds=86400,
                                extra_headers=_lhdrs)
                return
            # No proxy (older fire, or generation failed): fall through
            # and serve the full image rather than nothing.
            sys.stderr.write(
                f'[preview] no low proxy for {view}; serving full\n')
        try:
            from ..preview import JPEG_VIEWS
            _b = os.path.splitext(os.path.basename(png))[0]
            if _b in JPEG_VIEWS:
                _jpg = os.path.splitext(png)[0] + '.jpg'
                if os.path.isfile(_jpg):
                    # Only if it is actually smaller -- a pathological
                    # JPEG bigger than its PNG would be a silent
                    # pessimisation.
                    if os.path.getsize(_jpg) < os.path.getsize(png):
                        serve_path, serve_type, fmt = (
                            _jpg, 'image/jpeg', 'jpeg')
        except Exception as _jexc:
            sys.stderr.write(f'[preview] JPEG selection skipped: '
                             f'{_jexc}\n')

        try:
            _sz = os.path.getsize(serve_path)
            _png_sz = os.path.getsize(png)
            sys.stderr.write(
                f'[perf] /preview/{view} {fire_numbe}: '
                f'{(time.time() - _t_prev) * 1000:.0f} ms server-side, '
                f'{_sz / 1e6:.2f} MB as {fmt}'
                + (f' (PNG would be {_png_sz / 1e6:.2f} MB, '
                   f'{_png_sz / max(1, _sz):.1f}x)' if fmt == 'jpeg'
                   else '')
                + f', src={_src or "current"}, '
                f'file={os.path.basename(serve_path)}\n')
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
        _hdrs = self._geo_headers(fire, png, view)
        # Report the chosen format so the browser console can show it;
        # a silent fallback to PNG would look like the optimisation
        # simply not working.
        _hdrs['X-Preview-Format'] = fmt
        # Which source's pixels these actually are. If it disagrees
        # with what was asked for, the client says so loudly instead of
        # the panes silently showing the same picture twice.
        _hdrs['X-Preview-Source'] = (
            _src if _stash_dir else _cur_src)
        _hdrs['X-Preview-Requested-Source'] = _src or _cur_src
        try:
            _hdrs['X-Preview-Png-Bytes'] = str(os.path.getsize(png))
        except OSError:
            pass
        _hdrs['Access-Control-Expose-Headers'] = (
            _hdrs.get('Access-Control-Expose-Headers', '')
            + ',X-Preview-Format,X-Preview-Png-Bytes'
            + ',X-Preview-Source,X-Preview-Requested-Source').strip(',')
        self._send_file(serve_path, serve_type, cache_seconds=86400,
                        extra_headers=_hdrs)

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

    def _apply_band_flag_changes(self, fire, changes) -> None:
        """Fold checkbox changes into the current band selection."""
        from osgeo import gdal
        from ..band_select import (apply_flag_change, select_bands)

        if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
            return
        ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
        if ds is None:
            return
        names = [ds.GetRasterBand(i + 1).GetDescription() or ''
                 for i in range(ds.RasterCount)]
        ds = None

        cur = list(getattr(fire, 'band_override', None) or [])
        if not cur:
            # No explicit selection yet: start from what the flags
            # produced BEFORE this change, so the change is a delta on
            # what the user was actually looking at.
            prev = {k: bool(getattr(fire, k, None))
                    for k, _ in changes}
            for k, v in changes:
                setattr(fire, k, not v)          # undo, momentarily
            base = select_bands(
                names,
                bool(getattr(fire, 'exclude_b8', True)),
                bool(getattr(fire, 'exclude_pre_fire', True)),
                bool(getattr(fire, 'exclude_diff', True)),
                bool(getattr(fire, 'diff_only', False)))
            for k in prev:
                setattr(fire, k, prev[k])        # restore
            cur = list(base['keep'])

        for key, turned_on in changes:
            cur = apply_flag_change(
                names, cur, key, turned_on,
                log=lambda m: fire.console_log.append(m))

        with state.lock:
            fire.band_override = sorted(set(int(i) for i in cur))

    def handle_api_bands(self, fire_numbe):
        """List the AOI stack's bands and which are currently selected.

        GET returns every band with its current state, so the picker
        opens showing exactly what the classifier would receive right
        now -- whether that comes from the checkboxes or from an
        earlier custom selection.

        POST sets a custom selection (or clears it when 'indices' is
        absent), which then overrides the checkboxes until one of them
        is toggled.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]

        if self.command == 'POST':
            try:
                body = self._read_body() or {}
            except Exception:
                body = {}
            if 'indices' in body:
                try:
                    idx = sorted({int(i) for i in (body['indices'] or [])})
                except (TypeError, ValueError):
                    self._send_json({'error': 'bad indices'}, 400)
                    return
                with state.lock:
                    fire.band_override = idx
                sys.stderr.write(
                    f'[bands] {fire_numbe}: custom selection of '
                    f'{len(idx)} band(s)\n')
            else:
                with state.lock:
                    fire.band_override = []
                sys.stderr.write(
                    f'[bands] {fire_numbe}: custom selection cleared\n')
            try:
                from ..persistence import _save_fire_state
                _save_fire_state()
            except Exception:
                pass
            self._send_json({'ok': True,
                             'override': list(fire.band_override)})
            return

        try:
            from osgeo import gdal
            from ..band_select import select_bands
            if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
                self._send_json(
                    {'error': 'The AOI stack is not built yet.'}, 409)
                return
            ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
            if ds is None:
                self._send_json(
                    {'error': 'cannot open the AOI stack'}, 500)
                return
            names = [ds.GetRasterBand(i + 1).GetDescription() or
                     f'band {i + 1}' for i in range(ds.RasterCount)]
            ds = None
            override = list(getattr(fire, 'band_override', None) or [])
            sel = select_bands(
                names,
                bool(getattr(fire, 'exclude_b8', True)),
                bool(getattr(fire, 'exclude_pre_fire', True)),
                bool(getattr(fire, 'exclude_diff', True)),
                bool(getattr(fire, 'diff_only', False)),
                override=override)
            keep = set(sel['keep'])
            self._send_json({
                'bands': [{'index': i, 'name': nm,
                           'selected': i in keep}
                          for i, nm in enumerate(names)],
                'custom': bool(override),
                'summary': sel.get('summary', ''),
            })
        except Exception as exc:
            sys.stderr.write(
                f'[bands] {fire_numbe}: {type(exc).__name__}: {exc}\n')
            self._send_json({'error': str(exc)}, 500)

    def handle_api_erase(self, fire_numbe):
        """Apply manual eraser strokes to the classification."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        try:
            body = self._read_body() or {}
        except Exception:
            body = {}
        boxes = body.get('boxes') or []
        try:
            size = int(body.get('size', 11) or 11)
        except (TypeError, ValueError):
            size = 11
        if not isinstance(boxes, list) or not boxes:
            self._send_json({'error': 'No stroke points supplied'}, 400)
            return
        # Bound the payload: a long drag can generate thousands of
        # points, and every one is a slice assignment.
        boxes = boxes[:20000]

        try:
            from ..erase import (active_classified, apply_erase,
                                 refresh_after_edit)
            clf = active_classified(fire)
            if not clf:
                self._send_json(
                    {'error': 'No ML classification to edit. Run a '
                              'mapping first.'}, 409)
                return
            res = apply_erase(
                fire, clf, boxes, size,
                log=lambda m: fire.console_log.append(m),
                outside_bcws_only=bool(
                    body.get('outside_bcws_only', False)))
            if not res.get('ok'):
                self._send_json(
                    {'error': res.get('error', 'erase failed')}, 500)
                return
            res.update(refresh_after_edit(
                fire, clf, log=lambda m: fire.console_log.append(m)))
            self._send_json(res)
        except Exception as exc:
            sys.stderr.write(
                f'[erase] {fire_numbe}: {type(exc).__name__}: {exc}\n')
            self._send_json({'error': str(exc)}, 500)

    def handle_api_bcws_mask(self, fire_numbe):
        """Binary BCWS perimeter as a PNG, for the eraser's preview.

        Optional ?w=&h= renders it at the preview's dimensions so the
        client can index it directly against the displayed image.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        q = parse_qs(urlparse(self.path).query)

        def _int(name):
            try:
                return int((q.get(name, ['0']) or ['0'])[0])
            except (TypeError, ValueError):
                return 0

        w, h = _int('w'), _int('h')
        try:
            from ..erase import perimeter_mask_png
            out = os.path.join(fire.cache_dir, 'previews',
                               f'_bcws_mask_{w}x{h}.png')
            perimeter_mask_png(fire, out, w or None, h or None)
            self._send_file(out, 'image/png', cache_seconds=0)
        except Exception as exc:
            self._send_json({'error': str(exc)}, 409)

    def handle_api_erase_revert(self, fire_numbe):
        """Undo every eraser stroke since the mask was last produced."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        try:
            from ..erase import (active_classified, revert,
                                 refresh_after_edit)
            clf = active_classified(fire)
            if not clf:
                self._send_json(
                    {'error': 'No ML classification to revert.'}, 409)
                return
            res = revert(fire, clf,
                         log=lambda m: fire.console_log.append(m))
            if not res.get('ok'):
                self._send_json(
                    {'error': res.get('error', 'revert failed')}, 409)
                return
            res.update(refresh_after_edit(
                fire, clf, log=lambda m: fire.console_log.append(m)))
            self._send_json(res)
        except Exception as exc:
            sys.stderr.write(
                f'[erase] {fire_numbe}: revert: '
                f'{type(exc).__name__}: {exc}\n')
            self._send_json({'error': str(exc)}, 500)

    def handle_api_fire_rename(self, fire_numbe):
        """Rename a fire, moving its directories with it."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        try:
            body = self._read_body() or {}
        except Exception:
            body = {}
        new_name = (body.get('name') or '').strip()
        try:
            from ..prepare import rename_fire
            res = rename_fire(fire_numbe, new_name)
        except Exception as exc:
            sys.stderr.write(
                f'[rename] {fire_numbe}: {type(exc).__name__}: {exc}\n')
            self._send_json({'error': str(exc)}, 500)
            return
        if not res.get('ok'):
            # 409: the usual failure is a name already in use, which is
            # a conflict rather than a malformed request.
            self._send_json({'error': res.get('error', 'Rename failed')},
                            409)
            return
        self._send_json(res)

    def handle_api_exclude_b8(self, fire_numbe):
        """Set whether B8/B8A are withheld from ML and from export."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        try:
            body = self._read_body() or {}
        except Exception:
            body = {}
        # One endpoint for all three exclusions: they are the same
        # kind of setting and always applied together, so a single
        # round trip keeps them consistent.
        with state.lock:
            changed = []
            # Non-boolean settings that share this endpoint. Each is
            # merged or replaced independently, so a client posting one
            # of them never disturbs the others.
            if isinstance(body.get('scaling'), dict):
                fire.scaling = dict(body['scaling'])
            if isinstance(body.get('kgc_params'), dict):
                fire.kgc_params = dict(body['kgc_params'])
            if isinstance(body.get('ui_state'), dict):
                # Merged, not replaced: the client posts only what
                # changed, and a replace would drop settings saved a
                # moment earlier by another control.
                _ui = dict(getattr(fire, 'ui_state', None) or {})
                _ui.update(body['ui_state'])
                fire.ui_state = _ui
            if 'restrict_hint_bcws' in body:
                fire.restrict_hint_bcws = bool(body['restrict_hint_bcws'])
            for key in ('exclude_b8', 'exclude_pre_fire',
                        'exclude_diff', 'diff_only', 'clip_to_bcws'):
                if key in body:
                    if bool(getattr(fire, key, None)) != bool(body[key]):
                        changed.append((key, bool(body[key])))
                    setattr(fire, key, bool(body[key]))

        # Apply each rule change INCREMENTALLY to the selection in
        # force, rather than recomputing it from all the flags.
        #
        # Recomputing would discard a hand-picked selection every time
        # any box was clicked -- and would also "fix" combinations the
        # user deliberately created. Adding or removing only the bands
        # the clicked box governs leaves every other choice alone,
        # which is the behaviour asked for.
        band_flags = [c for c in changed
                      if c[0] != 'clip_to_bcws']
        if band_flags:
            try:
                self._apply_band_flag_changes(fire, band_flags)
            except Exception as exc:
                sys.stderr.write(
                    f'[bands] {fire_numbe}: incremental update failed '
                    f'({exc}); the checkbox rules still apply\n')
        val = bool(getattr(fire, 'exclude_b8', True))
        try:
            from ..persistence import _save_fire_state
            _save_fire_state()
        except Exception:
            pass
        flags = {k: bool(getattr(fire, k, dflt))
                 for k, dflt in (('exclude_b8', True),
                                 ('exclude_pre_fire', True),
                                 ('exclude_diff', True),
                                 ('diff_only', False),
                                 ('clip_to_bcws', False),
                                 ('restrict_hint_bcws', False))}
        flags['scaling'] = dict(getattr(fire, 'scaling', None) or {})
        flags['band_override'] = list(
            getattr(fire, 'band_override', None) or [])
        sys.stderr.write(f'[bands] {fire_numbe}: {flags}\n')
        self._send_json({'ok': True, **flags})

    # Frames arrive as raw PNG bytes and are megabytes each, so this
    # endpoint reads the body itself rather than through _read_body(),
    # whose 1 MB JSON limit exists to protect the small API surface.
    _GIF_MAX_BODY = 80 * 1024 * 1024

    def handle_api_interlaced_gif(self, fire_numbe):
        """Blink GIF built from the frames the panes are DISPLAYING.

        The client posts the two rendered images rather than naming
        views for the server to look up. That is the only way to get
        what was actually asked for:

          * The overlays -- tile grid, BCWS perimeter, labels -- are
            composited in the browser, so a server-side preview PNG
            does not contain them.
          * A pane can show a view that has no per-source preview file
            (the ML classification exists only for the source that
            produced it), which is what made the lookup fail with
            "no preview available" for panes that were plainly on
            screen.

        Frames are sent at natural resolution, so the GIF is the whole
        image rather than the zoomed and cropped viewport.

        Body: multipart-free, two PNGs separated by a boundary we
        control -- see the length-prefixed framing below.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return

        try:
            length = int(self.headers.get('Content-Length', 0) or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            self._send_json({'error': 'No frames were posted'}, 400)
            return
        if length > self._GIF_MAX_BODY:
            self._send_json(
                {'error': f'Frames are {length / 1e6:.0f} MB, over the '
                          f'{self._GIF_MAX_BODY / 1e6:.0f} MB limit.'},
                413)
            return

        raw = self.rfile.read(length)

        # Framing: 4-byte big-endian length, then that many bytes, per
        # frame. Simple, exact, and free of multipart parsing.
        frames = []
        off = 0
        try:
            while off + 4 <= len(raw) and len(frames) < 8:
                n = int.from_bytes(raw[off:off + 4], 'big')
                off += 4
                if n <= 0 or off + n > len(raw):
                    break
                frames.append(raw[off:off + n])
                off += n
        except Exception:
            frames = []
        if len(frames) < 2:
            self._send_json(
                {'error': f'Expected 2 frames, decoded {len(frames)}.'},
                400)
            return

        q = parse_qs(urlparse(self.path).query)

        def _arg(name, default=''):
            return (q.get(name, [default]) or [default])[0]

        try:
            ms = max(120, min(5000, int(_arg('ms', '700'))))
        except (TypeError, ValueError):
            ms = 700
        left_cap = _arg('left_label', 'left')
        right_cap = _arg('right_label', 'right')

        try:
            from PIL import Image, ImageDraw
        except ImportError:
            self._send_json(
                {'error': 'Pillow is not installed on the server, so '
                          'GIFs cannot be written.'}, 500)
            return

        import io
        tmp_dir = None
        try:
            f1 = Image.open(io.BytesIO(frames[0])).convert('RGB')
            f2 = Image.open(io.BytesIO(frames[1])).convert('RGB')

            # Frames must match exactly or the animation jitters. Both
            # panes show the same AOI, so any difference is preview
            # downsampling; match the larger to the smaller.
            w = min(f1.width, f2.width)
            h = min(f1.height, f2.height)
            if (f1.width, f1.height) != (w, h):
                f1 = f1.resize((w, h), Image.LANCZOS)
            if (f2.width, f2.height) != (w, h):
                f2 = f2.resize((w, h), Image.LANCZOS)

            def _label(im, text):
                d = ImageDraw.Draw(im)
                d.rectangle([0, im.height - 22, im.width, im.height],
                            fill=(10, 12, 16))
                d.text((8, im.height - 16), text, fill=(200, 215, 235))
                return im

            f1 = _label(f1, f'{fire_numbe}  -  {left_cap}')
            f2 = _label(f2, f'{fire_numbe}  -  {right_cap}')

            tmp_dir = tempfile.mkdtemp(prefix=f'{fire_numbe}_gif_')
            safe = re.sub(r'[^A-Za-z0-9_.-]+', '_',
                          f'{left_cap}__{right_cap}')[:80]
            out = os.path.join(tmp_dir, f'{fire_numbe}_{safe}_blink.gif')
            f1.save(out, save_all=True, append_images=[f2],
                    duration=ms, loop=0, optimize=True)

            size = os.path.getsize(out)
            self.send_response(200)
            # octet-stream so nothing renders it inline; this response
            # only ever exists to be saved.
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(size))
            self.send_header(
                'Content-Disposition',
                f'attachment; filename="{os.path.basename(out)}"')
            self.send_header('Access-Control-Expose-Headers',
                             'Content-Disposition')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with open(out, 'rb') as fh:
                shutil.copyfileobj(fh, self.wfile)
            sys.stderr.write(
                f'[gif] {fire_numbe}: {w}x{h}, {ms} ms/frame, '
                f'{size / 1e6:.2f} MB  [{left_cap} | {right_cap}]\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            sys.stderr.write(
                f'[gif] {fire_numbe}: failed: '
                f'{type(exc).__name__}: {exc}\n')
            try:
                self._send_json({'error': str(exc)}, 500)
            except Exception:
                pass
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def handle_api_download_imagery(self, fire_numbe):
        """Export the pre/post reflectance bands as an ENVI stack.

        Eight bands -- B12, B11, B9, B8 for pre and for post -- taken
        from whichever AOI stack corresponds to the requested source,
        written as ENVI BSQ float32 with band names and the map info /
        CRS the source carries. Zipped with its .hdr, since the pair is
        useless apart.

        The anomaly bands are deliberately excluded: they are derived,
        and anyone wanting them can compute them from what is here.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]

        q = parse_qs(urlparse(self.path).query)
        want_src = (q.get('src', [''])[0] or '').strip().lower()
        if want_src not in ('l2', 'mrap'):
            want_src = getattr(fire, 'post_source', 'l2') or 'l2'

        try:
            from ..aoi_stack import ensure_aoi_stack
            from osgeo import gdal
            import numpy as np

            # Stacks are per-source; ask for the one the caller wants
            # rather than assuming the fire's current selection, which
            # is what the right-hand pane may be overriding.
            info = ensure_aoi_stack(
                fire.fire_numbe, fire.bbox_native,
                post_source=want_src)
            stack_path = info['path'] if isinstance(info, dict) else info
        except Exception as exc:
            self._send_json(
                {'error': f'Could not obtain the {want_src.upper()} '
                          f'AOI stack: {type(exc).__name__}: {exc}'}, 500)
            return

        # Bound before the try, so the finally block cannot raise
        # NameError when an early failure happens before mkdtemp.
        tmp_dir = None
        try:
            ds = gdal.Open(stack_path, gdal.GA_ReadOnly)
            if ds is None:
                self._send_json(
                    {'error': f'Could not open {stack_path}'}, 500)
                return
            names = []
            for i in range(1, ds.RasterCount + 1):
                names.append(ds.GetRasterBand(i).GetDescription() or '')

            # Export EXACTLY the bands the classifier receives, by
            # calling the SAME selector. Sharing the implementation is
            # the only reliable way to keep the archive and the model
            # input identical; two parallel band lists drifted before.
            from ..band_select import select_bands
            excl = (bool(getattr(fire, 'exclude_b8', True)),
                    bool(getattr(fire, 'exclude_pre_fire', True)),
                    bool(getattr(fire, 'exclude_diff', True)),
                    bool(getattr(fire, 'diff_only', False)))
            sel = select_bands(
                names, *excl,
                override=list(getattr(fire, 'band_override', None)
                              or []))
            picks = [(i_b, names[i_b]) for i_b in sel['keep']]
            if not picks:
                self._send_json(
                    {'error': f'No bands to export from '
                              f'{os.path.basename(stack_path)}. Bands '
                              f'present: {"; ".join(names)}'}, 500)
                ds = None
                return
            sys.stderr.write(
                f'[imagery] {fire_numbe}: {sel["summary"]}\n')

            w, h = ds.RasterXSize, ds.RasterYSize
            drv = gdal.GetDriverByName('ENVI')
            tmp_dir = tempfile.mkdtemp(prefix=f'{fire_numbe}_img_')
            base = f'{fire_numbe}_{want_src}_prepost'
            out_bin = os.path.join(tmp_dir, base + '.bin')
            # INTERLEAVE=BSQ and Float32 as specified.
            # No SUFFIX=ADD: that writes <name>.bin.hdr, whereas ENVI
            # convention (and every tool that reads these) expects
            # <name>.hdr beside <name>.bin.
            out_ds = drv.Create(out_bin, w, h, len(picks),
                                gdal.GDT_Float32,
                                options=['INTERLEAVE=BSQ'])
            out_ds.SetGeoTransform(ds.GetGeoTransform())
            out_ds.SetProjection(ds.GetProjection())
            for out_i, (src_i, nm) in enumerate(picks, start=1):
                arr = ds.GetRasterBand(src_i + 1).ReadAsArray()
                ob = out_ds.GetRasterBand(out_i)
                ob.WriteArray(arr.astype('float32'))
                ob.SetDescription(nm)
                ob = None
            out_ds = None
            ds = None

            # Same scaling the classifier receives, applied after band
            # selection, so the archive and the model input are the
            # same product. Written in place so the filenames and the
            # header handling below are unchanged.
            try:
                from ..scaling import scale_raster, scaling_tag
                _sp = dict(getattr(fire, 'scaling', None) or {})
                if _sp and str(_sp.get('method') or '') not in (
                        '', 'none'):
                    _tmp = os.path.join(tmp_dir, base + '_scaled.bin')
                    if scale_raster(out_bin, _tmp, _sp) == _tmp:
                        for _e in ('.bin', '.hdr'):
                            _a = os.path.splitext(_tmp)[0] + _e
                            _b = os.path.splitext(out_bin)[0] + _e
                            if os.path.isfile(_a):
                                os.replace(_a, _b)
                        sys.stderr.write(
                            f'[imagery] {fire_numbe}: applied '
                            f'{scaling_tag(_sp)} scaling\n')
            except Exception as _sexc:
                sys.stderr.write(
                    f'[imagery] {fire_numbe}: scaling skipped '
                    f'({_sexc})\n')

            # GDAL writes <base>.hdr with SUFFIX=ADD; make sure band
            # names really landed, since some GDAL builds drop them.
            hdr = os.path.splitext(out_bin)[0] + '.hdr'
            if not os.path.isfile(hdr) and os.path.isfile(out_bin
                                                          + '.hdr'):
                # Some GDAL builds still append; normalise the name so
                # the archive is consistent whatever the build does.
                os.replace(out_bin + '.hdr', hdr)
            try:
                with open(hdr, encoding='utf-8') as f:
                    hdr_txt = f.read()
                if 'band names' not in hdr_txt.lower():
                    with open(hdr, 'a', encoding='utf-8') as f:
                        f.write('band names = {\n'
                                + ',\n'.join(nm for _, nm in picks)
                                + '}\n')
            except OSError:
                pass

            # Hint masks as ENVI rasters, one pair per mode.
            #
            # The hints are what the classifier is scored against, so
            # an export without them cannot be independently checked or
            # reproduced. Written in the stack's grid and CRS so they
            # overlay the imagery directly.
            try:
                from ..prepare import (build_derived_hint_for_fire,
                                       ALL_HINT_MODES)
                for mode in ALL_HINT_MODES:
                    mask_path = None
                    if mode == 'viirs':
                        mp = getattr(fire, 'viirs_bin', '') or ''
                        if mp and os.path.isfile(mp):
                            mask_path = mp
                    else:
                        mp, merr = build_derived_hint_for_fire(fire, mode)
                        if mp and os.path.isfile(mp):
                            mask_path = mp
                        elif merr:
                            sys.stderr.write(
                                f'[imagery] {fire_numbe}: hint {mode} '
                                f'unavailable: {merr}\n')
                    if not mask_path:
                        continue
                    mds = gdal.Open(mask_path, gdal.GA_ReadOnly)
                    if mds is None:
                        continue
                    hp = os.path.join(tmp_dir,
                                      f'{fire_numbe}_hint_{mode}.bin')
                    hout = drv.Create(hp, mds.RasterXSize,
                                      mds.RasterYSize, 1,
                                      gdal.GDT_Float32,
                                      options=['INTERLEAVE=BSQ'])
                    hout.SetGeoTransform(mds.GetGeoTransform())
                    hout.SetProjection(mds.GetProjection())
                    hb = hout.GetRasterBand(1)
                    hb.WriteArray(
                        mds.GetRasterBand(1).ReadAsArray().astype(
                            'float32'))
                    hb.SetDescription(f'hint {mode}')
                    hb = None
                    hout = None
                    mds = None
                    hhdr = os.path.splitext(hp)[0] + '.hdr'
                    if (not os.path.isfile(hhdr)
                            and os.path.isfile(hp + '.hdr')):
                        os.replace(hp + '.hdr', hhdr)
                    try:
                        with open(hhdr, encoding='utf-8') as f:
                            t = f.read()
                        if 'band names' not in t.lower():
                            with open(hhdr, 'a', encoding='utf-8') as f:
                                f.write('band names = {\n'
                                        f'hint {mode}' + '}\n')
                    except OSError:
                        pass
            except Exception as hexc:
                sys.stderr.write(
                    f'[imagery] {fire_numbe}: hint export skipped: '
                    f'{type(hexc).__name__}: {hexc}\n')

            import zipfile
            zip_path = os.path.join(tmp_dir, base + '.zip')
            with zipfile.ZipFile(zip_path, 'w',
                                 zipfile.ZIP_DEFLATED) as zf:
                for f_ in sorted(os.listdir(tmp_dir)):
                    # .aux.xml is GDAL bookkeeping (statistics etc.),
                    # not part of the ENVI product, and only confuses
                    # whoever opens the archive.
                    if f_.endswith('.zip') or f_.endswith('.aux.xml'):
                        continue
                    zf.write(os.path.join(tmp_dir, f_), f_)

            size = os.path.getsize(zip_path)
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(size))
            self.send_header(
                'Content-Disposition',
                f'attachment; filename="{base}.zip"')
            self.end_headers()
            with open(zip_path, 'rb') as f:
                shutil.copyfileobj(f, self.wfile)
            sys.stderr.write(
                f'[imagery] {fire_numbe}: exported {len(picks)} band(s) '
                f'from the {want_src.upper()} stack '
                f'({size / 1e6:.1f} MB zipped)\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            sys.stderr.write(
                f'[imagery] {fire_numbe}: export failed: '
                f'{type(exc).__name__}: {exc}\n')
            try:
                self._send_json({'error': str(exc)}, 500)
            except Exception:
                pass
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

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

        # Backfill vectors for fires accepted before vectorization
        # existed (K41351 and friends): the raster is in the canonical
        # dir but the perimeter is not, and re-accepting just to get a
        # shapefile would be an absurd requirement.
        try:
            fire = state.fires[fire_numbe]
            # Regenerate when the vectors are MISSING or STALE.
            #
            # Only checking for absence meant that a rebrush (or an
            # eraser edit) after acceptance shipped the perimeter from
            # before the edit, while the raster in the same archive
            # reflected it -- an internally inconsistent download, and
            # the vector is the part most likely to be used downstream.
            vec_files = [f for f in os.listdir(result_dir)
                         if f.endswith(('.shp', '.kml'))]
            stale_vec = False
            try:
                from ..erase import active_classified
                _live = active_classified(fire)
                if vec_files and _live and os.path.isfile(_live):
                    newest_vec = max(
                        os.path.getmtime(os.path.join(result_dir, f))
                        for f in vec_files)
                    if os.path.getmtime(_live) > newest_vec + 1.0:
                        stale_vec = True
                        sys.stderr.write(
                            f'[download] {fire_numbe}: classification '
                            f'is newer than the vector products; '
                            f'regenerating\n')
            except Exception as exc:
                sys.stderr.write(
                    f'[download] {fire_numbe}: staleness check failed '
                    f'({exc}); keeping the existing vectors\n')
            has_vec = bool(vec_files) and not stale_vec
            if not has_vec:
                from ..prepare import vectorize_classified
                clf = None
                for f in sorted(os.listdir(result_dir)):
                    if 'classified' in f and f.endswith('.bin'):
                        clf = os.path.join(result_dir, f)
                        break
                if clf:
                    sys.stderr.write(
                        f'[download] {fire_numbe}: no vector product '
                        f'in the accepted dir; generating from '
                        f'{os.path.basename(clf)}\n')
                    vres = vectorize_classified(fire, clf_path=clf)
                    # Written into cache_dir by the helper; copy the
                    # parts into the accepted dir so they ship.
                    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg',
                                '.kml'):
                        src = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_perimeter{ext}')
                        if os.path.isfile(src):
                            shutil.copy2(src, result_dir)
                    if vres.get('error'):
                        sys.stderr.write(
                            f'[download] {fire_numbe}: vectorization '
                            f'failed: {vres["error"]}\n')
        except Exception as exc:
            sys.stderr.write(
                f'[download] {fire_numbe}: vector backfill skipped: '
                f'{type(exc).__name__}: {exc}\n')

        # Refresh the accepted RASTER when the working copy is newer.
        # The vector regeneration above derives from the live mask, so
        # without this the archive could hold a perimeter and a raster
        # from different states of the same fire.
        try:
            from ..erase import active_classified
            _live = active_classified(fire)
            if _live and os.path.isfile(_live):
                _dst = os.path.join(result_dir, os.path.basename(_live))
                if (not os.path.isfile(_dst)
                        or os.path.getmtime(_live)
                        > os.path.getmtime(_dst) + 1.0):
                    for _e in ('.bin', '.hdr'):
                        _a = os.path.splitext(_live)[0] + _e
                        _b = os.path.splitext(_dst)[0] + _e
                        if os.path.isfile(_a):
                            shutil.copy2(_a, _b)
                    sys.stderr.write(
                        f'[download] {fire_numbe}: refreshed the '
                        f'accepted raster from the working copy\n')
        except Exception as exc:
            sys.stderr.write(
                f'[download] {fire_numbe}: raster refresh skipped '
                f'({exc})\n')

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
        # KGC reports its own stages and its own ETA (parsed from the
        # executable's output), so pass that through untouched.
        # _progress_snapshot() reasons about the CLI's stage order and
        # its historical timings; a kgc_* stage is not in that order,
        # so routing it through there produced an empty or meaningless
        # snapshot -- which is why the bar stayed blank while the
        # button still said "Running KGC".
        raw = dict(getattr(fire, 'progress', {}) or {})
        if str(raw.get('stage', '')).startswith('kgc_'):
            snap = raw
        else:
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
