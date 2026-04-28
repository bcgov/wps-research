"""Fire-list / per-fire navigation routes (home page, fire page, notes).

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


class FireListRoutes:
    """Fire-list / per-fire navigation routes (home page, fire page, notes)."""


    # -- Page handlers --

    def handle_fire_list(self):
        is_admin = (getattr(self, '_role', '') == 'admin')
        admin_link = ('<a href="/admin" class="btn" '
                      'style="font-size:11px;padding:3px 10px">'
                      'Admin</a>'
                      if is_admin else '')
        years_sorted = sorted(state.rasters_by_year)
        html = render_template('fire_list.html', {
            'raster_name': os.path.basename(state.raster_path),
            'polygon_name': '(user-drawn bbox)',
            'n_fires': str(len(state.fires)),
            'admin_link': admin_link,
            'all_years_json': json.dumps(years_sorted),
            'active_year_json': json.dumps(int(state.active_year)),
            'is_admin_json': json.dumps(bool(is_admin)),
        })
        self._send_html(html)

    def handle_fire_page(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_html('Fire not found', 404)
            return
        fire = state.fires[fire_numbe]
        # Clear the "new" badge here rather than relying on the
        # fire-list's onclick fire-and-forget fetch — page navigation
        # often cancels that request before it lands, leaving the badge
        # stuck on. Rendering this page is unambiguous proof the user
        # opened the fire.
        if getattr(fire, 'is_new', False):
            with state.lock:
                fire.is_new = False
            threading.Thread(
                target=_save_fire_state, daemon=True).start()
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

    # -- API handlers --

    def handle_api_fires(self):
        with state.lock:
            fires = [
                {
                    'fire_numbe': f.fire_numbe,
                    'fire_date': f.fire_date,
                    'fire_year': f.fire_year,
                    'fire_size_ha': f.fire_size_ha,
                    'status': f.status.value,
                    'previously_accepted': f.previously_accepted,
                    'previously_accepted_agreement_pct': (
                        f.previously_accepted_agreement_pct),
                    'agreement_pct': f.agreement_pct,
                    'ml_area_ha': f.ml_area_ha,
                    'notes': f.notes,
                    'has_override': bool(
                        getattr(f, 'recommended_override', None)),
                    'is_new': bool(getattr(f, 'is_new', False)),
                    'error_msg': f.error_msg,
                    'sub_stage': (f.progress.get('stage', '')
                                  if f.progress else ''),
                    'sub_stage_idx': (f.progress.get('stage_idx', 0)
                                      if f.progress else 0),
                    'sub_stage_total': (f.progress.get('total_stages', 0)
                                        if f.progress else 0),
                    'sub_stage_detail': (f.progress.get('detail', '')
                                         if f.progress else ''),
                }
                for f in state.fires.values()
                if not f.hidden
            ]
        self._send_json(fires)

    def handle_api_notes(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        body = self._read_body()
        if body is None:
            return
        state.fires[fire_numbe].notes = body.get('notes', '')
        _save_notes()
        self._send_json({'status': 'saved'})

    _VALID_FN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_. -]*$')

    def handle_api_remove(self, fire_numbe):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        # Refuse to hide a fire while a worker is actively using its
        # cache_dir — removing the cache out from under the running
        # subprocess causes spurious failures.
        if fire.status in (FireStatus.MAPPING, FireStatus.PREPARING):
            self._send_json(
                {'error': f'Cannot remove while {fire.status.value}'}, 409)
            return
        # Drop the .web_cache/<FIRE>/ directory so memory doesn't leak
        # on hide/unhide cycles. The canonical output dir (if the fire
        # was accepted) is preserved separately. Re-preparing on unhide
        # will rebuild the cache from scratch.
        cache_dir = fire.cache_dir
        with state.lock:
            fire.hidden = True
            fire.cache_dir = ''
            fire.crop_bin = ''
            fire.hint_bin = ''
            fire.perim_bin = ''
            fire.viirs_bin = ''
            fire.available_views = []
            fire.last_comparison = ''
        if cache_dir and os.path.isdir(cache_dir):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception:
                pass
        _save_fire_state()
        self._send_json({'status': 'removed'})

    def handle_api_unhide(self, fire_numbe):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        state.fires[fire_numbe].hidden = False
        _save_fire_state()
        self._send_json({'status': 'restored'})

    def handle_api_fires_hidden(self):
        """Return list of hidden fires (for admin restore UI)."""
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        fires = [
            {
                'fire_numbe': f.fire_numbe,
                'fire_date': f.fire_date,
                'fire_year': f.fire_year,
                'fire_size_ha': f.fire_size_ha,
                'status': f.status.value,
            }
            for f in state.fires.values()
            if f.hidden
        ]
        self._send_json(fires)

    # ====================================================================
    # VIIRS-web: new fire creation (bbox + dates → background prepare)
    # ====================================================================

    def handle_new_fire_page(self):
        """Render the bbox-drawing UI bound to the active year."""
        is_admin = (getattr(self, '_role', '') == 'admin')
        years_sorted = sorted(state.rasters_by_year)
        active = int(state.active_year)
        html = render_template('new_fire.html', {
            'active_year': str(active),
            'all_years_json': json.dumps(years_sorted),
            'active_year_json': json.dumps(active),
            'is_admin_json': json.dumps(bool(is_admin)),
            'multi_year_json': json.dumps(len(years_sorted) > 1),
        })
        self._send_html(html)

    def _resolve_year(self, y):
        try:
            yi = int(y)
        except (TypeError, ValueError):
            return None
        return yi if yi in state.rasters_by_year else None

    def handle_api_year_overview_png(self, y):
        yi = self._resolve_year(y)
        if yi is None:
            self._send_json({'error': 'unknown year'}, 404)
            return
        path = state.overview_png_by_year.get(yi)
        if not path or not os.path.isfile(path):
            self._send_json({'error': 'overview not generated'}, 404)
            return
        self._send_file(path, media_type='image/png')

    def handle_api_year_overview_meta(self, y):
        yi = self._resolve_year(y)
        if yi is None:
            self._send_json({'error': 'unknown year'}, 404)
            return
        path = state.overview_meta_by_year.get(yi)
        if not path or not os.path.isfile(path):
            self._send_json({'error': 'overview meta not generated'}, 404)
            return
        try:
            with open(path, encoding='utf-8') as f:
                meta = json.loads(f.read())
        except Exception as exc:
            self._send_json({'error': f'parse failed: {exc}'}, 500)
            return
        self._send_json(meta)

    def handle_api_fire_create(self):
        from ..validation import (
            _validate_fire_name, _validate_bbox, _validate_date_range,
            _validate_fire_date,
        )
        from ..state import FireInfo, FireStatus
        from ..viirs_worker import submit_fire

        body = self._read_body()
        if body is None:
            return

        errors = []
        # Year
        year_raw = body.get('year', state.active_year)
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            errors.append({'field': 'year', 'message': 'year must be an integer'})
            year = None
        if year is not None and year not in state.rasters_by_year:
            errors.append({'field': 'year', 'message': f'unknown year {year}'})
            year = None

        # Name
        name_raw = body.get('name', '')
        existing = list(state.fires.keys())
        try:
            name = _validate_fire_name(name_raw, existing_names=existing)
        except ValueError as exc:
            errors.append({'field': 'name', 'message': str(exc)})
            name = None

        # Bbox in raster CRS
        meta_path = (state.overview_meta_by_year.get(year)
                     if year is not None else None)
        meta = None
        if meta_path and os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding='utf-8') as f:
                    meta = json.loads(f.read())
            except Exception:
                meta = None

        bbox_clipped = None
        if year is not None and meta is not None:
            try:
                raster_extent = tuple(meta['extent_native'])
                bbox_clipped = _validate_bbox(
                    body.get('bbox_native'), raster_extent)
            except ValueError as exc:
                errors.append({'field': 'bbox_native',
                               'message': str(exc)})
            except Exception as exc:
                errors.append({'field': 'bbox_native',
                               'message': f'invalid bbox: {exc}'})

        # Dates
        start_raw = body.get('start', '')
        end_raw = body.get('end', '')
        default_start = (meta or {}).get('default_start', '') or ''
        default_end = (meta or {}).get('default_end', '') or ''
        date_pair = None
        try:
            date_pair = _validate_date_range(
                start_raw, end_raw,
                default_start=default_start,
                default_end=default_end)
        except ValueError as exc:
            errors.append({'field': 'dates', 'message': str(exc)})

        # Optional user-supplied fire_date. When blank, fall through to
        # the validated end date below.
        fire_date_raw = body.get('fire_date', '')
        fire_date_str = ''
        if fire_date_raw and str(fire_date_raw).strip():
            try:
                fire_date_str = _validate_fire_date(
                    fire_date_raw, field_name='fire_date')
            except ValueError as exc:
                errors.append({'field': 'fire_date', 'message': str(exc)})

        if errors:
            self._send_json({'errors': errors}, 400)
            return

        # Reproject bbox to WGS84 for the LAADS URL
        from ..overview import _bbox_to_wgs84
        try:
            bbox_wgs84 = _bbox_to_wgs84(
                meta['crs_wkt'], *bbox_clipped)
        except Exception as exc:
            self._send_json(
                {'errors': [{'field': 'bbox_native',
                             'message': f'WGS84 reproject failed: {exc}'}]},
                400)
            return
        if bbox_wgs84[2] <= bbox_wgs84[0] or bbox_wgs84[3] <= bbox_wgs84[1]:
            self._send_json(
                {'errors': [{'field': 'bbox_native',
                             'message': 'bbox crosses antimeridian or is '
                                        'degenerate after reprojection'}]},
                400)
            return

        start_date, end_date = date_pair

        # Atomic add under state.lock; refuse if name reappears in race.
        with state.lock:
            if any(f.lower() == name.lower() for f in state.fires.keys()):
                self._send_json(
                    {'errors': [{'field': 'name',
                                 'message': 'Fire name already in use'}]},
                    409)
                return
            fire = FireInfo(
                fire_numbe=name,
                fire_date=(fire_date_str or end_date.isoformat()),
                fire_year=year,
                fire_size_ha=0.0,
                status=FireStatus.PREPARING,
                perimeter_type='viirs',
            )
            fire.bbox_native = tuple(float(v) for v in bbox_clipped)
            fire.bbox_wgs84 = tuple(float(v) for v in bbox_wgs84)
            fire.viirs_start_date = start_date.isoformat()
            fire.viirs_end_date = end_date.isoformat()
            import threading as _t
            fire.cancel_event = _t.Event()
            state.fires[name] = fire

        # If the user previewed before confirming, seed cache_dir with
        # the cumulative shapefile so the worker skips accumulate (the
        # dominant cost). The seed is only honoured when the request's
        # year + dates + bbox match the preview's persisted meta.json
        # exactly — otherwise fire creation falls back to a full
        # accumulate, no questions asked.
        preview_id = body.get('preview_id', '') or ''
        if (preview_id and isinstance(preview_id, str)
                and re.fullmatch(r'[A-Za-z0-9_-]+', preview_id)):
            try:
                self._seed_cache_from_preview(
                    name, preview_id, year=year,
                    start_iso=start_date.isoformat(),
                    end_iso=end_date.isoformat(),
                    bbox_native=fire.bbox_native)
            except Exception as exc:
                # Best-effort — log and fall back to a fresh accumulate.
                sys.stderr.write(
                    f'[create] preview seed failed for {name!r}: '
                    f'{exc}\n')

        try:
            submit_fire(fire)
        except Exception as exc:
            with state.lock:
                fire.status = FireStatus.ERROR
                fire.error_msg = f'failed to enqueue prepare: {exc}'
            self._send_json({'error': str(exc)}, 500)
            return

        # Background the YAML rewrite — the worker will save again on
        # any state transition, and in-memory state.fires is already
        # consistent. Keeps the create response under ~10ms.
        threading.Thread(target=_save_fire_state, daemon=True).start()
        self._send_json(
            {'name': name, 'status': fire.status.value}, 202)

    def handle_api_fire_cancel_create(self, fire_numbe):
        from ..viirs_worker import cancel_fire
        from ..state import FireStatus
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        with state.lock:
            if fire.status not in (FireStatus.PREPARING,
                                    FireStatus.PENDING):
                self._send_json(
                    {'error': f'Cannot cancel: fire is {fire.status.value}',
                     'hint': 'Use /api/fire/<name>/remove to delete an '
                             'accepted fire instead.'}, 409)
                return
        # Signal the worker (cancel_event + SIGTERM any subproc) and drop
        # the FireInfo from the live registry immediately. The worker
        # cooperatively bails out at its next stage boundary and rmtrees
        # cache_dir from its WorkerCancelled handler. We also kick off
        # an async rmtree in case the worker had already finished;
        # rmtree(ignore_errors=True) is idempotent.
        cancel_fire(fire)
        with state.lock:
            state.fires.pop(fire_numbe, None)
        # _save_fire_state rewrites the whole YAML and can take a few
        # hundred ms with many fires. Do it off the request thread so
        # cancel feels instant; the in-memory state is already
        # consistent (fire popped above) so on-disk catch-up can lag.
        threading.Thread(target=_save_fire_state, daemon=True).start()
        cache_dir = os.path.join(state.output_root, '.web_cache',
                                 fire_numbe)
        if os.path.isdir(cache_dir):
            threading.Thread(
                target=lambda d=cache_dir: shutil.rmtree(
                    d, ignore_errors=True),
                daemon=True).start()
        self._send_json({'status': 'cancelled'})

    def handle_api_fire_clear_new(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        with state.lock:
            state.fires[fire_numbe].is_new = False
        _save_fire_state()
        self._send_json({'status': 'cleared'})

    def handle_api_fire_set_date(self, fire_numbe):
        """Update ``fire.fire_date`` for a not-yet-accepted fire.

        Body: ``{"fire_date": "YYYY-MM-DD"}``. Rejects if the fire is
        already accepted (the params YAML has been written and the
        canonical output dir promoted; date is locked at that point).
        """
        from ..validation import _validate_fire_date
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        body = self._read_body()
        if body is None:
            return
        try:
            fire_date_str = _validate_fire_date(
                body.get('fire_date', ''), field_name='fire_date')
        except ValueError as exc:
            self._send_json(
                {'errors': [{'field': 'fire_date',
                             'message': str(exc)}]}, 400)
            return
        with state.lock:
            fire = state.fires[fire_numbe]
            if fire.status == FireStatus.ACCEPTED:
                self._send_json(
                    {'errors': [{'field': 'fire_date',
                                 'message': 'Cannot edit date after the '
                                            'fire has been accepted'}]},
                    409)
                return
            fire.fire_date = fire_date_str
        _save_fire_state()
        self._send_json({'status': 'saved', 'fire_date': fire_date_str})

    # ====================================================================
    # VIIRS-web: hint preview (accumulate + rasterize on a user bbox)
    # ====================================================================

    _PREVIEW_LOCK = threading.Lock()
    _PREVIEW_TTL_S = 1800  # 30 min — old previews are reaped on each call

    def _preview_root(self):
        # Outside .web_cache so the cache_retention sweeper doesn't list
        # ephemeral preview dirs alongside real fires in the admin UI.
        return os.path.join(state.output_root, '_preview_cache')

    @staticmethod
    def _bbox_close(a, b, tol=1e-3):
        """True if two (xmin, ymin, xmax, ymax) tuples agree to ``tol`` in
        each component. Tolerance is in raster CRS units (metres for UTM)."""
        if not a or not b or len(a) != 4 or len(b) != 4:
            return False
        try:
            return all(abs(float(a[i]) - float(b[i])) <= tol
                       for i in range(4))
        except (TypeError, ValueError):
            return False

    def _seed_cache_from_preview(self, fire_name, preview_id, *,
                                 year, start_iso, end_iso, bbox_native):
        """Copy the preview's cumulative VIIRS shapefile into the fire's
        cache_dir so the worker skips ``accumulate``. No-op when the
        preview's meta.json doesn't match the request exactly."""
        preview_dir = os.path.join(self._preview_root(), preview_id)
        meta_path = os.path.join(preview_dir, 'meta.json')
        if not os.path.isfile(meta_path):
            return False
        try:
            with open(meta_path, encoding='utf-8') as f:
                pmeta = json.loads(f.read())
        except Exception:
            return False
        if (int(pmeta.get('year', 0) or 0) != int(year)
                or pmeta.get('start') != start_iso
                or pmeta.get('end') != end_iso
                or not self._bbox_close(pmeta.get('bbox_native'),
                                         bbox_native)):
            return False

        cache_dir = os.path.join(state.output_root, '.web_cache', fire_name)
        os.makedirs(cache_dir, exist_ok=True)
        # Copy every sidecar of the cumulative shapefile. Names follow
        # the pattern VIIRS_VNP14IMG_<startdt>_<enddt>.{shp,dbf,...}
        # written by viirs.utils.accumulate.
        copied = False
        for shp in glob.glob(os.path.join(
                preview_dir, 'VIIRS_VNP14IMG_*.shp')):
            stem = os.path.splitext(shp)[0]
            for ext in ('.shp', '.dbf', '.shx', '.prj', '.cpg'):
                src = stem + ext
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(
                        cache_dir, os.path.basename(src)))
            copied = True
            break  # one cumulative shapefile per preview
        return copied

    def _sweep_old_previews(self):
        root = self._preview_root()
        if not os.path.isdir(root):
            return
        cutoff = time.time() - self._PREVIEW_TTL_S
        for name in os.listdir(root):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            try:
                if os.path.getmtime(d) < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    def handle_api_fire_preview_hint(self):
        from ..validation import _validate_bbox, _validate_date_range
        from ..viirs_worker import (
            accumulate_for_fire, _verify_viirs_bin_nonzero,
            _compute_viirs_area_ha, WorkerError,
        )
        from ..state import FireInfo
        from batch_fire_mapping.run_fire_mapping import crop_raster
        from viirs.utils.rasterize import rasterize_shapefile
        from ..mapping import _overlay_mask_on_post
        from ..preview import generate_all_previews

        body = self._read_body()
        if body is None:
            return

        errors = []
        year_raw = body.get('year', state.active_year)
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            errors.append({'field': 'year',
                           'message': 'year must be an integer'})
            year = None
        if year is not None and year not in state.rasters_by_year:
            errors.append({'field': 'year',
                           'message': f'unknown year {year}'})
            year = None

        meta_path = (state.overview_meta_by_year.get(year)
                     if year is not None else None)
        meta = None
        if meta_path and os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding='utf-8') as f:
                    meta = json.loads(f.read())
            except Exception:
                meta = None

        bbox_clipped = None
        if year is not None and meta is not None:
            try:
                raster_extent = tuple(meta['extent_native'])
                bbox_clipped = _validate_bbox(
                    body.get('bbox_native'), raster_extent)
            except ValueError as exc:
                errors.append({'field': 'bbox_native',
                               'message': str(exc)})
            except Exception as exc:
                errors.append({'field': 'bbox_native',
                               'message': f'invalid bbox: {exc}'})

        start_raw = body.get('start', '')
        end_raw = body.get('end', '')
        default_start = (meta or {}).get('default_start', '') or ''
        default_end = (meta or {}).get('default_end', '') or ''
        date_pair = None
        try:
            date_pair = _validate_date_range(
                start_raw, end_raw,
                default_start=default_start,
                default_end=default_end)
        except ValueError as exc:
            errors.append({'field': 'dates', 'message': str(exc)})

        if errors:
            self._send_json({'errors': errors}, 400)
            return

        start_date, end_date = date_pair
        ref_raster = state.rasters_by_year[year]

        # Sweep old previews and create a new dir keyed by a short token
        with self._PREVIEW_LOCK:
            self._sweep_old_previews()
            preview_id = f'p{int(time.time() * 1000)}'
            preview_dir = os.path.join(self._preview_root(), preview_id)
            os.makedirs(preview_dir, exist_ok=True)

        # Build a throwaway FireInfo to reuse the worker helpers.
        ephemeral = FireInfo(
            fire_numbe=preview_id,
            fire_date=end_date.isoformat(),
            fire_year=year, fire_size_ha=0.0,
        )
        ephemeral.bbox_native = tuple(float(v) for v in bbox_clipped)
        ephemeral.viirs_start_date = start_date.isoformat()
        ephemeral.viirs_end_date = end_date.isoformat()
        ephemeral.cache_dir = preview_dir

        # Crop to user's bbox (no tight-crop yet — preview uses the bbox
        # the user drew, since the whole point is to verify their choice).
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = ephemeral.bbox_native
        crop_bin = os.path.join(preview_dir, 'preview_crop.bin')
        if not crop_raster(ref_raster, crop_bin,
                           crop_xmin, crop_ymin, crop_xmax, crop_ymax):
            self._send_json(
                {'errors': [{'field': 'bbox_native',
                             'message': 'GDAL crop failed'}]}, 500)
            return

        try:
            acc_shp = accumulate_for_fire(
                ephemeral, preview_dir, ref_raster)
        except WorkerError as exc:
            self._send_json(
                {'errors': [{'field': 'dates', 'message': str(exc)}]}, 200)
            return
        except Exception as exc:
            self._send_json(
                {'errors': [{'field': 'dates',
                             'message': f'accumulate failed: {exc}'}]},
                500)
            return

        crop_rast_dir = os.path.join(preview_dir, '_viirs_crop')
        os.makedirs(crop_rast_dir, exist_ok=True)
        try:
            viirs_bin = rasterize_shapefile(
                shp_path=acc_shp, ref_image=crop_bin,
                output_dir=crop_rast_dir, buffer_m=375.0,
            )
        except Exception as exc:
            self._send_json(
                {'errors': [{'field': 'dates',
                             'message': f'rasterize failed: {exc}'}]},
                500)
            return
        if not viirs_bin or not os.path.isfile(viirs_bin):
            self._send_json(
                {'errors': [{
                    'field': 'dates',
                    'message': 'No VIIRS fire pixels in this bbox + date '
                               'range. Try a larger bbox or wider dates.'
                }]}, 200)
            return

        try:
            _verify_viirs_bin_nonzero(viirs_bin)
        except Exception as exc:
            self._send_json(
                {'errors': [{'field': 'dates', 'message': str(exc)}]}, 200)
            return

        ephemeral.crop_bin = crop_bin
        ephemeral.hint_bin = viirs_bin
        ephemeral.viirs_bin = viirs_bin
        # The preview module looks at fire.cache_dir/previews/.
        generate_all_previews(crop_bin, preview_dir, preview_id)
        _overlay_mask_on_post(
            ephemeral, viirs_bin, 'hint', (0.0, 0.8, 0.2))

        area_ha = _compute_viirs_area_ha(viirs_bin)

        # Persist a small meta.json so handle_api_fire_create can validate
        # that a preview_id passed by the client still matches the form.
        # If it does, the create handler copies the cumulative .shp into
        # the fire's cache_dir and the worker skips the accumulate stage
        # (which is the dominant cost — geopandas reading hundreds of
        # per-granule shapefiles).
        try:
            meta_blob = {
                'year': year,
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'bbox_native': list(ephemeral.bbox_native),
                'acc_shp': os.path.basename(acc_shp),
            }
            with open(os.path.join(preview_dir, 'meta.json'), 'w',
                      encoding='utf-8') as f:
                f.write(json.dumps(meta_blob))
        except OSError as exc:
            sys.stderr.write(
                f'[preview] meta.json write failed: {exc}\n')

        out = {
            'preview_id': preview_id,
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'bbox_native': list(ephemeral.bbox_native),
            'area_ha': area_ha,
            'views': {
                'hint': f'/api/fire/preview_hint/{preview_id}/hint.png',
                'post': f'/api/fire/preview_hint/{preview_id}/post.png',
            },
        }
        self._send_json(out)

    def handle_api_fire_preview_hint_png(self, preview_id, view):
        # Whitelist views; only serve PNGs that exist under the preview dir.
        if view not in ('hint', 'post', 'pre'):
            self._send_json({'error': 'unknown view'}, 404)
            return
        path = os.path.join(self._preview_root(), preview_id,
                            'previews', f'{view}.png')
        if not os.path.isfile(path):
            self._send_json({'error': 'preview not found'}, 404)
            return
        self._send_file(path, media_type='image/png')
