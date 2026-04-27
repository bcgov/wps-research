"""Rebrush routes.

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


class RebrushRoutes:
    """Rebrush routes."""


    def handle_api_rebrush(self, fire_numbe):
        """Re-run class_brush on an existing classification; no re-map.

        POST body:
          brush_size (int)         — required
          point_threshold (int)    — required
          brush_all_segments (bool)— optional, default false
          run_id (int|null)        — optional; when set, rebrushes the
                                     serial run's classified.bin and
                                     updates its per-run brush PNG copy.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if fire.status == FireStatus.MAPPING:
            self._send_json(
                {'error': 'Mapping in progress; rebrush disabled'}, 409)
            return
        body = self._read_body()
        if body is None:
            return

        try:
            bs  = _validate_param('brush_size',
                                  body.get('brush_size', 15))
            pt  = _validate_param('point_threshold',
                                  body.get('point_threshold', 10))
            bas = _validate_param('brush_all_segments',
                                  body.get('brush_all_segments', False))
        except ValueError as exc:
            self._send_json({'error': str(exc)}, 400)
            return

        run_id = body.get('run_id')
        clf_path = None         # canonical (overwritten with brushed)
        extra_brush_png = None
        if run_id not in (None, ''):
            serial_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_{run_id}_classified.bin')
            if os.path.isfile(serial_clf):
                clf_path = serial_clf
                extra_brush_png = os.path.join(
                    fire.cache_dir,
                    f'{fire_numbe}_serial_{run_id}_brush.png')
        if clf_path is None:
            for pat in (f'{fire_numbe}_crop.bin_classified.bin',
                        f'{fire_numbe}_crop_classified.bin',
                        f'{fire_numbe}_classified.bin'):
                cand = os.path.join(fire.cache_dir, pat)
                if os.path.isfile(cand):
                    clf_path = cand
                    break
        if clf_path is None:
            self._send_json(
                {'error': 'No classification found to rebrush'}, 404)
            return

        # Prefer the pre-brush sibling as brush input so we're not
        # re-brushing an already-brushed mask. Fall back to the canonical
        # file if no _raw backup exists (older runs predating the fix).
        raw_sibling = os.path.splitext(clf_path)[0] + '_raw.bin'
        source_path = raw_sibling if os.path.isfile(raw_sibling) else clf_path

        post_png = os.path.join(fire.cache_dir, 'previews', 'post.png')
        if not os.path.isfile(post_png):
            self._send_json(
                {'error': 'Post-fire preview missing; prepare the fire first'},
                404)
            return

        # AUDIT-C5: atomic claim of the _rebrush_procs[fire_numbe] slot.
        # The previous check-then-spawn left a window during which two
        # simultaneous POSTs could both pass the check, then both Popens
        # would race; the second Popen overwrote the first's registration
        # and the first subprocess became unkillable from the cancel
        # endpoint. Plant a sentinel under the lock; _run_class_brush_only
        # replaces it with the real Popen and pops it in its finally.
        with _rebrush_procs_lock:
            if fire_numbe in _rebrush_procs:
                self._send_json(
                    {'error': 'A rebrush is already running for this fire'},
                    409)
                return
            _rebrush_procs[fire_numbe] = None  # claim sentinel

        try:
            raw = _read_envi_mask(source_path)
            brushed, cancelled = _run_class_brush_only(
                source_path, int(bs), int(pt), bool(bas),
                fire_numbe=fire_numbe)

            if cancelled:
                self._send_json({
                    'status': 'cancelled',
                    'run_id': run_id,
                }, 200)
                return

            # Promote brushed result to the canonical classified.bin so
            # the ML-classification overlay, agreement%, and ML-area
            # metric reflect the new post-brush polygon. Preserve the
            # pre-brush raster as _raw.bin for subsequent rebrushes.
            # ORDER MATTERS: back up the pre-brush source BEFORE we
            # overwrite clf_path, otherwise the "raw backup" would
            # actually end up holding the brushed mask.
            if brushed is not None:
                raw_backup = os.path.splitext(clf_path)[0] + '_raw.bin'
                if source_path != raw_backup:
                    try:
                        shutil.copy2(source_path, raw_backup)
                        hdr_src = os.path.splitext(source_path)[0] + '.hdr'
                        if not os.path.isfile(hdr_src):
                            hdr_src = source_path + '.hdr'
                        if os.path.isfile(hdr_src):
                            shutil.copy2(
                                hdr_src,
                                os.path.splitext(raw_backup)[0] + '.hdr')
                    except OSError:
                        pass
                _write_envi_mask_like(brushed, clf_path, source_path)

            out_png = os.path.join(
                fire.cache_dir, f'{fire_numbe}_brush_comparison.png')
            start = getattr(fire, 'acc_start', '') or ''
            end   = getattr(fire, 'acc_end', '') or ''
            title = (f'Fire: {fire_numbe}  —  class_brush comparison '
                     f'(size={bs}, threshold={pt}, '
                     f'all_segments={bool(bas)})')
            if start or end:
                title += f'\nStart: {start}   |   End: {end}'
            _render_brush_comparison_png(raw, brushed, post_png,
                                         out_png, title)
            if extra_brush_png:
                try:
                    shutil.copy2(out_png, extra_brush_png)
                except OSError:
                    pass

            # Refresh the ML-classification overlay the UI actually
            # reads for this context. Serial runs read
            # previews/serial_{rid}.png; the main classification reads
            # previews/result.png. Also recompute the summary metrics
            # so the gallery/header reflect the new post-brush polygon.
            if brushed is not None:
                is_serial = run_id not in (None, '')
                overlay_name = (f'serial_{run_id}'
                                if is_serial else 'result')
                try:
                    _overlay_mask_on_post(
                        fire, clf_path, overlay_name, (0.9, 0.1, 0.0))
                    if not is_serial:
                        fire.agreement_pct = _compute_agreement(fire)
                        fire.ml_area_ha = _compute_ml_area(fire, clf_path)
                        # Persist the brush params onto last_params so a
                        # later accept writes them to the canonical
                        # params YAML and accepted_params.csv. Without
                        # this, an operator who rebrushes and then
                        # accepts gets a CSV row reflecting the original
                        # brush settings, not the final ones.
                        with state.lock:
                            if fire.last_params is None:
                                fire.last_params = {}
                            fire.last_params['brush_size'] = int(bs)
                            fire.last_params['point_threshold'] = int(pt)
                            fire.last_params['brush_all_segments'] = bool(bas)

                        # Regenerate the saved perimeter-style comparison
                        # PNG so the PDF report's detail pages and any
                        # viewer that reads <fire>_comparison.png reflect
                        # the rebrushed mask. Rebrush previously only
                        # updated previews/result.png + _brush_comparison,
                        # leaving _comparison.png stale and making the
                        # PDF look "horrible" despite a correct UI.
                        comp_path = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_comparison.png')
                        if _render_comparison_png(fire, clf_path, comp_path):
                            fire.last_comparison = comp_path
                            # For already-accepted fires, also push the
                            # refreshed PNG into the canonical fire dir
                            # so a PDF generated without re-accepting
                            # still shows the current mask.
                            if fire.status == FireStatus.ACCEPTED:
                                canon = os.path.join(
                                    state.output_root, fire_numbe)
                                if os.path.isdir(canon):
                                    try:
                                        shutil.copy2(
                                            comp_path,
                                            os.path.join(
                                                canon,
                                                f'{fire_numbe}_comparison.png'))
                                    except OSError:
                                        pass
                    else:
                        new_area = _compute_ml_area(fire, clf_path)
                        new_agr = _compute_agreement(
                            fire, clf_path=clf_path)
                        try:
                            rid_match = int(run_id)
                        except (TypeError, ValueError):
                            rid_match = run_id
                        with state.lock:
                            for sr in fire.serial_results:
                                if sr.get('run_id') == rid_match:
                                    sr['ml_area_ha'] = new_area
                                    sr['agreement_pct'] = new_agr
                                    # Update the run's own params dict
                                    # so a subsequent serial-accept of
                                    # this run promotes the new brush
                                    # values into fire.last_params (see
                                    # handle_api_serial_accept).
                                    p = sr.get('params')
                                    if p is None:
                                        p = {}
                                        sr['params'] = p
                                    p['brush_size'] = int(bs)
                                    p['point_threshold'] = int(pt)
                                    p['brush_all_segments'] = bool(bas)
                                    break
                    _save_fire_state()
                except Exception:
                    pass
        except Exception as exc:
            self._send_json({'error': f'Rebrush failed: {exc}'}, 500)
            return
        finally:
            # AUDIT-C5: drop the sentinel if no Popen ever replaced it
            # (e.g., class_brush.exe missing or _read_envi_mask raised
            # before _run_class_brush_only's own pop in its finally).
            with _rebrush_procs_lock:
                if _rebrush_procs.get(fire_numbe) is None:
                    _rebrush_procs.pop(fire_numbe, None)

        brushed_px = int(brushed.sum()) if brushed is not None else 0
        raw_px     = int(raw.sum())
        _push_notification(
            self._session_hash(), 'success',
            f'Rebrush done — {fire_numbe}',
            f'raw={raw_px:,}px → brushed={brushed_px:,}px.',
            fire=fire_numbe)
        self._send_json({
            'status': 'ok',
            'brush_size': int(bs),
            'point_threshold': int(pt),
            'brush_all_segments': bool(bas),
            'run_id': run_id,
            'raw_px': raw_px,
            'brushed_px': brushed_px,
        })

    def handle_api_rebrush_cancel(self, fire_numbe):
        """Terminate a running class_brush.exe for this fire, if any."""
        fire_numbe = unquote(fire_numbe)
        with _rebrush_procs_lock:
            proc = _rebrush_procs.get(fire_numbe)
        if proc is None:
            self._send_json({'status': 'idle'}, 200)
            return
        try:
            proc.terminate()
        except Exception:
            pass
        self._send_json({'status': 'cancelling'}, 200)
