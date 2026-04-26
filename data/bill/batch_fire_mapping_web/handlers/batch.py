"""Batch-mapping routes (status, cancel).

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


class BatchRoutes:
    """Batch-mapping routes (start, status, cancel)."""


    def handle_api_batch_map(self):
        # Batch mapping requires admin role (blocks GPU for everyone)
        if (getattr(self, '_role', '') != 'admin'
                and state.admin_password):
            self._send_json({'error': 'Admin only'}, 403)
            return
        # Atomic check-and-set to prevent duplicate batches
        with state.lock:
            if (state.batch_status
                    and state.batch_status.get('running')):
                self._send_json(
                    {'error': 'A batch is already running'}, 400)
                return
            # Mark as starting inside the lock
            state.batch_status = {'running': True, 'total': 0,
                                  'completed': 0, 'current_fire': '',
                                  'errors': []}
        # From here until the worker thread is started, any exception
        # (e.g. rfile.read raises ConnectionResetError mid-POST) must
        # clear batch_status -- otherwise the server rejects every
        # future batch with "already running" until restart.
        started = False
        try:
            body = self._read_body()
            if body is None:
                return
            fire_numbes = body.get('fire_numbes', [])
            fire_numbes = [
                fn for fn in fire_numbes
                if fn in state.fires
                and state.fires[fn].status != FireStatus.ACCEPTED
            ]
            if not fire_numbes:
                self._send_json(
                    {'error': 'No eligible fires selected'}, 400)
                return
            with state.lock:
                state.batch_status['total'] = len(fire_numbes)
            _batch_cancel.clear()
            sess_hash = self._session_hash()
            _batch_thread[0] = threading.Thread(
                target=_batch_map_worker,
                args=(fire_numbes, sess_hash),
                daemon=True)
            _batch_thread[0].start()
            started = True
            self._send_json({
                'status': 'started',
                'total': len(fire_numbes),
            })
        finally:
            if not started:
                with state.lock:
                    state.batch_status = None

    def handle_api_batch_status(self):
        with state.lock:
            snap = dict(state.batch_status) if state.batch_status else None
            if snap is not None and isinstance(snap.get('errors'), list):
                snap['errors'] = list(snap['errors'])
        self._send_json(snap or {'running': False})

    def handle_api_batch_cancel(self):
        """Cancel a running batch mapping.

        Sets the batch-wide cancel event (stops the loop between fires)
        AND flips ``serial_canceled`` on the currently-running fire so
        its in-flight sweep unwinds instead of running to completion.
        Without the second step, clicking Cancel while a fire's N×K
        sweep is in progress would make the user wait for that whole
        sweep to finish before the batch actually stops.
        """
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        with state.lock:
            batch = (dict(state.batch_status)
                     if state.batch_status else None)
        if not batch or not batch.get('running'):
            self._send_json({'error': 'No batch running'}, 400)
            return
        _batch_cancel.set()
        current = batch.get('current_fire') or ''
        if current and current in state.fires:
            fire = state.fires[current]
            if fire.status == FireStatus.MAPPING:
                fire.serial_canceled = True
                _terminate_serial_proc(current)
        self._send_json({'status': 'cancelling'})
