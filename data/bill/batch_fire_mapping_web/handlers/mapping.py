"""Single-shot mapping + accept + presets/settings routes.

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


class MappingRoutes:
    """Single-shot mapping + accept + presets/settings routes."""


    def handle_api_map(self, fire_numbe):
        """Run fire_mapping_cli.py and stream output as SSE."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return

        fire = state.fires[fire_numbe]
        body = self._read_body()
        if body is None:
            return
        params = body.get('params', {})

        # AUDIT-C5: atomic test-and-claim of fire.status. Two simultaneous
        # POSTs would otherwise both pass the readiness check and both
        # enqueue, leading to duplicate mapping runs that overwrite each
        # other's last_params / agreement_pct / ml_area_ha.
        with state.lock:
            if fire.status not in (
                    FireStatus.READY, FireStatus.MAPPED,
                    FireStatus.ACCEPTED):
                _prev_status = None
            else:
                _prev_status = fire.status
                if fire.status == FireStatus.ACCEPTED:
                    fire.previously_accepted = True
                    if fire.agreement_pct >= 0:
                        fire.previously_accepted_agreement_pct = (
                            fire.agreement_pct)
                fire.status = FireStatus.MAPPING
        if _prev_status is None:
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
        fire.console_log.clear()

        def sse(event_type, data):
            payload = json.dumps({'type': event_type, **data})
            # Buffer for reconnection
            if event_type == 'log':
                fire.console_log.append(data.get('message', ''))
            try:
                self.wfile.write(f'data: {payload}\n\n'.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected. Propagate so _stream_subprocess's
                # finally block kills the running CLI instead of orphaning
                # it. Any outer sse('error', ...) will re-raise here too,
                # which is fine — the finally blocks still clean up.
                raise

        # GPU serialisation with queue tracking
        client_ip = self._client_ip()
        job_entry = {
            'fire_numbe': fire_numbe,
            'client_ip': client_ip,
            'queued_at': datetime.datetime.now().isoformat(
                timespec='seconds'),
        }

        with _gpu_queue_lock:
            queue_pos = _gpu_queue[0]
            _gpu_queue[0] += 1
            state.waiting_jobs.append(job_entry)

        try:
            # sse() can raise BrokenPipeError if the client already
            # disconnected; the surrounding try/finally guarantees the
            # queue/waiting_jobs cleanup runs even then.
            if _gpu_lock.locked():
                sse('log', {
                    'message': f'Queued — {queue_pos} job(s) ahead. '
                               f'Waiting for GPU...',
                })

            with _gpu_lock:
                # AUDIT-C5: status was already flipped to MAPPING under
                # state.lock at the top of the handler; we only need to
                # update queue bookkeeping here.
                with state.lock:
                    if job_entry in state.waiting_jobs:
                        state.waiting_jobs.remove(job_entry)
                    state.current_job = {
                        **job_entry,
                        'started_at': datetime.datetime.now().isoformat(
                            timespec='seconds'),
                    }
                cmd = _build_mapping_cmd(fire, params)
                short_cmd = ' '.join(
                    os.path.basename(c) if i < 3 else c
                    for i, c in enumerate(cmd))
                sse('log', {'message': f'$ {short_cmd}'})

                try:
                    def _on_line(text):
                        sse('log', {'message': text})

                    tracker = _ProgressTracker(
                        fire, total_runs=1, run_id=1, pipeline='full')
                    tracker.mark_run_start(1, pipeline='full')
                    rc, killed = _stream_subprocess(
                        cmd, state.project_root, _on_line,
                        tracker=tracker, fire_numbe=fire_numbe)
                    tracker.mark_run_end()

                except Exception as exc:
                    fire.status = FireStatus.READY
                    with state.lock:
                        fire.progress = {}
                    sse('error', {
                        'message': f'Failed to start subprocess: {exc}',
                    })
                    return

                if killed:
                    fire.status = FireStatus.READY
                    sse('error', {
                        'message': (
                            f'Mapping killed after '
                            f'{_SUBPROCESS_SILENCE_TIMEOUT}s of '
                            f'silent CLI output (watchdog).'),
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
                    with state.lock:
                        fire.progress = {}
                    _save_fire_state()
                    _push_notification(
                        self._session_hash(), 'success',
                        f'Mapping complete — {fire_numbe}',
                        f'Agreement: {fire.agreement_pct}%, '
                        f'ML area: {fire.ml_area_ha} ha.',
                        fire=fire_numbe,
                        action={'url': f'/fire/{fire_numbe}',
                                'label': 'Open fire'})
                    sse('complete', {
                        'comparison_url': (
                            f'/api/fire/{fire_numbe}/comparison'
                            f'?t={int(time.time())}'),
                        'agreement_pct': fire.agreement_pct,
                        'ml_area_ha': fire.ml_area_ha,
                    })
                else:
                    fire.status = FireStatus.READY
                    with state.lock:
                        fire.progress = {}
                    _push_notification(
                        self._session_hash(), 'error',
                        f'Mapping failed — {fire_numbe}',
                        f'CLI exited with code {rc}. See console for details.',
                        fire=fire_numbe)
                    sse('error', {
                        'message': (
                            f'fire_mapping_cli.py exited with code {rc}'),
                    })
        finally:
            with _gpu_queue_lock:
                _gpu_queue[0] = max(0, _gpu_queue[0] - 1)
            with state.lock:
                state.current_job = None
                if job_entry in state.waiting_jobs:
                    state.waiting_jobs.remove(job_entry)
                # If we exited via client-disconnect before rc was
                # evaluated, fire.status is still MAPPING. Unstick it so
                # the next request can run.
                if fire.status == FireStatus.MAPPING:
                    fire.status = FireStatus.READY
                fire.progress = {}

    def handle_api_settings_get(self):
        self._send_json({
            'k_runs_per_setting': int(state.k_runs_per_setting),
            'k_jitter': int(state.k_jitter),
            'settings': [
                {'label': str(s.get('label', '')),
                 'params': dict(s.get('params', {}))}
                for s in state.recommended_settings
            ],
        })

    def handle_api_settings_post(self):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        body = self._read_body()
        if body is None:
            return
        settings = body.get('settings', [])
        if not isinstance(settings, list) or not settings:
            self._send_json(
                {'error': 'settings must be a non-empty list'}, 400)
            return
        clean = []
        for row in settings:
            if not isinstance(row, dict):
                self._send_json(
                    {'error': 'each setting must be an object'}, 400)
                return
            label = str(row.get('label', '') or '').strip()
            p = dict(row.get('params', {}) or {})
            eb = p.get('embed_bands')
            if eb is not None and isinstance(eb, (int, float)):
                sys.stderr.write(
                    f'[settings] WARNING: embed_bands was numeric '
                    f'({eb!r}) — converting to empty string\n')
                p['embed_bands'] = ''
            if not label:
                label = f'setting_{len(clean) + 1}'
            clean.append({'label': label, 'params': p})

        try:
            k_runs = int(body.get(
                'k_runs_per_setting', state.k_runs_per_setting))
        except (TypeError, ValueError):
            k_runs = state.k_runs_per_setting
        try:
            k_jitter = int(body.get('k_jitter', state.k_jitter))
        except (TypeError, ValueError):
            k_jitter = state.k_jitter
        k_runs = max(1, min(10, k_runs))
        k_jitter = max(0, k_jitter)

        state.recommended_settings = clean
        state.k_runs_per_setting = k_runs
        state.k_jitter = k_jitter
        _save_settings()
        self._send_json({'status': 'saved'})

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
        if not fire.cache_dir or not os.path.isdir(fire.cache_dir):
            self._send_json(
                {'error': 'Fire cache directory is missing — '
                          're-prepare the fire and re-run mapping.'},
                400)
            return
        # Serialize accept against any running mapping/brush worker for
        # this fire. _gpu_lock already serializes the heavy pipeline, so
        # holding it here guarantees the cache dir is stable while we
        # copy it into the canonical output dir.
        with _gpu_lock:
            _accept_fire_sync(fire_numbe)
            # Clear any stale serial gallery left over from a prior
            # cancelled sweep. _accept_fire_sync intentionally leaves
            # gallery cleanup to the caller (the mapping worker needs
            # the list for its own file cleanup in the MAPPING case),
            # but in this handler no worker is running (status was
            # MAPPED), so it is safe to strip the gallery here. On-disk
            # serial_* files will be reaped by the cache sweep below.
            if fire.serial_results or fire.serial_settings:
                with state.lock:
                    fire.serial_results = []
                    fire.serial_settings = []
                _save_fire_state()
        _push_notification(
            self._session_hash(), 'success',
            f'Accepted — {fire_numbe}',
            f'Canonical output written. Agreement: {fire.agreement_pct}%.',
            fire=fire_numbe)
        # Trigger an opportunistic cache sweep — the accept copied the
        # cache into canonical output, so .web_cache for this fire is
        # free to reclaim if it falls below the age threshold.
        threading.Thread(target=_cache_sweep, daemon=True).start()
        self._send_json({'status': 'accepted'})

    # -- Serial mapping & parameter ranking API --

    def handle_api_recommended_get(self, fire_numbe):
        """Return this fire's effective recommended settings + K config."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        override = getattr(fire, 'recommended_override', None)
        self._send_json({
            'has_override': bool(override),
            'settings': _get_recommended_settings(fire),
            'global_settings': [
                {'label': str(s.get('label', '')),
                 'params': dict(s.get('params', {}))}
                for s in state.recommended_settings
            ],
            'k_runs_per_setting': int(state.k_runs_per_setting),
            'k_jitter': int(state.k_jitter),
        })

    def handle_api_recommended_post(self, fire_numbe):
        """Save or clear this fire's per-fire recommended override."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        body = self._read_body()
        if body is None:
            return
        settings = body.get('settings', None)
        if settings is None or (isinstance(settings, list)
                                 and len(settings) == 0):
            # Clear override → falls back to global yaml
            fire.recommended_override = None
            _save_fire_state()
            self._send_json({'status': 'cleared'})
            return
        if not isinstance(settings, list):
            self._send_json(
                {'error': 'settings must be a list'}, 400)
            return
        clean = []
        for row in settings:
            if not isinstance(row, dict):
                self._send_json(
                    {'error': 'each setting must be an object'}, 400)
                return
            label = str(row.get('label', '') or '').strip()
            p = dict(row.get('params', {}) or {})
            eb = p.get('embed_bands')
            if eb is not None and isinstance(eb, (int, float)):
                p['embed_bands'] = ''
            if not label:
                label = f'setting_{len(clean) + 1}'
            clean.append({'label': label, 'params': p})
        fire.recommended_override = clean
        _save_fire_state()
        self._send_json({'status': 'saved', 'count': len(clean)})

    def handle_api_presets_get(self):
        """Return the preset bundles loaded from recommended_settings.yaml."""
        with state.lock:
            presets = {k: {
                'label': v.get('label', k),
                'description': v.get('description', ''),
                'params': dict(v.get('params', {})),
            } for k, v in state.presets.items()}
        self._send_json({'presets': presets})

    def handle_api_preset_post(self, fire_numbe):
        """Persist which preset the user last applied to a fire's form."""
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        body = self._read_body()
        if body is None:
            return
        name = str(body.get('preset', '') or '').strip()
        if name and name not in state.presets:
            self._send_json(
                {'error': f'Unknown preset: {name!r}'}, 400)
            return
        fire = state.fires[fire_numbe]
        with state.lock:
            fire.last_preset = name
        _save_fire_state()
        self._send_json({'ok': True, 'preset': name})
