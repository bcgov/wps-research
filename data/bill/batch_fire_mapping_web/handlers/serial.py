"""Serial-mapping (parameter-search) routes.

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


class SerialRoutes:
    """Serial-mapping (parameter-search) routes."""


    def handle_api_serial_map(self, fire_numbe):
        """Run N settings × K HDBSCAN replicates.

        Body: {
          mode: 'primary' | 'all' | omitted,
          settings: [{label, params}, ...] | null,
        }
        If `settings` is provided, use it. Otherwise pick from
        recommended (override-aware). `mode='primary'` restricts to the
        first setting (for one-click Map Fire). `mode='all'` (default
        when settings not provided) uses the full list.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if fire.status == FireStatus.MAPPING:
            self._send_json({'error': 'Already mapping'}, 400)
            return

        body = self._read_body()
        if body is None:
            return

        mode = str(body.get('mode', '') or '').lower()
        user_settings = body.get('settings', None)

        if isinstance(user_settings, list) and user_settings:
            settings = []
            for row in user_settings:
                if not isinstance(row, dict):
                    continue
                settings.append({
                    'label': str(row.get('label', '') or 'custom'),
                    'params': dict(row.get('params', {}) or {}),
                })
        else:
            settings = _get_recommended_settings(fire)

        if not settings:
            self._send_json(
                {'error': 'No recommended settings configured'}, 400)
            return

        if mode == 'primary':
            settings = settings[:1]

        k_runs = max(1, min(10, int(state.k_runs_per_setting)))
        k_jitter = max(0, int(state.k_jitter))

        # Set status BEFORE starting thread to avoid race. Save the
        # pre-sweep status so the worker can restore it on cancel.
        fire.serial_prev_status = fire.status
        if fire.status == FireStatus.ACCEPTED:
            fire.previously_accepted = True
            if fire.agreement_pct >= 0:
                fire.previously_accepted_agreement_pct = fire.agreement_pct
        fire.status = FireStatus.MAPPING
        fire.serial_results = []
        fire.serial_settings = [_clone_setting(s) for s in settings]
        fire.console_log.clear()
        fire.progress = {}

        # Capture session so notifications can target the initiating user.
        sess_hash = self._session_hash()

        thread = threading.Thread(
            target=_serial_map_worker,
            args=(fire_numbe, settings, k_runs, k_jitter, sess_hash),
            daemon=True)
        thread.start()

        self._send_json({
            'status': 'started',
            'n_settings': len(settings),
            'k_runs': k_runs,
            'total': len(settings) * k_runs,
        })

    def handle_api_serial_results(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        with state.lock:
            raw = list(fire.serial_results)
            settings_used = [
                {'label': str(s.get('label', '')),
                 'params': dict(s.get('params', {}))}
                for s in fire.serial_settings
            ]
        results = []
        for r in raw:
            results.append({
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
        self._send_json({
            'status': fire.status.value,
            'results': results,
            'settings_used': settings_used,
            'k_runs_per_setting': int(state.k_runs_per_setting),
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
        # Refuse when cache_dir is gone (e.g. pruned by _cache_sweep
        # after serial_results was persisted). Without this, the
        # accept path falls through to _accept_fire_sync which would
        # raise RuntimeError and surface as a 500; a 400 with a clear
        # message is easier for the analyst to act on.
        if not fire.cache_dir or not os.path.isdir(fire.cache_dir):
            self._send_json(
                {'error': 'Fire cache directory is missing — '
                          're-prepare the fire and re-run mapping.'},
                400)
            return
        # Only fires that have or had an active sweep are eligible
        # (MAPPING during sweep, MAPPED after completion, ACCEPTED if
        # re-picking a different run). Reject obviously-wrong states
        # rather than partially running the copy + write.
        if fire.status not in (
                FireStatus.MAPPING, FireStatus.MAPPED,
                FireStatus.ACCEPTED):
            self._send_json(
                {'error': f'Cannot accept: status is '
                          f'{fire.status.value}'},
                400)
            return

        # Fast path to release _gpu_lock: if the worker is currently
        # running a mapping subprocess, set the cancel flags and SIGTERM
        # its CLI BEFORE we wait for the lock. Without this step, the
        # lock wait below takes the full duration of the in-flight
        # replicate (minutes), during which the user sees "mapping
        # still running" even though they just clicked Accept.
        #
        # Lock-race safety (why this is NOT the old pre-lock setup):
        # the worker's cancel cleanup is gated on fire.serial_accept_event.
        # We clear the event before setting flags, then set it once
        # _accept_fire_sync has copied files into the canonical dir.
        # So even though serial_canceled is set before _gpu_lock here,
        # the worker's cleanup block (which deletes serial_* files)
        # will wait for us to finish regardless of which thread wins
        # the lock race — fixing the original race where flag-before-
        # lock let the worker delete files before accept copied them.
        worker_running = (fire.status == FireStatus.MAPPING)
        accept_event = None
        if worker_running:
            accept_event = threading.Event()
            accept_event.clear()
            with state.lock:
                fire.serial_accept_event = accept_event
                # Pin serial_prev_status to ACCEPTED so the worker's
                # generic revert lands there (it reverts to
                # prev_status regardless of cancel source).
                fire.serial_prev_status = FireStatus.ACCEPTED
                fire.serial_accept_promoted = True
                fire.serial_canceled = True
            # SIGTERM the running CLI. The worker's _stream_subprocess
            # returns promptly, the worker exits its with _gpu_lock
            # block, and _gpu_lock is released so we can acquire it
            # below without waiting minutes for the replicate to
            # finish on its own.
            _terminate_serial_proc(fire_numbe)

        # Serialize the copy + accept under _gpu_lock. The worker takes
        # the same lock per replicate AND for its cancel cleanup, so
        # this blocks until the worker is not actively writing to
        # cache_dir. With the SIGTERM above, the wait is bounded by
        # subprocess teardown (seconds), not full replicate duration.
        try:
            _gpu_lock.acquire()
        except Exception:
            if accept_event is not None:
                accept_event.set()
            raise
        try:
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
                # Propagate the pre-brush raw mask into the main slot
                # too. Without this, a post-accept rebrush would read
                # whichever raw backup the LAST replicate happened to
                # leave in cache_dir — not the one that pairs with the
                # accepted brushed mask. The canonical dir intentionally
                # skips *_raw.bin (it's a cache-only artifact), but the
                # main-slot raw has to match the main-slot classified so
                # that rebrush + brush-comparison figure regen stay
                # coherent with the accepted run.
                ser_raw = (
                    os.path.splitext(serial_clf)[0] + '_raw.bin')
                if os.path.isfile(ser_raw):
                    main_raw = (
                        os.path.splitext(main_clf)[0] + '_raw.bin')
                    shutil.copy2(ser_raw, main_raw)
                    ser_raw_hdr = (
                        os.path.splitext(ser_raw)[0] + '.hdr')
                    if not os.path.isfile(ser_raw_hdr):
                        ser_raw_hdr = ser_raw + '.hdr'
                    if os.path.isfile(ser_raw_hdr):
                        shutil.copy2(
                            ser_raw_hdr,
                            os.path.splitext(main_raw)[0] + '.hdr')
                # Regenerate the "ML classification" overlay from the
                # accepted run's (possibly rebrushed) mask. The worker
                # overwrites previews/result.png on every completed run,
                # so without this it points at whichever run finished
                # last.
                try:
                    _overlay_mask_on_post(
                        fire, main_clf, 'result', (0.9, 0.1, 0.0))
                except Exception:
                    pass
            if serial_comp and os.path.isfile(serial_comp):
                main_comp = os.path.join(
                    fire.cache_dir, f'{fire_numbe}_comparison.png')
                shutil.copy2(serial_comp, main_comp)
                fire.last_comparison = main_comp

            # Propagate the accepted run's brush comparison PNG into
            # the main slot so _accept_fire_sync's *.png glob copies
            # it into the canonical dir. Without this, accepting a
            # non-last serial run left the canonical output with
            # whichever brush PNG the LAST replicate produced — or
            # nothing at all, if the original map run predated
            # class_brush.exe being available.
            serial_brush = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_{run_id}_brush.png')
            if os.path.isfile(serial_brush):
                main_brush = os.path.join(
                    fire.cache_dir,
                    f'{fire_numbe}_brush_comparison.png')
                shutil.copy2(serial_brush, main_brush)

            # Hold state.lock across the four mutations so readers on
            # other threads (gallery, console, _save_fire_state) never
            # observe a torn fire (e.g. status=MAPPED with stale
            # agreement_pct / ml_area_ha / last_params).
            with state.lock:
                fire.agreement_pct = result.get('agreement_pct', -1)
                fire.ml_area_ha = result.get('ml_area_ha', -1.0)
                fire.last_params = result.get('params', {})
                fire.status = FireStatus.MAPPED

            # Now accept via the normal flow
            _accept_fire_sync(fire_numbe)

            # If the worker is still running, leave serial_results /
            # serial file cleanup to it — racing to clear the list here
            # would fight with the worker's in-flight append for the
            # next replicate, and racing to delete serial files would
            # fight its in-flight writes.
            if not worker_running:
                for sr in fire.serial_results:
                    rid = sr.get('run_id')
                    if rid is None:
                        continue
                    for pat in (
                        f'{fire_numbe}_serial_{rid}_classified.bin',
                        f'{fire_numbe}_serial_{rid}_classified.hdr',
                        f'{fire_numbe}_serial_{rid}_classified_raw.bin',
                        f'{fire_numbe}_serial_{rid}_classified_raw.hdr',
                        f'{fire_numbe}_serial_{rid}.png',
                        f'{fire_numbe}_serial_{rid}_brush.png',
                    ):
                        p = os.path.join(fire.cache_dir, pat)
                        if os.path.isfile(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                    overlay = os.path.join(
                        fire.cache_dir, 'previews', f'serial_{rid}.png')
                    if os.path.isfile(overlay):
                        try:
                            os.remove(overlay)
                        except OSError:
                            pass
                # Free per-setting T-SNE+RF caches (single heaviest
                # artifact in .web_cache — 30-100 MB each, typically 3
                # per sweep). They exist to let replicates skip the
                # full pipeline within a sweep; once accepted, they
                # are dead weight.
                for npz in glob.glob(os.path.join(
                        fire.cache_dir,
                        f'{fire_numbe}_serial_state_s*.npz')):
                    try:
                        os.remove(npz)
                    except OSError:
                        pass
                with state.lock:
                    fire.serial_results = []
                    fire.serial_settings = []
                # Re-persist — _accept_fire_sync already wrote
                # fire_state.yaml once, but at that point
                # serial_results was still populated. Writing again
                # now ensures on-disk state matches the final
                # in-memory state (empty gallery for an ACCEPTED fire)
                # so a restart doesn't resurrect a stale gallery.
                _save_fire_state()
        finally:
            _gpu_lock.release()
            # Signal the worker's cleanup to proceed — now that the
            # canonical dir is written and status is ACCEPTED, it is
            # safe for the worker to delete the per-run serial_*
            # files. The event is always set, even on exception, so
            # the worker never hangs waiting for us.
            if accept_event is not None:
                accept_event.set()

        self._send_json({'status': 'accepted'})

    def handle_api_serial_cancel(self, fire_numbe):
        """Cancel a running N×K serial mapping without accepting any run.

        Sets the cancel flag; the worker sees it between replicates,
        bails out, cleans up its serial cache files, and restores the
        pre-sweep status (saved in ``fire.serial_prev_status`` when the
        sweep started). If no sweep is running, 400.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if fire.status != FireStatus.MAPPING:
            self._send_json(
                {'error': f'Not mapping (status={fire.status.value})'},
                400)
            return
        fire.serial_canceled = True
        # SIGTERM the running CLI so the worker's _gpu_lock releases
        # promptly and the cancel takes effect within seconds rather
        # than waiting up to the full duration of the current
        # replicate. Safe no-op if the worker happens to be between
        # replicates (no proc registered for this fire).
        _terminate_serial_proc(fire_numbe)
        self._send_json({'status': 'cancelling'})
