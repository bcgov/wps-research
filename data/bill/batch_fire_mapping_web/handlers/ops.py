"""Cross-cutting ops routes (years, queue, notifications, cache, report).

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


class OpsRoutes:
    """Cross-cutting ops routes (years, queue, notifications, cache, report)."""


    def handle_api_years(self):
        self._send_json({
            'years': sorted(state.rasters_by_year),
            'active': state.active_year,
        })

    def handle_api_year_switch(self):
        # Admin-only: switching is disruptive (affects every logged-in user).
        if (getattr(self, '_role', '') != 'admin'
                and state.admin_password):
            self._send_json({'error': 'Admin only'}, 403)
            return
        body = self._read_body()
        if body is None:
            return
        try:
            year = int(body.get('year'))
        except (TypeError, ValueError):
            self._send_json({'error': 'year must be an int'}, 400)
            return
        ok, msg = _switch_year(year)
        if not ok:
            self._send_json({'error': msg}, 409)
            return
        self._send_json({'ok': True, 'year': year})

    def handle_api_report(self):
        """Generate PDF report of selected accepted fires."""
        from batch_fire_mapping.generate_report import generate_report
        from .preview import (
            parse_envi_band_names, detect_band_groups, generate_preview_png)
        import tempfile

        # Get selected fire numbers from query params
        parsed = urlparse(self.path)
        from urllib.parse import parse_qs
        qs = parse_qs(parsed.query)
        fire_list = qs.get('fire', [])
        # mode=full (default) → ML hero + detail pages + brush comparison.
        # mode=brief → ML hero pages only (no params table, no brush).
        mode_raw = (qs.get('mode') or ['full'])[0].lower()
        report_mode = 'brief' if mode_raw == 'brief' else 'full'

        if not fire_list:
            self._send_json(
                {'error': 'No fires specified'}, 400)
            return

        # Stage each selected fire into a real tmp subdir (not a symlink
        # to the canonical fire dir). Symlink the individual files we
        # need in, then generate an EPHEMERAL <fire>_ml_overlay.png in
        # that subdir for the PDF's Pass-1 hero pages. Using a real
        # subdir means the ephemeral file never lands in the canonical
        # fire dir, which it would if the parent were a symlink.
        # Validate each fire number to prevent path traversal.
        root = os.path.realpath(state.output_root)
        tmp_dir = tempfile.mkdtemp(prefix='fire_report_sel_')
        try:
            for fn in fire_list:
                # Must be a known fire with valid characters
                if fn not in state.fires:
                    continue
                if not self._VALID_FN.fullmatch(fn):
                    continue
                src = os.path.realpath(
                    os.path.join(state.output_root, fn))
                # Must resolve inside output_root
                if src != root and not src.startswith(root + os.sep):
                    continue
                if not os.path.isdir(src):
                    continue

                tmp_fire_dir = os.path.join(tmp_dir, fn)
                os.makedirs(tmp_fire_dir, exist_ok=True)
                for entry in os.listdir(src):
                    fsrc = os.path.join(src, entry)
                    if not os.path.isfile(fsrc):
                        continue
                    try:
                        os.symlink(fsrc, os.path.join(tmp_fire_dir, entry))
                    except FileExistsError:
                        pass

                # Render the ephemeral ML-classification hero PNG from
                # the canonical crop + classified bin. This is only
                # used by generate_report.py's Pass-1 pages; the saved
                # <fire>_comparison.png (perimeter style) continues to
                # drive Pass-2 detail pages.
                try:
                    crop_bin = os.path.join(src, f'{fn}_crop.bin')
                    clf_bin = os.path.join(
                        src, f'{fn}_crop.bin_classified.bin')
                    if (os.path.isfile(crop_bin)
                            and os.path.isfile(clf_bin)):
                        post_tmp = os.path.join(
                            tmp_fire_dir, '_post_tmp.png')
                        band_names = parse_envi_band_names(crop_bin)
                        groups = detect_band_groups(band_names)
                        post_idx = (groups.get('post')
                                    or groups.get('pre'))
                        if post_idx and generate_preview_png(
                                crop_bin, post_idx, post_tmp):
                            f_obj = state.fires.get(fn)
                            subtitle_parts = []
                            hint_for_overlay = ''
                            hint_label = 'Hint'
                            if f_obj:
                                if f_obj.ml_area_ha >= 0:
                                    subtitle_parts.append(
                                        f'ML area: '
                                        f'{f_obj.ml_area_ha:.1f} ha')
                                if f_obj.agreement_pct >= 0:
                                    subtitle_parts.append(
                                        f'Agreement: '
                                        f'{f_obj.agreement_pct:.1f}%')
                                pt = (f_obj.perimeter_type or '').lower()
                                if pt == 'viirs':
                                    hint_label = 'VIIRS hint'
                                elif pt in ('polygon_perimeter',
                                             'polygon', 'traditional'):
                                    hint_label = ('Traditional perimeter '
                                                  '(hint)')
                            # Look up the hint raster inside the CANONICAL
                            # accepted dir (copied there by _accept_fire_sync
                            # via its *.bin glob). fire.hint_bin on the
                            # FireInfo points to the cache path, which may
                            # be missing for fires accepted in a prior
                            # session — that was why some PDFs silently
                            # dropped the lime overlay. Fall back to the
                            # cache path only if the canonical files are
                            # also missing.
                            viirs_matches = sorted(glob.glob(
                                os.path.join(src, 'VIIRS_*.bin')))
                            perim_in_canon = os.path.join(
                                src, f'{fn}_perimeter.bin')
                            if viirs_matches:
                                hint_for_overlay = viirs_matches[-1]
                                if hint_label == 'Hint':
                                    hint_label = 'VIIRS hint'
                            elif os.path.isfile(perim_in_canon):
                                hint_for_overlay = perim_in_canon
                                if hint_label == 'Hint':
                                    hint_label = ('Traditional perimeter '
                                                  '(hint)')
                            elif f_obj:
                                # Last resort: whatever the FireInfo has,
                                # even if it's a stale cache path — the
                                # renderer will silently skip if missing.
                                hint_for_overlay = (
                                    f_obj.hint_bin
                                    or f_obj.viirs_bin
                                    or f_obj.perim_bin
                                    or '')
                            ml_png = os.path.join(
                                tmp_fire_dir, f'{fn}_ml_overlay.png')
                            _render_ml_classification_png(
                                fn, post_tmp, clf_bin, ml_png,
                                '  |  '.join(subtitle_parts),
                                hint_path=hint_for_overlay,
                                hint_label=hint_label,
                                crop_bin=crop_bin)
                        if os.path.isfile(post_tmp):
                            try:
                                os.remove(post_tmp)
                            except OSError:
                                pass
                except Exception as exc:
                    sys.stderr.write(
                        f'[report] ML overlay render failed for '
                        f'{fn}: {exc}\n')
                    sys.stderr.flush()

            tmp_pdf = os.path.join(tmp_dir, 'report.pdf')
            pdf = generate_report(tmp_dir, tmp_pdf, mode=report_mode)
            if pdf is None or not os.path.isfile(pdf):
                self._send_json(
                    {'error': 'Report generation failed'}, 500)
                return

            with open(pdf, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Length', str(len(data)))
            fname = ('fire_report_ml_only.pdf'
                     if report_mode == 'brief'
                     else 'fire_report.pdf')
            self.send_header('Content-Disposition',
                             f'attachment; filename="{fname}"')
            self.end_headers()
            self.wfile.write(data)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def handle_api_queue(self):
        """Unified view of running + queued jobs across all users.

        Public (any authenticated user): lets analysts see what's in
        flight before clicking Map. Admin sees the same thing via
        /api/admin/queue plus IP details.
        """
        with state.lock:
            current_raw = (dict(state.current_job)
                           if state.current_job else None)
            waiting_raw = [dict(w) for w in state.waiting_jobs]
            batch = (dict(state.batch_status)
                     if state.batch_status else None)

        # Enrich current with ETA from the progress tracker (if the
        # running fire has a live progress snapshot).
        enriched_current = None
        if current_raw:
            cur_fire = current_raw.get('fire_numbe', '')
            # Strip " (run i/N)" suffix that _serial_map_worker adds.
            base_fn = cur_fire.split(' (run')[0].strip()
            f_obj = state.fires.get(base_fn)
            prog = _progress_snapshot(f_obj) if f_obj else {}
            enriched_current = {
                'fire_numbe': base_fn or cur_fire,
                'display': cur_fire,
                'started_at': current_raw.get('started_at', ''),
                'progress': prog,
            }

        # Active rebrushes (a different GPU-less pipeline).
        with _rebrush_procs_lock:
            rebrushes = list(_rebrush_procs.keys())

        self._send_json({
            'current': enriched_current,
            'waiting': [
                {'fire_numbe': w.get('fire_numbe', ''),
                 'queued_at': w.get('queued_at', '')}
                for w in waiting_raw
            ],
            'rebrushes': rebrushes,
            'batch': batch,
            'active_year': int(getattr(state, 'active_year', 0)),
        })

    def handle_api_notifications_get(self):
        """Return + dequeue pending notifications for this session.

        Personal entries are removed from the queue on this poll;
        broadcast entries advance the session's cursor but remain for
        other sessions.
        """
        sess = self._session_hash()
        if not sess:
            # No session → no notifications. Common when the user is
            # browsing with --insecure_no_auth on.
            self._send_json({'notifications': []})
            return
        items = _pop_notifications(sess)
        self._send_json({'notifications': items})

    def handle_api_notifications_ack(self):
        """Explicit acknowledgement — no-op (notifications are popped
        on GET), but accepted for UI convenience when the user clicks
        X on a toast before polling happens again.
        """
        body = self._read_body()
        if body is None:
            return
        # Nothing to do server-side. Return OK so the frontend doesn't
        # error on non-2xx.
        self._send_json({'ok': True})

    def handle_api_cache_status(self):
        """Summary of .web_cache disk usage and retention config."""
        scan = _cache_scan()
        with state.lock:
            cfg = dict(state.cache_retention)
            last = float(state.cache_last_sweep)
        self._send_json({
            'total_bytes': scan['total_bytes'],
            'pinned_bytes': scan['pinned_bytes'],
            'by_year': scan['by_year'],
            'n_fires': len(scan['entries']),
            'config': cfg,
            'last_sweep_ts': last,
            # Only send a summary per fire — 100s of fires × full paths
            # would bloat the payload.
            'entries': [
                {'fire_numbe': e['fire_numbe'],
                 'year': e['year'],
                 'bytes': e['bytes'],
                 'mtime_s': e['mtime_s'],
                 'pinned': e['pinned'],
                 'pin_reason': e['pin_reason']}
                for e in sorted(
                    scan['entries'],
                    key=lambda e: e['bytes'], reverse=True)[:200]
            ],
        })

    def handle_api_cache_sweep(self):
        """Admin: trigger a synchronous cache sweep.

        Body: {dry_run: bool, max_gb?: float, max_age_days?: int,
               enabled?: bool}. Config keys (if any) are persisted.
        """
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        body = self._read_body()
        if body is None:
            return
        dry_run = bool(body.get('dry_run', False))

        # Optional config update in the same call.
        cfg_patch = {}
        for k in ('max_gb',):
            if k in body:
                try:
                    cfg_patch[k] = float(body[k])
                except (TypeError, ValueError):
                    self._send_json(
                        {'error': f'{k} must be a number'}, 400)
                    return
        for k in ('max_age_days', 'sweep_interval_hours'):
            if k in body:
                try:
                    cfg_patch[k] = max(1, int(body[k]))
                except (TypeError, ValueError):
                    self._send_json(
                        {'error': f'{k} must be an int'}, 400)
                    return
        if 'enabled' in body:
            cfg_patch['enabled'] = bool(body['enabled'])
        if cfg_patch:
            with state.lock:
                state.cache_retention.update(cfg_patch)
            _save_cache_retention()

        result = _cache_sweep(dry_run=dry_run)
        self._send_json(result)
