"""Auth + admin route mixin: login/logout/admin pages and IP queue.

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


class AuthRoutes:
    """Auth + admin route mixin: login/logout/admin pages and IP queue."""


    def handle_login_page(self):
        # Already logged in? Redirect to home.
        token = self._get_cookie('session')
        if token and _hash_token(token) in state.sessions:
            self._redirect('/')
            return
        html = render_template('login.html', {'error_msg': ''})
        self._send_html(html)

    def handle_login_post(self):
        import hmac
        import secrets

        ip = self._client_ip()

        # Rate limit login attempts
        if not _check_login_rate(ip):
            html = render_template('login.html', {
                'error_msg': '<div class="error-msg" style="display:block">'
                             'Too many login attempts. '
                             'Please try again later.</div>',
            })
            self._send_html(html, 429)
            return

        # Parse form body (application/x-www-form-urlencoded)
        try:
            length = int(self.headers.get('Content-Length', 0))
        except (TypeError, ValueError):
            self.send_error(400, 'Malformed Content-Length')
            return
        if length < 0 or length > 10000:
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
            _record_failed_login(ip)
            html = render_template('login.html', {
                'error_msg': '<div class="error-msg" style="display:block">'
                             'Invalid password.</div>',
            })
            self._send_html(html, 401)
            return

        # Create session — store hashed token, cookie gets raw token
        raw_token = secrets.token_hex(32)
        hashed = _hash_token(raw_token)
        with state.lock:
            swept = _sweep_expired_sessions()
            state.sessions[hashed] = {
                'role': role,
                'username': username,
                'ip': self._client_ip(),
                'created_at': datetime.datetime.now().isoformat(
                    timespec='seconds'),
            }
        if swept:
            sys.stderr.write(
                f'[auth] swept {swept} expired session(s)\n')
        _save_sessions()

        # Set cookie and redirect to home. The Secure flag is only
        # valid over HTTPS — browsers silently drop Secure cookies on
        # plain-HTTP non-localhost connections (e.g. LAN IPs reached
        # over a VPN), which manifests as an endless bounce back to
        # /login despite a correct password. Detect HTTPS via proxy
        # header when --trust_proxy is set; otherwise omit Secure.
        secure_flag = ''
        xfp = self.headers.get('X-Forwarded-Proto', '').lower()
        if state.trust_proxy and xfp == 'https':
            secure_flag = 'Secure; '
        cookie = (f'session={raw_token}; HttpOnly; SameSite=Lax; '
                  f'{secure_flag}Path=/; Max-Age={_SESSION_MAX_AGE}')
        self.send_response(302)
        self.send_header('Location', '/')
        self.send_header('Set-Cookie', cookie)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def handle_logout(self):
        # Clear session
        raw_token = self._get_cookie('session')
        if raw_token:
            hashed = _hash_token(raw_token)
            with state.lock:
                if hashed in state.sessions:
                    del state.sessions[hashed]
                # Notification bookkeeping follows the session's lifetime.
                state.notifications.pop(hashed, None)
                state.broadcast_cursor.pop(hashed, None)
            _save_sessions()
            _save_notifications()
        # Clear cookie and redirect to login
        self.send_response(302)
        self.send_header('Location', '/login')
        self.send_header('Set-Cookie',
                         'session=; HttpOnly; SameSite=Lax; '
                         'Path=/; Max-Age=0')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def handle_admin_page(self):
        if getattr(self, '_role', '') != 'admin':
            self.send_error(403, 'Admin access required')
            return
        html = render_template('admin.html', {})
        self._send_html(html)

    # -- Access control & admin API --

    def handle_api_access_status(self):
        """Called by the pending page to check if IP was approved."""
        ip = self._client_ip()
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
        with state.lock:
            payload = {
                'approved': {k: dict(v)
                             for k, v in state.approved_ips.items()},
                'blocked': {k: dict(v)
                            for k, v in state.blocked_ips.items()},
                'pending': {k: dict(v)
                            for k, v in state.pending_ips.items()},
            }
        self._send_json(payload)

    def handle_api_admin_queue(self):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        with state.lock:
            current = (dict(state.current_job)
                       if state.current_job else None)
            waiting = [dict(w) for w in state.waiting_jobs]
        self._send_json({
            'current': current,
            'waiting': waiting,
        })

    def handle_api_admin_ip_action(self, action):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        body = self._read_body()
        if body is None:
            return
        ip = body.get('ip', '').strip()
        if not ip:
            self._send_json({'error': 'No IP provided'}, 400)
            return

        now = datetime.datetime.now().isoformat(timespec='seconds')

        with state.lock:
            if action == 'approve':
                # Preserve username from pending entry
                pending_info = state.pending_ips.get(ip, {})
                state.approved_ips[ip] = {
                    'username': pending_info.get('username', ''),
                    'role': 'user',
                    'approved_by': self._client_ip(),
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
                    'blocked_by': self._client_ip(),
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
