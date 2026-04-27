"""Base FireHandler mixin: route tables, auth gate, response helpers.

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


class BaseHandler:
    # -- Routing tables (compiled once) --
    ROUTES_GET = [
        (re.compile(r'^/$'), 'handle_fire_list'),
        (re.compile(r'^/login$'), 'handle_login_page'),
        (re.compile(r'^/logout$'), 'handle_logout'),
        (re.compile(r'^/admin$'), 'handle_admin_page'),
        (re.compile(r'^/new_fire$'), 'handle_new_fire_page'),
        (re.compile(r'^/api/year/(?P<y>\d+)/overview\.png$'),
         'handle_api_year_overview_png'),
        (re.compile(r'^/api/year/(?P<y>\d+)/overview_meta$'),
         'handle_api_year_overview_meta'),
        (re.compile(
            r'^/api/fire/preview_hint/(?P<preview_id>[A-Za-z0-9_-]+)/'
            r'(?P<view>[A-Za-z0-9_-]+)\.png$'),
         'handle_api_fire_preview_hint_png'),
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
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/console$'),
         'handle_api_console'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial_results$'),
         'handle_api_serial_results'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/recommended$'),
         'handle_api_recommended_get'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial/(?P<run_id>[0-9]+)/image$'),
         'handle_api_serial_image'),
        (re.compile(r'^/api/report$'), 'handle_api_report'),
        (re.compile(r'^/api/fires/hidden$'), 'handle_api_fires_hidden'),
        (re.compile(r'^/api/years$'), 'handle_api_years'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/progress$'),
         'handle_api_progress'),
        (re.compile(r'^/api/queue$'), 'handle_api_queue'),
        (re.compile(r'^/api/notifications$'),
         'handle_api_notifications_get'),
        (re.compile(r'^/api/cache/status$'), 'handle_api_cache_status'),
        (re.compile(r'^/static/(?P<path>.+)$'), 'handle_static'),
    ]
    ROUTES_POST = [
        (re.compile(r'^/login$'), 'handle_login_post'),
        (re.compile(r'^/api/fire/create$'), 'handle_api_fire_create'),
        (re.compile(r'^/api/fire/preview_hint$'),
         'handle_api_fire_preview_hint'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/cancel_create$'),
         'handle_api_fire_cancel_create'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/clear_new$'),
         'handle_api_fire_clear_new'),
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
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial_map$'),
         'handle_api_serial_map'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/recommended$'),
         'handle_api_recommended_post'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial/(?P<run_id>[0-9]+)/accept$'),
         'handle_api_serial_accept'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/serial/cancel$'),
         'handle_api_serial_cancel'),
        (re.compile(r'^/api/admin/ip/(?P<action>approve|block|revoke|unblock)$'),
         'handle_api_admin_ip_action'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/unhide$'),
         'handle_api_unhide'),
        (re.compile(r'^/api/batch/cancel$'), 'handle_api_batch_cancel'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/rebrush$'),
         'handle_api_rebrush'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/rebrush/cancel$'),
         'handle_api_rebrush_cancel'),
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/abort$'),
         'handle_api_fire_abort'),
        (re.compile(r'^/api/notifications/ack$'),
         'handle_api_notifications_ack'),
        (re.compile(r'^/api/cache/sweep$'), 'handle_api_cache_sweep'),
        (re.compile(r'^/api/year/switch$'), 'handle_api_year_switch'),
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

    def _client_ip(self) -> str:
        """Get the client IP, respecting --trust_proxy."""
        raw = self.client_address[0]
        if getattr(state, 'trust_proxy', False):
            xff = self.headers.get('X-Forwarded-For', '')
            if xff:
                raw = xff.split(',')[-1].strip()
        return _normalize_ip(raw)

    def _session_hash(self) -> str:
        """Return the SHA-256 hash of the session cookie, or '' if none.

        Used to key per-session notifications. Safe to call after
        _check_session has set or cleared ``self._session_hash_cached``.
        """
        return getattr(self, '_session_hash_cached', '') or ''

    def _check_session(self) -> str | None:
        """Check session cookie. Returns role or None."""
        self._username = ''
        self._role = ''
        self._session_hash_cached = ''

        # No passwords configured → everyone is admin
        if not state.admin_password and not state.user_password:
            self._role = 'admin'
            return 'admin'

        raw_token = self._get_cookie('session')
        if not raw_token:
            return None
        token = _hash_token(raw_token)

        with state.lock:
            if token not in state.sessions:
                return None
            session = state.sessions[token]

            # Check expiry
            try:
                created = datetime.datetime.fromisoformat(
                    session['created_at'])
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
        self._session_hash_cached = token
        return self._role

    def _check_ip(self, role: str) -> bool:
        """Check IP access. Admins auto-approve. Returns True if allowed."""
        ip = self._client_ip()
        username = getattr(self, '_username', '')

        save_needed = False
        with state.lock:
            # Blocked IPs are blocked even for admins
            if ip in state.blocked_ips:
                blocked = True
            else:
                blocked = False

            if not blocked:
                if role == 'admin':
                    if ip not in state.approved_ips:
                        state.approved_ips[ip] = {
                            'username': username,
                            'role': 'admin',
                            'approved_by': 'auto (admin)',
                            'timestamp': datetime.datetime.now().isoformat(
                                timespec='seconds'),
                        }
                        save_needed = True
                    else:
                        state.approved_ips[ip]['username'] = username
                        state.approved_ips[ip]['role'] = 'admin'
                    state.pending_ips.pop(ip, None)
                    approved_admin = True
                    approved_user = False
                elif ip in state.approved_ips:
                    state.approved_ips[ip]['username'] = username
                    state.approved_ips[ip]['role'] = 'user'
                    approved_admin = False
                    approved_user = True
                else:
                    # Unknown IP → pending
                    now = datetime.datetime.now().isoformat(
                        timespec='seconds')
                    if ip not in state.pending_ips:
                        state.pending_ips[ip] = {
                            'username': username,
                            'first_seen': now,
                            'last_seen': now,
                        }
                    else:
                        state.pending_ips[ip]['last_seen'] = now
                        state.pending_ips[ip]['username'] = username
                    approved_admin = False
                    approved_user = False

        if blocked:
            self._send_html(render_template('pending.html', {
                'ip': ip,
                'title': 'Access Denied',
                'message': 'Your IP address has been blocked by an '
                           'administrator.',
                'auto_refresh': 'false',
            }), 403)
            return False

        if save_needed:
            _save_ip_list()
        if approved_admin or approved_user:
            return True

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
                # Accept if origin matches startup-computed set OR
                # the Host header the client actually connected to
                # (same-origin: browser sets Origin = scheme://host)
                allowed = set(state.allowed_origins)
                host_hdr = self.headers.get('Host', '')
                if host_hdr:
                    allowed.add(f'http://{host_hdr}')
                if origin not in allowed:
                    self.send_error(403, 'Cross-origin request blocked')
                    return
            elif not x_req:
                # X-Requested-With triggers CORS preflight, so
                # cross-origin requests with this header are blocked
                # by the browser (server sends no CORS headers).
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

    def _read_body(self) -> dict | None:
        """Read and parse JSON body. Returns None (and sends 413/400) on error."""
        try:
            length = int(self.headers.get('Content-Length', 0))
        except (TypeError, ValueError):
            self.send_error(400, 'Malformed Content-Length')
            return None
        if length < 0:
            self.send_error(400, 'Malformed Content-Length')
            return None
        if length == 0:
            return {}
        if length > self._MAX_BODY:
            self.send_error(413, 'Request body too large')
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

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
