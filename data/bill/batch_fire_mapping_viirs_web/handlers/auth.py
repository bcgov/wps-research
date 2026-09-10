"""Auth + admin route mixin: login/logout/admin pages and IP queue.

This is one slice of FireHandler. Methods reference module-level
helpers from ``app`` via top-of-file imports; ``state`` is rebound
in :func:`init` so it tracks the live :class:`AppState` instance
created by ``app.init_app``.
"""

from urllib.parse import quote
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
        # Already logged in? Go where they were headed.
        token = self._get_cookie('session')
        nxt = self._login_next()
        sess = (state.sessions.get(_hash_token(token))
                if token else None) or {}
        if sess.get('role') == 'admin' and self._admin_fresh(sess):
            self._redirect(nxt or '/')
            return
        if sess:
            # A leftover non-admin session. Left in place it bounces
            # forever: this page would send them on, and the admin gate
            # would send them straight back. Drop it and show the form.
            try:
                with state.lock:
                    state.sessions.pop(_hash_token(token), None)
                _save_sessions()
            except Exception:
                pass
            self._stale_session = True
        html = render_template('login.html', {
            'error_msg': '',
            'next_qs': (f'?next={quote(nxt, safe="")}' if nxt else ''),
        })
        if getattr(self, '_stale_session', False):
            # Expire the cookie along with the form, so the browser
            # stops presenting a session the server has discarded.
            body = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Set-Cookie',
                             'session=; HttpOnly; SameSite=Lax; '
                             'Path=/; Max-Age=0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_html(html)

    def _next_qs(self) -> str:
        nxt = self._login_next()
        return f'?next={quote(nxt, safe="")}' if nxt else ''

    def _login_next(self) -> str:
        """The page to return to after logging in.

        Only same-site absolute paths are honoured: taking an arbitrary
        URL from a query string and redirecting to it after
        authenticating is an open-redirect, which is worth avoiding
        even on an internal tool.
        """
        from urllib.parse import urlparse, parse_qs, unquote
        try:
            q = parse_qs(urlparse(self.path).query)
            nxt = unquote((q.get('next') or [''])[0])
        except Exception:
            return ''
        if nxt.startswith('/') and not nxt.startswith('//'):
            return nxt
        return ''

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
                'next_qs': self._next_qs(),
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

        # Both halves must be right, and only the admin credential
        # grants anything.
        #
        # The username was read and discarded, so the password alone
        # was the credential. And a correct USER password still minted
        # a session -- which now confers nothing, yet was enough to
        # make the login page think the visitor was signed in. That is
        # what produced the redirect loop: the login page sent them to
        # /admin, the gate sent them back for not being an admin.
        want_user = (getattr(state, 'admin_username', '') or 'admin')
        role = None
        if (state.admin_password
                and hmac.compare_digest(password, state.admin_password)
                and hmac.compare_digest(username.lower(),
                                        want_user.lower())):
            role = 'admin'

        if role is None:
            _record_failed_login(ip)
            html = render_template('login.html', {
                'error_msg': '<div class="error-msg" style="display:block">'
                             'Incorrect username or password.</div>',
                'next_qs': self._next_qs(),
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
                # When the password was actually proved. The admin area
                # checks this, not merely that a session exists.
                'admin_verified_at': time.time(),
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
        self.send_header('Location', self._login_next() or '/')
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

    def handle_api_client_info(self):
        """Record what only the browser knows: screen size and DPR.

        Posted once per page load. Deliberately open to any visitor --
        there are no logins for ordinary use, and refusing this would
        simply leave the column empty for everyone who is not an admin.

        Values are clamped and type-checked: this is unauthenticated
        input, and it lands in a file an admin reads, so a hostile
        payload must not be able to write arbitrary content there.
        """
        body = self._read_body()
        if body is None:
            return
        ip = self._client_ip()

        def _num(v, lo, hi):
            try:
                n = int(float(v))
            except (TypeError, ValueError):
                return 0
            return n if lo <= n <= hi else 0

        w = _num(body.get('w'), 1, 100000)
        h = _num(body.get('h'), 1, 100000)
        try:
            dpr = round(float(body.get('dpr') or 0), 2)
        except (TypeError, ValueError):
            dpr = 0.0
        if not (0.1 <= dpr <= 10.0):
            dpr = 0.0

        changed = False
        with state.lock:
            entry = state.approved_ips.get(ip)
            if entry is not None:
                if w and h and entry.get('screen') != f'{w}x{h}':
                    entry['screen'] = f'{w}x{h}'
                    changed = True
                if dpr and entry.get('dpr') != dpr:
                    entry['dpr'] = dpr
                    changed = True
        if changed:
            _save_ip_list()
        self._send_json({'status': 'ok'})

    def handle_api_admin_known_clear(self):
        """Erase the record of addresses seen.

        The RECORD only. Revocations and blocks are separate lists and
        are left untouched: clearing a log of who has visited must not
        quietly readmit someone an admin deliberately shut out.
        """
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin access required'}, 403)
            return
        with state.lock:
            removed = len(state.approved_ips)
            state.approved_ips.clear()
            # Pending is vestigial, but clearing it here keeps the two
            # from drifting if an approval flow ever returns.
            state.pending_ips.clear()
        _save_ip_list()
        sys.stderr.write(
            f'[access] known-address list cleared by admin '
            f'({removed} entr{"y" if removed == 1 else "ies"})\n')
        self._send_json({'status': 'ok', 'removed': removed})

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

            elif action in ('revoke', 'restore', 'unrevoke'):
                # Revoke is now block; restore is now unblock. The
                # actions are kept as aliases so an open admin page or
                # a bookmarked call still does the expected thing
                # rather than silently failing.
                if action == 'revoke':
                    approved_info = state.approved_ips.get(ip, {})
                    state.blocked_ips[ip] = {
                        'username': approved_info.get('username', ''),
                        'blocked_by': self._client_ip(),
                        'timestamp': now,
                        'first_seen': approved_info.get('first_seen', ''),
                        'last_seen': approved_info.get('last_seen', ''),
                    }
                    state.approved_ips.pop(ip, None)
                    state.pending_ips.pop(ip, None)
                else:
                    state.blocked_ips.pop(ip, None)

            elif action == 'unblock':
                state.blocked_ips.pop(ip, None)

        _save_ip_list()
        self._send_json({'status': 'ok'})
