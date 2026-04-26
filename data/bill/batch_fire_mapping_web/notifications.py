"""Toast notifications — per-session queues + broadcast channel.

Persists to ``notifications.yaml`` under ``state.shared_root``. Stateful
helpers read shared state through the module-level ``state`` attribute,
set by :func:`init` at server boot.
"""

import os
import sys
import time

from .state import AppState
from .io_utils import _atomic_yaml_dump

state: AppState = None


_NOTIFICATIONS_MAX_PER_SESSION = 50
_NOTIFICATIONS_MAX_BROADCAST = 20
_NOTIFICATION_TTL_SECONDS = 7 * 24 * 3600  # 7 days

_NOTIFICATION_KINDS = {'info', 'success', 'warning', 'error'}


def init(app_state: AppState):
    global state
    state = app_state


def _save_notifications():
    """Persist notifications to notifications.yaml (best-effort)."""
    if not state.shared_root:
        return
    try:
        path = os.path.join(state.shared_root, 'notifications.yaml')
        with state.lock:
            snap = {
                'counter': int(state.notification_counter),
                'broadcast_counter': int(state.broadcast_counter),
                'queues': {k: [dict(n) for n in v]
                           for k, v in state.notifications.items()},
                'cursor': dict(state.broadcast_cursor),
            }
        _atomic_yaml_dump(path, snap, mode=0o600)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: notifications: {exc}\n')


def _load_notifications():
    if not state.shared_root:
        return
    path = os.path.join(state.shared_root, 'notifications.yaml')
    if not os.path.isfile(path):
        return
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        with state.lock:
            state.notification_counter = int(
                data.get('counter', 0) or 0)
            state.broadcast_counter = int(
                data.get('broadcast_counter', 0) or 0)
            queues = data.get('queues', {}) or {}
            if isinstance(queues, dict):
                for k, v in queues.items():
                    if isinstance(v, list):
                        state.notifications[str(k)] = [
                            dict(e) for e in v if isinstance(e, dict)]
            cursor = data.get('cursor', {}) or {}
            if isinstance(cursor, dict):
                for k, v in cursor.items():
                    try:
                        state.broadcast_cursor[str(k)] = int(v)
                    except (TypeError, ValueError):
                        pass
    except Exception as exc:
        sys.stderr.write(
            f'[load] WARNING: notifications: {exc}\n')


def _prune_notifications_unlocked():
    """Drop expired / over-limit notifications. Call under state.lock.

    Cursors for sessions are left untouched here — a session key may
    not appear in ``state.sessions`` yet (e.g. notification pushed
    before the first poll from that session, or insecure_no_auth
    mode). Cursor cleanup happens in ``_sweep_expired_sessions`` via
    the session-expiry path.
    """
    cutoff = time.time() - _NOTIFICATION_TTL_SECONDS
    for key, queue in list(state.notifications.items()):
        pruned = [n for n in queue
                  if n.get('ts', 0) >= cutoff]
        limit = (_NOTIFICATIONS_MAX_BROADCAST if key == '__broadcast__'
                 else _NOTIFICATIONS_MAX_PER_SESSION)
        if len(pruned) > limit:
            pruned = pruned[-limit:]
        if pruned:
            state.notifications[key] = pruned
        else:
            del state.notifications[key]


def _push_notification(session_hash: str | None,
                       kind: str,
                       title: str,
                       body: str = '',
                       fire: str | None = None,
                       action: dict | None = None):
    """Enqueue a notification.

    *session_hash=None* → broadcast to every logged-in session.
    """
    if kind not in _NOTIFICATION_KINDS:
        kind = 'info'
    now = time.time()
    with state.lock:
        state.notification_counter += 1
        nid = state.notification_counter
        entry = {
            'id': int(nid),
            'ts': float(now),
            'kind': kind,
            'title': str(title)[:200],
            'body': str(body)[:2000],
            'fire': (str(fire)[:128] if fire else None),
            'action': (dict(action) if isinstance(action, dict)
                       else None),
        }
        if session_hash is None:
            state.broadcast_counter += 1
            entry['broadcast_id'] = int(state.broadcast_counter)
            state.notifications.setdefault(
                '__broadcast__', []).append(entry)
        else:
            state.notifications.setdefault(
                session_hash, []).append(entry)
        _prune_notifications_unlocked()
    _save_notifications()
    return entry


def _pop_notifications(session_hash: str) -> list:
    """Return + dequeue this session's pending notifications.

    Combines personal queue with any broadcast entries newer than the
    session's broadcast_cursor. Broadcast entries remain in the
    broadcast bucket for other sessions.
    """
    out: list = []
    with state.lock:
        # Personal — returned and cleared.
        personal = state.notifications.pop(session_hash, [])
        out.extend(personal)
        # Broadcast — returned once per session.
        last_seen = int(state.broadcast_cursor.get(session_hash, 0))
        max_id = last_seen
        for entry in state.notifications.get('__broadcast__', []):
            bid = int(entry.get('broadcast_id', 0))
            if bid > last_seen:
                out.append(entry)
                if bid > max_id:
                    max_id = bid
        state.broadcast_cursor[session_hash] = max_id
        _prune_notifications_unlocked()
    if out:
        _save_notifications()
    # Stable order by id
    out.sort(key=lambda e: e.get('id', 0))
    return out
