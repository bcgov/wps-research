"""Session/auth/IP helpers — token hashing, login rate limiting, session sweep.

Stateful helpers read shared state through the module-level ``state``
attribute, set by :func:`init` at server boot before any handler runs.
"""

import datetime
import hashlib
import ipaddress

from .state import AppState

state: AppState = None


# Login rate limiting
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 300  # 5 minutes

_SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days in seconds


def init(app_state: AppState):
    """Bind the shared AppState. Must run before any other call here."""
    global state
    state = app_state


def _hash_token(token: str) -> str:
    """SHA-256 hash a session token for storage (never persist raw tokens)."""
    return hashlib.sha256(token.encode()).hexdigest()


def _normalize_ip(ip_str: str) -> str:
    """Normalize IP address. Maps IPv6-mapped IPv4 (::ffff:x.x.x.x) to IPv4."""
    try:
        addr = ipaddress.ip_address(ip_str)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(addr)
    except ValueError:
        return ip_str


def _check_login_rate(ip: str) -> bool:
    """Return True if the IP is under the rate limit."""
    import time
    now = time.time()
    with state.lock:
        attempts = state.login_attempts.get(ip, [])
        attempts = [t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS]
        if attempts:
            state.login_attempts[ip] = attempts
        else:
            state.login_attempts.pop(ip, None)
        # Opportunistic global sweep: bound memory under IP-spray attacks.
        if len(state.login_attempts) > 1024:
            stale = [k for k, v in state.login_attempts.items()
                     if not v or now - v[-1] >= _LOGIN_WINDOW_SECONDS]
            for k in stale:
                state.login_attempts.pop(k, None)
        return len(attempts) < _LOGIN_MAX_ATTEMPTS


def _record_failed_login(ip: str):
    """Record a failed login attempt for rate limiting."""
    import time
    now = time.time()
    with state.lock:
        attempts = state.login_attempts.get(ip, [])
        attempts.append(now)
        state.login_attempts[ip] = attempts


def _sweep_expired_sessions():
    """Drop sessions past _SESSION_MAX_AGE. Call under state.lock.

    Also drops broadcast-cursors + personal notification queues for
    sessions that are being evicted, so notification state doesn't
    outlive its session.
    """
    now = datetime.datetime.now()
    stale = []
    for tok, sess in state.sessions.items():
        try:
            created = datetime.datetime.fromisoformat(sess['created_at'])
            if (now - created).total_seconds() > _SESSION_MAX_AGE:
                stale.append(tok)
        except (KeyError, ValueError, TypeError):
            stale.append(tok)
    for tok in stale:
        state.sessions.pop(tok, None)
        state.notifications.pop(tok, None)
        state.broadcast_cursor.pop(tok, None)
    return len(stale)
