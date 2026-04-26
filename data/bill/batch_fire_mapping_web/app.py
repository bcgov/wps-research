"""Stdlib-only web server for interactive fire mapping.

Uses http.server.ThreadingHTTPServer — no FastAPI, no uvicorn, no Jinja2.
SSE (Server-Sent Events) via fetch() for real-time console streaming.
"""

import os
import signal
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from osgeo import gdal

from .state import AppState
from .preview import generate_all_previews
from .io_utils import _atomic_yaml_dump
from . import auth as _auth
from .auth import (
    _hash_token, _normalize_ip, _check_login_rate, _record_failed_login,
    _sweep_expired_sessions, _SESSION_MAX_AGE,
)
from . import notifications as _notifications
from .notifications import (
    _save_notifications, _load_notifications, _prune_notifications_unlocked,
    _push_notification, _pop_notifications,
)
from . import cache_retention as _cache_retention
from .cache_retention import (
    _save_cache_retention, _load_cache_retention, _dir_bytes_and_mtime,
    _cache_scan, _cache_sweep, _cache_sweep_loop, _cache_sweep_lock,
)
from . import progress as _progress
from .progress import (
    _STAGE_MARKERS, _STAGE_ORDER_FULL, _STAGE_ORDER_RESUME, _STAGE_LABELS,
    _STAGE_TIMINGS_MAX_SAMPLES, _STAGE_FALLBACK,
    _detect_stage, _save_stage_timings, _load_stage_timings,
    _record_stage_duration, _stage_median, _estimate_full_run_seconds,
    _ProgressTracker, _progress_snapshot, _ETA_FUDGE, _ETA_FLOOR_S,
)
from . import mapping as _mapping
from .mapping import (
    _compute_ml_area, _overlay_mask_on_post, _generate_result_preview,
    _compute_agreement,
)
from . import persistence as _persistence
from .persistence import (
    _save_sessions, _save_settings, _save_notes, _save_ip_list,
    _save_fire_state, _load_fire_state,
    _save_active_year, _switch_year,
)
from . import brush as _brush
from .brush import (
    _class_brush_exe, _read_envi_mask, _write_envi_mask_like,
    _run_class_brush_only, _align_mask_to_crop_frame,
    _render_comparison_png, _render_ml_classification_png,
    _render_brush_comparison_png,
)
from .templates import _html_escape, render_template
from .validation import _PARAM_SPEC, _validate_param, _validate_embed_bands
from . import mapping_cmd as _mapping_cmd
from .mapping_cmd import _build_mapping_cmd
from . import prepare as _prepare
from .prepare import (
    _prepare_fire_sync, _ensure_brush_comparison_in_cache, _accept_fire_sync,
)
from . import workers as _workers
from .workers import (
    _batch_map_worker, _serial_map_worker, _jitter_hdbscan,
    _serial_setup, _serial_snapshot_run0, _serial_run_replicate,
    _serial_handle_cancel, _serial_finalize,
)

gdal.UseExceptions()

_HERE = os.path.dirname(os.path.abspath(__file__))

# Global state — set by init_app() before the server starts
state: AppState = None

# GPU lock — serialises heavy operations (only one at a time)
_gpu_lock = threading.Lock()
# One-element container for the queue depth: handlers in sibling modules
# rebind the count, and Python ``global`` only works in the module that
# owns the binding. A list lets every module share the same object.
_gpu_queue = [0]          # _gpu_queue[0] = number of tasks waiting or running
_gpu_queue_lock = threading.Lock()   # protects the counter

# Batch cancellation flag
_batch_cancel = threading.Event()

# Serialises read-modify-write of shared on-disk records
# (accepted_params.csv, fire_status.yaml). Prevents concurrent accepts
# from corrupting these files or losing rows.
_accept_file_lock = threading.Lock()

# Set of fire_numbers whose accept is currently in progress. Read by
# _cache_scan so the background cache sweeper cannot rmtree a
# cache_dir mid-copy. _gpu_lock already blocks the mapping worker, but
# _cache_sweep uses its own separate lock, so without this guard an
# age-based eviction could race the glob-copy loop in
# _accept_fire_sync and leave the canonical output partially
# populated.
_accept_in_progress: set = set()
_accept_in_progress_lock = threading.Lock()

# Live mapping subprocesses, keyed by fire_numbe. Lets the accept /
# cancel handlers SIGTERM the running CLI so _gpu_lock releases
# promptly instead of waiting up to several minutes for the current
# replicate to finish naturally. Mirrors the rebrush-side
# _rebrush_procs dict.
_serial_procs: dict = {}
_serial_procs_lock = threading.Lock()

# Registry of running rebrush subprocesses, keyed by fire_numbe. The
# cancel endpoint uses this to terminate a running class_brush.exe.
# Hoisted to the top of the module (instead of living inside the brush
# helpers) so init_app can pass it to cache_retention/persistence
# before the brush module gets a reference back.
_rebrush_procs: dict = {}
_rebrush_procs_lock = threading.Lock()


def _terminate_serial_proc(fire_numbe: str) -> bool:
    """SIGTERM the mapping CLI subprocess for *fire_numbe* if one is
    running. Returns True iff a proc was found and signalled. Safe
    to call when no proc is registered (no-op).

    Signals the entire process group so helper grandchildren the CLI
    spawns via subprocess.run (gdal_translate, qgis, …) terminate
    too — same group-kill discipline _stream_subprocess uses for the
    silence watchdog.
    """
    with _serial_procs_lock:
        proc = _serial_procs.get(fire_numbe)
    if proc is None:
        return False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Already exited or in another session — fall back to a
        # direct SIGTERM on the child PID so we at least try.
        try:
            proc.terminate()
        except Exception:
            pass
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    return True

# Kill fire_mapping_cli.py if stdout goes silent this long. Without this,
# a hung CLI would hold _gpu_lock forever and brick every mapping until
# the server is restarted.
_SUBPROCESS_SILENCE_TIMEOUT = 1800  # 30 minutes


def _stream_subprocess(cmd, cwd, on_line, tracker=None, fire_numbe=None):
    """Run *cmd*, pass each non-empty stdout line to ``on_line(text)``.

    Arms a watchdog timer that kills the process if stdout is silent
    for more than ``_SUBPROCESS_SILENCE_TIMEOUT`` seconds. The process
    is always reaped before this function returns, so the caller never
    has to worry about leaking PIDs.

    The child runs in its own session (new process group) so a kill
    can signal the whole group -- the CLI spawns helpers via
    ``subprocess.run`` (gdal, qgis) and without group kill those
    grandchildren would orphan to init on watchdog timeout.

    Returns ``(rc, killed)``. When the watchdog fires, ``rc`` is None
    and ``killed`` is True.

    *tracker* (optional) is a ``_ProgressTracker``; each line is fed to
    its ``observe()`` for stage detection before ``on_line`` runs.

    *fire_numbe* (optional) registers the spawned Popen in
    ``_serial_procs`` so the accept / cancel handlers can SIGTERM it
    via ``_terminate_serial_proc``. Always deregistered on return,
    even on exception.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        start_new_session=True,
    )

    if fire_numbe is not None:
        with _serial_procs_lock:
            _serial_procs[fire_numbe] = proc

    killed = [False]
    timer_box = [None]

    def _kill_group(sig):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:
            # Fall back to killing just the direct child so we at least
            # release the GPU lock holder even on exotic platforms.
            try:
                proc.send_signal(sig)
            except Exception:
                pass

    def _watchdog_fire():
        killed[0] = True
        _kill_group(signal.SIGKILL)

    def _arm():
        t_old = timer_box[0]
        if t_old is not None:
            t_old.cancel()
        t = threading.Timer(
            _SUBPROCESS_SILENCE_TIMEOUT, _watchdog_fire)
        t.daemon = True
        t.start()
        timer_box[0] = t

    try:
        _arm()
        for raw_line in iter(proc.stdout.readline, b''):
            _arm()
            text = raw_line.decode(errors='replace').rstrip()
            if text:
                if tracker is not None:
                    try:
                        tracker.observe(text)
                    except Exception:
                        pass
                on_line(text)
        rc = proc.wait()
    finally:
        # Deregister BEFORE the kill-cleanup so external callers
        # (accept/cancel handlers) can no longer see this proc once
        # it's on its way down. The killpg below is idempotent.
        if fire_numbe is not None:
            with _serial_procs_lock:
                if _serial_procs.get(fire_numbe) is proc:
                    del _serial_procs[fire_numbe]
        t_last = timer_box[0]
        if t_last is not None:
            t_last.cancel()
        if proc.poll() is None:
            # Give the group a chance to exit cleanly before the hammer.
            _kill_group(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except Exception:
                _kill_group(signal.SIGKILL)
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
    return (None if killed[0] else rc), killed[0]


def init_app(app_state: AppState):
    global state
    state = app_state
    _auth.init(app_state)
    _notifications.init(app_state)
    _progress.init(app_state)
    _mapping.init(app_state)
    # Persistence must initialise BEFORE cache_retention because
    # cache_retention captures _save_fire_state as a callback and that
    # callback dereferences persistence.state on each call.
    _persistence.init(
        app_state,
        _rebrush_procs, _rebrush_procs_lock,
        _compute_agreement, _compute_ml_area,
        _push_notification,
    )
    _cache_retention.init(
        app_state,
        _rebrush_procs, _rebrush_procs_lock,
        _accept_in_progress, _accept_in_progress_lock,
        _save_fire_state,
    )
    _brush.init(app_state, _rebrush_procs, _rebrush_procs_lock)
    _mapping_cmd.init(app_state)
    # _prepare uses the shared accept-in-progress registry and the
    # CSV-write file lock; pass them in so the cache sweeper and accept
    # handler all coordinate through the same primitives.
    _prepare.init(
        app_state,
        _set_fire_status,
        _accept_in_progress, _accept_in_progress_lock,
        _accept_file_lock,
        _CSV_FIELDNAMES,
    )
    # _workers needs every sibling module's init to have run first
    # (it imports their now-bound functions directly), and it needs
    # the same shared globals/helpers that _wire_handlers passes to
    # the mixins. Reuse the helpers dict shape — workers.init picks
    # out what it needs.
    _workers.init(app_state, {
        '_gpu_lock': _gpu_lock,
        '_batch_cancel': _batch_cancel,
        '_SUBPROCESS_SILENCE_TIMEOUT': _SUBPROCESS_SILENCE_TIMEOUT,
        '_set_fire_status': _set_fire_status,
        '_get_recommended_settings': _get_recommended_settings,
        '_clone_setting': _clone_setting,
        '_stream_subprocess': _stream_subprocess,
    })
    _wire_handlers(app_state)


def _set_fire_status(fire, new_status, error_msg=None):
    """Atomically update fire.status (and error_msg) under state.lock.

    Prevents readers like handle_api_status from observing a newly-ERROR
    status paired with a stale error_msg from the previous failure.
    """
    with state.lock:
        fire.status = new_status
        if error_msg is not None:
            fire.error_msg = error_msg
# Fixed CSV fieldnames for accepted_params.csv (prevents header drift)
_CSV_FIELDNAMES = [
    'fire_numbe', 'fire_size_ha', 'agreement_pct', 'padding', 'timestamp',
    'sample_rate', 'min_samples', 'max_samples', 'seed',
    'embed_bands', 'tsne_perplexity', 'tsne_learning_rate',
    'tsne_max_iter', 'tsne_init', 'tsne_n_components',
    'tsne_random_state',
    'controlled_ratio', 'hdbscan_min_samples',
    'rf_n_estimators', 'rf_max_depth', 'rf_max_features',
    'rf_random_state', 'contour_width',
    'brush_size', 'point_threshold', 'brush_all_segments',
]


# =========================================================================
# Recommended settings — flat list, no fire-size gating
# =========================================================================

def _clone_setting(s: dict) -> dict:
    return {
        'label': str(s.get('label', '') or ''),
        'params': dict(s.get('params', {})),
    }


def _get_recommended_settings(fire) -> list[dict]:
    """Return effective recommended settings for a fire.

    If the fire has a non-empty override list, use that; otherwise
    fall back to the global list. Always returns fresh copies so
    callers cannot mutate the canonical state.
    """
    override = getattr(fire, 'recommended_override', None)
    if override:
        return [_clone_setting(s) for s in override]
    return [_clone_setting(s) for s in state.recommended_settings]


# One-element container so handlers/batch.py can rebind the thread
# handle without needing ``global`` (which only resolves in the module
# that owns the name).
_batch_thread = [None]


# =========================================================================
# HTTP request handler
# =========================================================================

from .handlers import (
    BaseHandler, AuthRoutes, FireListRoutes, FireRoutes, MappingRoutes,
    SerialRoutes, RebrushRoutes, BatchRoutes, OpsRoutes, StaticRoutes,
)
from . import handlers as _handlers


def _wire_handlers(app_state):
    """Inject app-level shared globals into each handler mixin so
    unmodified method bodies see the same locks, registries, and
    helpers they used to see when they lived in this module.
    """
    helpers = {
        '_HERE': _HERE,
        '_gpu_lock': _gpu_lock,
        '_gpu_queue': _gpu_queue,
        '_gpu_queue_lock': _gpu_queue_lock,
        '_batch_thread': _batch_thread,
        '_batch_cancel': _batch_cancel,
        '_SUBPROCESS_SILENCE_TIMEOUT': _SUBPROCESS_SILENCE_TIMEOUT,
        '_serial_procs': _serial_procs,
        '_serial_procs_lock': _serial_procs_lock,
        '_rebrush_procs': _rebrush_procs,
        '_rebrush_procs_lock': _rebrush_procs_lock,
        '_accept_in_progress': _accept_in_progress,
        '_accept_in_progress_lock': _accept_in_progress_lock,
        '_accept_file_lock': _accept_file_lock,
        '_set_fire_status': _set_fire_status,
        '_terminate_serial_proc': _terminate_serial_proc,
        '_stream_subprocess': _stream_subprocess,
        '_get_recommended_settings': _get_recommended_settings,
        '_clone_setting': _clone_setting,
        '_batch_map_worker': _batch_map_worker,
        '_serial_map_worker': _serial_map_worker,
        '_jitter_hdbscan': _jitter_hdbscan,
        '_prepare_fire_sync': _prepare_fire_sync,
        '_accept_fire_sync': _accept_fire_sync,
        '_ensure_brush_comparison_in_cache': _ensure_brush_comparison_in_cache,
    }
    for mod in (_handlers.base, _handlers.auth, _handlers.fire_list,
                _handlers.fire, _handlers.mapping, _handlers.serial,
                _handlers.rebrush, _handlers.batch, _handlers.ops,
                _handlers.static):
        mod.init(app_state, helpers)


class FireHandler(
        AuthRoutes,
        FireListRoutes,
        FireRoutes,
        MappingRoutes,
        SerialRoutes,
        RebrushRoutes,
        BatchRoutes,
        OpsRoutes,
        StaticRoutes,
        BaseHandler,
        BaseHTTPRequestHandler,
):
    """Routes all HTTP requests for the fire mapping web interface.

    All method bodies live in the ``handlers`` subpackage; this class
    composes the mixins and ``_wire_handlers`` injects the shared
    state/locks/registries each mixin needs.
    """


def create_server(host: str = '0.0.0.0',
                  port: int = 8765) -> ThreadingHTTPServer:
    """Create and return the threaded HTTP server."""
    server = ThreadingHTTPServer((host, port), FireHandler)
    return server
