"""Stdlib-only web server for interactive fire mapping.

Uses http.server.ThreadingHTTPServer — no FastAPI, no uvicorn, no Jinja2.
SSE (Server-Sent Events) via fetch() for real-time console streaming.
"""

import datetime
import glob
import hashlib
import ipaddress
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
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs

import numpy as np
from osgeo import gdal

from .state import AppState, FireInfo, FireStatus
from .preview import generate_all_previews

gdal.UseExceptions()

_HERE = os.path.dirname(os.path.abspath(__file__))

# Global state — set by init_app() before the server starts
state: AppState = None

# GPU lock — serialises heavy operations (only one at a time)
_gpu_lock = threading.Lock()
_gpu_queue = 0            # number of tasks waiting or running
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


def _set_fire_status(fire, new_status, error_msg=None):
    """Atomically update fire.status (and error_msg) under state.lock.

    Prevents readers like handle_api_status from observing a newly-ERROR
    status paired with a stale error_msg from the previous failure.
    """
    with state.lock:
        fire.status = new_status
        if error_msg is not None:
            fire.error_msg = error_msg


_SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days in seconds


# =========================================================================
# Security helpers
# =========================================================================

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


def _atomic_yaml_dump(path: str, data, mode: int = 0o600):
    """Write YAML atomically via tmp + rename. Sets restrictive permissions.

    Uses a unique tmp suffix (pid + thread id) so concurrent writers to the
    same target path do not clobber each other's tmp file."""
    import yaml
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            # fsync before rename: os.replace is rename-atomic, but without
            # fsync the new file's bytes can still be in the page cache when
            # the directory entry flips. A power loss in that window leaves
            # a zero-length or truncated file after reboot even though the
            # rename "succeeded". Critical for fire_state.yaml et al.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# =========================================================================
# Stage-aware progress tracking — stage markers + running-median ETA
# =========================================================================

# Substring markers parsed from fire_mapping_cli.py stdout. Order matters:
# later stages override earlier ones when multiple match. 'full' is the
# from-scratch pipeline ([1..7/7]); 'resume' loads cached t-SNE+RF
# ([1..4/4]). 'brush' runs inside either. Stage IDs are stable across
# both pipelines so running medians pool both types.
# Each stage has a set of UNIQUE substrings. Avoid markers that overlap
# across stages (e.g. 'Mapping burn' printed by both RF step header and
# HDBSCAN body — we anchor to the numeric step header instead).
_STAGE_MARKERS = [
    ('load',     ('[1/4] Loading image', '[1/7] Loading image',
                  'Loading image')),
    ('hint',     ('[2/4] Loading hint', '[2/7] Loading hint',
                  'No hint provided — generating')),
    ('sample',   ('[3/7] Sampling', ' pixels sampled')),
    ('tsne',     ('[4/7] T-SNE', 'T-SNE embedding', 'T-SNE params',
                  'T-SNE done')),
    # 'rf' anchors on the [5/7] step header (full pipeline) so the RF
    # pill stays active for the entire RF training duration. Previously
    # the only marker was 'Forest mapping done' (end-of-stage) which
    # meant the pill flickered for a few ms between RF and HDBSCAN.
    ('rf',       ('[5/7] Mapping burn',)),
    # 'hdbscan' advances on:
    #   - 'Forest mapping done' — full mode, end of RF = start of HDBSCAN
    #   - '[4/4] Mapping burn'  — resume mode step header (HDBSCAN only,
    #                             no RF phase)
    #   - 'HDBSCAN done'        — legacy/safety fallback
    # Note: 'HDBSCAN min_cluster_size' is no longer a marker because
    # that line is printed BEFORE RF in full mode (see map_burn in
    # fire_mapping_cli.py), which would wrongly advance the stage
    # during RF training.
    ('hdbscan',  ('Forest mapping done', '[4/4] Mapping burn',
                  'HDBSCAN done')),
    # 'classify' fires on 'Burned clusters:' (end of classify_cluster's
    # internal loop) or 'Saving classification' (just before the save).
    # 'Classification saved' is kept as a legacy fallback. This way the
    # Classify pill is visible for the brief window between HDBSCAN
    # finishing and brush starting.
    ('classify', ('Burned clusters:', 'Saving classification',
                  'Classification saved')),
    ('brush',    ('Running class_brush.exe',
                  'class_brush done',
                  'class_brush post-processing')),
    ('figure',   ('Generating figures', '[7/7]',
                  'Comparison figure →',
                  'Brush comparison figure →')),
]
# Display order for UI rendering (left-to-right). Also defines the
# canonical stage sequence for "step N of M" math. 'hint', 'sample',
# 'classify' are small; hidden from ETA weighting if they never appear
# in medians yet (falls back to uniform weight).
_STAGE_ORDER_FULL = [
    'load', 'hint', 'sample', 'tsne', 'rf', 'hdbscan',
    'classify', 'brush', 'figure',
]
_STAGE_ORDER_RESUME = [
    'load', 'hint', 'hdbscan', 'classify', 'brush', 'figure',
]
_STAGE_LABELS = {
    'load':     'Loading image',
    'hint':     'Loading hint',
    'sample':   'Sampling',
    'tsne':     't-SNE embedding',
    'rf':       'Random Forest',
    'hdbscan':  'HDBSCAN clustering',
    'classify': 'Saving classification',
    'brush':    'class_brush post-process',
    'figure':   'Generating figures',
}

_STAGE_TIMINGS_MAX_SAMPLES = 30  # keep last N durations per stage


def _detect_stage(line: str) -> str | None:
    """Return the stage ID a CLI output line marks, or None.

    Scans markers in reverse stage order so later markers (e.g.
    'Classification saved') win over earlier ones on the same line
    (defensive — normally only one marker matches).
    """
    for stage_id, markers in reversed(_STAGE_MARKERS):
        for m in markers:
            if m in line:
                return stage_id
    return None


def _save_stage_timings():
    """Persist running stage durations to stage_timings.yaml."""
    if not state.shared_root:
        return
    try:
        path = os.path.join(state.shared_root, 'stage_timings.yaml')
        with state.lock:
            snap = {k: list(v) for k, v in state.stage_timings.items()}
        _atomic_yaml_dump(path, snap, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: stage_timings: {exc}\n')


def _load_stage_timings():
    """Restore stage timings on startup (best-effort)."""
    if not state.shared_root:
        return
    path = os.path.join(state.shared_root, 'stage_timings.yaml')
    if not os.path.isfile(path):
        return
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    state.stage_timings[str(k)] = [
                        float(x) for x in v[-_STAGE_TIMINGS_MAX_SAMPLES:]
                        if isinstance(x, (int, float)) and x > 0]
    except Exception as exc:
        sys.stderr.write(
            f'[load] WARNING: stage_timings: {exc}\n')


def _record_stage_duration(stage_id: str, seconds: float):
    """Append a stage duration to the running samples (bounded)."""
    if seconds <= 0 or seconds > 7200:  # sanity: reject > 2 hours
        return
    with state.lock:
        samples = state.stage_timings.setdefault(stage_id, [])
        samples.append(round(float(seconds), 2))
        if len(samples) > _STAGE_TIMINGS_MAX_SAMPLES:
            del samples[:-_STAGE_TIMINGS_MAX_SAMPLES]


def _stage_median(stage_id: str, fallback: float = 0.0) -> float:
    """Return the median of the last N durations for *stage_id*."""
    with state.lock:
        samples = list(state.stage_timings.get(stage_id, []))
    if not samples:
        return float(fallback)
    samples.sort()
    m = len(samples)
    mid = m // 2
    if m % 2:
        return float(samples[mid])
    return float((samples[mid - 1] + samples[mid]) / 2.0)


# Fallback durations (seconds) used when no samples exist. These are
# rough orders of magnitude so first-time users see *some* ETA rather
# than "unknown"; medians replace them within 1-2 runs.
_STAGE_FALLBACK = {
    'load': 15, 'hint': 3, 'sample': 10,
    'tsne': 180, 'rf': 90, 'hdbscan': 60,
    'classify': 3, 'brush': 30, 'figure': 10,
}


def _estimate_full_run_seconds(pipeline: str = 'full') -> float:
    order = (_STAGE_ORDER_FULL if pipeline == 'full'
             else _STAGE_ORDER_RESUME)
    return sum(_stage_median(s, _STAGE_FALLBACK.get(s, 0)) for s in order)


class _ProgressTracker:
    """Tracks pipeline stage transitions for one mapping run.

    Lives on ``fire.progress``; updated from the ``_on_line`` hook of
    ``_stream_subprocess``. Records each completed stage's duration
    into ``state.stage_timings`` so future runs can estimate ETA.
    """

    def __init__(self, fire, total_runs: int = 1, run_id: int = 1,
                 pipeline: str = 'full'):
        self.fire = fire
        self.total_runs = int(total_runs)
        self.run_id = int(run_id)
        self.pipeline = pipeline
        self.job_start = time.time()
        self.stage_start = self.job_start
        self.current_stage = 'init'
        self._run_starts: list = [self.job_start]
        self._completed_runs = 0
        # Snapshot onto the fire. We reassign dict so readers get a
        # consistent view without fine-grained locking.
        self._publish()

    @property
    def stage_order(self) -> list:
        return (_STAGE_ORDER_FULL if self.pipeline == 'full'
                else _STAGE_ORDER_RESUME)

    def observe(self, line: str):
        """Feed one CLI output line. Detects stage transitions."""
        if not line:
            return
        new_stage = _detect_stage(line)
        if new_stage and new_stage != self.current_stage:
            now = time.time()
            if self.current_stage in self.stage_order:
                _record_stage_duration(
                    self.current_stage, now - self.stage_start)
            self.current_stage = new_stage
            self.stage_start = now
            # 'load' on a later run marks a new pipeline invocation —
            # the previous run-boundary is the hand-off.
            self._publish()

    def mark_run_start(self, run_id: int, pipeline: str = 'full'):
        """Record the start of a new replicate in a serial sweep."""
        now = time.time()
        if self.current_stage in self.stage_order:
            _record_stage_duration(
                self.current_stage, now - self.stage_start)
        self.run_id = int(run_id)
        self.pipeline = pipeline
        self.stage_start = now
        self.current_stage = 'init'
        self._run_starts.append(now)
        self._completed_runs = max(0, len(self._run_starts) - 1)
        self._publish()

    def mark_run_end(self):
        """Record the tail-end stage duration + the total run duration.

        Totals are bucketed by pipeline ('full' vs 'resume') under a
        special key (``__total_full__`` / ``__total_resume__``) so the
        ETA snapshot can use a single robust number per-run rather
        than summing nine potentially-stale per-stage medians.
        """
        now = time.time()
        if self.current_stage in self.stage_order:
            _record_stage_duration(
                self.current_stage, now - self.stage_start)
        # Total for this run. self._run_starts[-1] is the last run's
        # start timestamp (set in mark_run_start). If mark_run_start
        # was never called (pathological — shouldn't happen), fall back
        # to job_start.
        run_start = (self._run_starts[-1] if self._run_starts
                     else self.job_start)
        total = now - run_start
        _record_stage_duration(
            f'__total_{self.pipeline}__', total)
        self._completed_runs += 1
        self._publish()
        _save_stage_timings()

    def _publish(self):
        """Write a snapshot to fire.progress (read by /progress)."""
        order = self.stage_order
        idx = (order.index(self.current_stage)
               if self.current_stage in order else -1)
        # Current run start = most recent entry in _run_starts. Falls
        # back to job_start if mark_run_start was never called (the
        # __init__-only publish, before the worker kicks off run 1).
        run_started_at = (self._run_starts[-1] if self._run_starts
                          else self.job_start)
        try:
            self.fire.progress = {
                'stage': self.current_stage,
                'stage_label': _STAGE_LABELS.get(
                    self.current_stage, self.current_stage),
                'stage_idx': idx + 1,   # 1-based for UI
                'total_stages': len(order),
                'stage_started_at': self.stage_start,
                'run_started_at': run_started_at,
                'job_started_at': self.job_start,
                'run_id': self.run_id,
                'total_runs': self.total_runs,
                'completed_runs': self._completed_runs,
                'pipeline': self.pipeline,
            }
        except Exception:
            pass


# Fudge factor — the "1.1×" safety margin on top of the past-run median
# so ETA doesn't collapse to 0 the instant a run crosses the median.
# Also the minimum remaining seconds we'll show while the run is still
# in flight (so the display doesn't render "0s left" until the run
# actually completes).
_ETA_FUDGE = 1.10
_ETA_FLOOR_S = 5


def _progress_snapshot(fire) -> dict:
    """Compute live progress + ETA from ``fire.progress``.

    ETA algorithm:
      * Primary signal is the running median of PAST TOTAL-RUN
        durations (``__total_full__`` / ``__total_resume__``). This
        dwarfs any per-stage sum in robustness — one number per run,
        directly comparable to current elapsed.
      * ``total_eta = median_total * 1.1 - run_elapsed`` for the
        current run, plus ``remaining_runs * median_per_run`` for the
        sweep tail.
      * If median drops below actual elapsed, extrapolate the current
        stage using ``max(median_total * 1.1 - elapsed,
        median_remaining_stages)`` so we keep climbing rather than
        clamping to zero.
      * If no history yet, return ``total_eta_s = None`` so the UI
        shows "Estimating…" instead of a misleading number.

    Stage-by-stage medians are still computed for the pill visual (so
    the user sees which stage is active), but not used for the ETA.
    """
    prog = dict(getattr(fire, 'progress', {}) or {})
    if not prog or fire.status not in (
            FireStatus.MAPPING, FireStatus.PREPARING):
        return {}
    now = time.time()
    pipeline = prog.get('pipeline', 'full')
    order = (_STAGE_ORDER_FULL if pipeline == 'full'
             else _STAGE_ORDER_RESUME)
    cur_stage = prog.get('stage', '')
    stage_idx = prog.get('stage_idx', 0)
    stage_started = prog.get('stage_started_at', now)
    job_started = prog.get('job_started_at', now)
    run_id = prog.get('run_id', 1)
    total_runs = max(1, prog.get('total_runs', 1))

    stage_elapsed = max(0.0, now - stage_started)
    job_elapsed = max(0.0, now - job_started)

    # --- per-stage ETA (used only for the "eta of current stage" label,
    # not the total) ---
    cur_median = _stage_median(cur_stage, _STAGE_FALLBACK.get(
        cur_stage, 0)) if cur_stage in order else 0
    stage_eta_s = max(0.0, cur_median * _ETA_FUDGE - stage_elapsed)

    # --- total ETA: run-level median of past runs ---
    with state.lock:
        full_samples = list(state.stage_timings.get('__total_full__', []))
        resume_samples = list(state.stage_timings.get(
            '__total_resume__', []))
    n_full = len(full_samples)
    n_resume = len(resume_samples)

    def _median(samples):
        if not samples:
            return 0.0
        s = sorted(samples)
        m = len(s)
        if m % 2:
            return float(s[m // 2])
        return float((s[m // 2 - 1] + s[m // 2]) / 2.0)

    median_full = _median(full_samples)
    median_resume = _median(resume_samples)

    # Current run: estimate from the total median of its pipeline.
    have_current_history = (
        (pipeline == 'full' and n_full > 0)
        or (pipeline == 'resume' and n_resume > 0))
    cur_run_elapsed = now - prog.get('run_started_at', job_started)

    if have_current_history:
        base = median_full if pipeline == 'full' else median_resume
        cur_run_remaining = max(
            _ETA_FLOOR_S, base * _ETA_FUDGE - cur_run_elapsed)
    else:
        cur_run_remaining = None

    # Future runs in this sweep. A serial sweep does full+resume×(K-1)
    # per setting; without setting boundaries here we use an average
    # shape. When we have both full and resume history, weight
    # accordingly; fall back to whichever we have.
    future_runs = max(0, total_runs - run_id)
    if n_full and n_resume:
        # A sweep of S settings × K replicates = S full + S×(K-1) resume.
        # Without S and K here we approximate: ratio observed from past.
        per_run_avg = (median_full + median_resume * 2) / 3.0
    elif n_full:
        per_run_avg = median_full
    elif n_resume:
        per_run_avg = median_resume
    else:
        per_run_avg = None

    if cur_run_remaining is not None and per_run_avg is not None:
        total_eta_s = cur_run_remaining + future_runs * per_run_avg
        total_eta_s = max(_ETA_FLOOR_S, total_eta_s)
    elif cur_run_remaining is not None:
        # Know current run but no per-run estimate for future runs.
        total_eta_s = cur_run_remaining
        if future_runs > 0:
            # Flag as partial — UI will render "…".
            pass
    else:
        total_eta_s = None  # "Estimating…"

    # --- percent ---
    # Prefer elapsed/median ratio on the current run (far more accurate
    # than summing stage fractions). Floor at 1, ceil at 99.
    if have_current_history:
        base = median_full if pipeline == 'full' else median_resume
        if base > 0:
            cur_run_frac = min(0.99, cur_run_elapsed / (base * _ETA_FUDGE))
        else:
            cur_run_frac = 0.0
    elif cur_stage in order:
        # First-ever run: fall back to stage-index fraction.
        cur_run_frac = order.index(cur_stage) / max(1, len(order))
    else:
        cur_run_frac = 0.0
    overall_pct = min(99, max(1, int(round(
        100 * ((run_id - 1) + cur_run_frac) / total_runs))))

    return {
        'stage': cur_stage,
        'stage_label': prog.get('stage_label', cur_stage),
        'stage_idx': stage_idx,
        'total_stages': prog.get('total_stages', len(order)),
        'stage_elapsed_s': round(stage_elapsed, 1),
        'stage_eta_s': round(stage_eta_s, 1),
        'run_id': run_id,
        'total_runs': total_runs,
        'completed_runs': prog.get('completed_runs', 0),
        'pipeline': pipeline,
        'job_elapsed_s': round(job_elapsed, 1),
        # None → UI renders "Estimating…" (no history for this pipeline).
        'total_eta_s': (None if total_eta_s is None
                        else round(total_eta_s, 1)),
        'percent': overall_pct,
        'n_samples_full': n_full,
        'n_samples_resume': n_resume,
    }


# =========================================================================
# Toast notifications — per-session queues + broadcast channel
# =========================================================================

_NOTIFICATIONS_MAX_PER_SESSION = 50
_NOTIFICATIONS_MAX_BROADCAST = 20
_NOTIFICATION_TTL_SECONDS = 7 * 24 * 3600  # 7 days

_NOTIFICATION_KINDS = {'info', 'success', 'warning', 'error'}


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


# =========================================================================
# Cache retention — prune .web_cache by size + age, never touch canonical
# =========================================================================

_cache_sweep_lock = threading.Lock()


def _save_cache_retention():
    if not state.shared_root:
        return
    try:
        path = os.path.join(state.shared_root, 'cache_retention.yaml')
        with state.lock:
            snap = {
                'config': dict(state.cache_retention),
                'last_sweep': float(state.cache_last_sweep),
            }
        _atomic_yaml_dump(path, snap, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: cache_retention: {exc}\n')


def _load_cache_retention():
    if not state.shared_root:
        return
    path = os.path.join(state.shared_root, 'cache_retention.yaml')
    if not os.path.isfile(path):
        return
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get('config', {}) or {}
        with state.lock:
            for k in ('max_gb', 'max_age_days', 'sweep_interval_hours'):
                if k in cfg:
                    try:
                        state.cache_retention[k] = (
                            float(cfg[k]) if k == 'max_gb'
                            else int(cfg[k]))
                    except (TypeError, ValueError):
                        pass
            if 'enabled' in cfg:
                state.cache_retention['enabled'] = bool(cfg['enabled'])
            try:
                state.cache_last_sweep = float(data.get('last_sweep', 0))
            except (TypeError, ValueError):
                state.cache_last_sweep = 0.0
    except Exception as exc:
        sys.stderr.write(
            f'[load] WARNING: cache_retention: {exc}\n')


def _dir_bytes_and_mtime(path: str) -> tuple[int, float]:
    """Return (total_size_bytes, latest_mtime) for a directory tree."""
    total = 0
    latest = 0.0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                st = os.stat(fp)
                total += int(st.st_size)
                if st.st_mtime > latest:
                    latest = st.st_mtime
            except OSError:
                continue
    return total, latest


def _cache_scan() -> dict:
    """Summarise all .web_cache dirs across all years.

    Returns {entries: [{fire_numbe, year, bytes, mtime_s, pinned,
    pin_reason}], total_bytes, pinned_bytes, by_year: {year: bytes}}.
    """
    entries: list = []
    total_bytes = 0
    pinned_bytes = 0
    by_year: dict = {}
    years = getattr(state, 'outdirs_by_year', {}) or {
        state.active_year: state.output_root}
    # Pin fires the analyst might still be working on. READY and MAPPED
    # mean there's cached work the user hasn't accepted yet — evicting
    # it would wipe their progress silently. ACCEPTED is safe to
    # reclaim because the canonical dir has everything durable.
    # PREPARING / MAPPING are hard pins (in-flight I/O).
    with state.lock:
        hard_pin = {FireStatus.PREPARING, FireStatus.MAPPING}
        soft_pin = {FireStatus.READY, FireStatus.MAPPED}
        status_by_fire = {fn: f.status for fn, f in state.fires.items()}
    hard_fires = {fn for fn, s in status_by_fire.items() if s in hard_pin}
    soft_fires = {fn for fn, s in status_by_fire.items() if s in soft_pin}
    with _rebrush_procs_lock:
        hard_fires |= set(_rebrush_procs.keys())
    # Accept-in-progress is a hard pin: the accept handler is copying
    # files from cache_dir into the canonical output and must not have
    # its source dir yanked out from under it.
    with _accept_in_progress_lock:
        hard_fires |= set(_accept_in_progress)
    for year, outdir in years.items():
        cache_root = os.path.join(outdir, '.web_cache')
        if not os.path.isdir(cache_root):
            continue
        for entry in os.scandir(cache_root):
            if not entry.is_dir():
                continue
            size, mtime = _dir_bytes_and_mtime(entry.path)
            is_hard = entry.name in hard_fires
            is_soft = entry.name in soft_fires
            is_pinned = is_hard or is_soft
            pin_reason = ('in-flight' if is_hard
                          else 'user-work' if is_soft
                          else '')
            entries.append({
                'fire_numbe': entry.name,
                'year': int(year),
                'bytes': size,
                'mtime_s': mtime,
                'path': entry.path,
                'pinned': is_pinned,
                'hard_pinned': is_hard,
                'pin_reason': pin_reason,
            })
            total_bytes += size
            by_year[int(year)] = by_year.get(int(year), 0) + size
            if is_pinned:
                pinned_bytes += size
    return {
        'entries': entries,
        'total_bytes': total_bytes,
        'pinned_bytes': pinned_bytes,
        'by_year': by_year,
    }


def _cache_sweep(dry_run: bool = False) -> dict:
    """Evict unpinned cache dirs until size <= max_gb and age <= max_age_days.

    Returns a summary dict. Safe to call concurrently — serialised by
    ``_cache_sweep_lock`` so overlapping triggers are no-ops.
    """
    if not _cache_sweep_lock.acquire(blocking=False):
        return {'status': 'busy', 'pruned_bytes': 0, 'pruned_fires': []}
    try:
        with state.lock:
            cfg = dict(state.cache_retention)
        if not cfg.get('enabled', True) and not dry_run:
            return {'status': 'disabled', 'pruned_bytes': 0,
                    'pruned_fires': []}
        max_bytes = int(float(cfg.get('max_gb', 20.0)) * (1024 ** 3))
        max_age_s = int(cfg.get('max_age_days', 30)) * 86400
        now = time.time()

        scan = _cache_scan()
        entries = scan['entries']
        total = scan['total_bytes']

        # Age-based eviction ignores soft pins — truly stale caches
        # (older than max_age_days) always go, because the user
        # obviously isn't actively working on them. Hard pins
        # (in-flight jobs) are never touched.
        age_cutoff = now - max_age_s
        pruned: list = []
        for e in list(entries):
            if e.get('hard_pinned'):
                continue
            if e['mtime_s'] and e['mtime_s'] < age_cutoff:
                pruned.append(e)

        # Size-based eviction respects soft pins — we don't evict
        # READY/MAPPED (user might still be working on the brush params)
        # just to stay under the size limit.
        pruned_ids = {(e['path']) for e in pruned}
        evictable = [e for e in entries
                     if not e.get('pinned')
                     and e['path'] not in pruned_ids]
        evictable.sort(key=lambda e: e['mtime_s'] or 0)
        projected_total = total - sum(e['bytes'] for e in pruned)
        while projected_total > max_bytes and evictable:
            e = evictable.pop(0)
            pruned.append(e)
            projected_total -= e['bytes']

        if dry_run:
            return {
                'status': 'dry_run',
                'pruned_bytes': sum(e['bytes'] for e in pruned),
                'pruned_fires': [e['fire_numbe'] for e in pruned],
                'total_bytes': total,
                'max_bytes': max_bytes,
            }

        # Apply.
        actually_pruned = []
        for e in pruned:
            try:
                shutil.rmtree(e['path'], ignore_errors=False)
                actually_pruned.append(e)
                # Best-effort: clear the fire's in-memory cache_dir
                # reference so callers don't read a stale path. Never
                # demote ACCEPTED — its canonical dir is intact and
                # still authoritative. MAPPED/READY cache only gets
                # pruned by the age-based path (size path respects the
                # soft pin), so demotion to PENDING is reasonable in
                # that case — the analyst would have to re-prepare
                # anyway since the crop/hint files are gone.
                with state.lock:
                    fire = state.fires.get(e['fire_numbe'])
                    if fire and fire.cache_dir == e['path']:
                        fire.cache_dir = ''
                        fire.available_views = []
                        fire.last_comparison = ''
                        if fire.status in (FireStatus.READY,
                                           FireStatus.MAPPED):
                            fire.status = FireStatus.PENDING
                            fire.last_params = {}
                            fire.agreement_pct = -1.0
                            fire.ml_area_ha = -1.0
                            fire.serial_results = []
                            fire.serial_settings = []
                            fire.progress = {}
            except Exception as exc:
                sys.stderr.write(
                    f'[cache] WARNING: could not remove {e["path"]}: '
                    f'{exc}\n')

        pruned_bytes = sum(e['bytes'] for e in actually_pruned)
        with state.lock:
            state.cache_last_sweep = time.time()
        _save_cache_retention()
        if actually_pruned:
            _save_fire_state()
            sys.stderr.write(
                f'[cache] Pruned {len(actually_pruned)} fire(s), '
                f'{pruned_bytes / (1024**2):.1f} MB freed.\n')
        return {
            'status': 'ok',
            'pruned_bytes': pruned_bytes,
            'pruned_fires': [e['fire_numbe'] for e in actually_pruned],
            'total_bytes': total,
            'max_bytes': max_bytes,
            'after_bytes': projected_total,
        }
    finally:
        _cache_sweep_lock.release()


def _cache_sweep_loop():
    """Background thread: periodically run _cache_sweep()."""
    while True:
        try:
            with state.lock:
                interval_h = int(state.cache_retention.get(
                    'sweep_interval_hours', 6))
            interval = max(1, interval_h) * 3600
            time.sleep(interval)
            _cache_sweep(dry_run=False)
        except Exception as exc:
            sys.stderr.write(f'[cache] sweep loop error: {exc}\n')
            time.sleep(600)


# Login rate limiting
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_WINDOW_SECONDS = 300  # 5 minutes


def _check_login_rate(ip: str) -> bool:
    """Return True if the IP is under the rate limit."""
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


# =========================================================================
# Parameter validation — typed bounds for subprocess arguments
# =========================================================================

_PARAM_SPEC = {
    'seed':                 ('int',   0, 2**31 - 1),
    'rf_n_estimators':      ('int',   1, 2000),
    'rf_max_depth':         ('int',   1, 100),
    'rf_max_features':      ('choice', {'sqrt', 'log2', 'auto'}),
    'rf_random_state':      ('int',   0, 2**31 - 1),
    'controlled_ratio':     ('float', 0.01, 2.0),
    'hdbscan_min_samples':  ('int',   1, 10000),
    'tsne_perplexity':      ('float', 1.0, 1000.0),
    'tsne_learning_rate':   ('float', 1.0, 10000.0),
    'tsne_max_iter':        ('int',   100, 10000),
    'tsne_init':            ('choice', {'random', 'pca'}),
    'tsne_n_components':    ('int',   2, 3),
    'tsne_random_state':    ('int',   0, 2**31 - 1),
    'contour_width':        ('float', 0.0, 10.0),
    'sample_rate':          ('float', 0.001, 1.0),
    'min_samples':          ('int',   1, 1000000),
    'max_samples':          ('int',   1, 1000000),
    'brush_size':           ('int',   1, 10000),
    'point_threshold':      ('int',   1, 10_000_000),
    'brush_all_segments':   ('bool',),
}


def _validate_param(key: str, raw):
    """Validate and coerce a single parameter. Raises ValueError on bad input."""
    if key not in _PARAM_SPEC:
        return raw
    spec = _PARAM_SPEC[key]
    kind = spec[0]
    if kind == 'int':
        _, lo, hi = spec
        v = int(float(raw))
        if not (lo <= v <= hi):
            raise ValueError(f'{key}={raw} out of range [{lo}, {hi}]')
        return v
    if kind == 'float':
        _, lo, hi = spec
        v = float(raw)
        if not (lo <= v <= hi):
            raise ValueError(f'{key}={raw} out of range [{lo}, {hi}]')
        return v
    if kind == 'choice':
        _, allowed = spec
        s = str(raw)
        if s not in allowed:
            raise ValueError(
                f'{key}={raw} must be one of {sorted(allowed)}')
        return s
    if kind == 'bool':
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off', ''):
            return False
        raise ValueError(f'{key}={raw} must be boolean')
    return raw


def _validate_embed_bands(eb) -> str | None:
    """Validate embed_bands (comma-separated positive ints). Returns cleaned string."""
    if not eb or not str(eb).strip():
        return None
    parts = [p.strip() for p in str(eb).split(',') if p.strip()]
    if not all(p.isdigit() and 1 <= int(p) <= 999 for p in parts):
        raise ValueError(f'Invalid embed_bands: {eb!r}')
    return ','.join(parts)


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


def _save_sessions():
    """Persist sessions to disk (tokens are already hashed in memory)."""
    if not state.session_file:
        return
    try:
        with state.lock:
            snap = dict(state.sessions)
        _atomic_yaml_dump(state.session_file, snap)
    except Exception as exc:
        sys.stderr.write(f'[save] WARNING: Failed to save sessions: {exc}\n')
        sys.stderr.flush()


def _save_settings():
    """Persist recommended settings to shared_root (not per-year)."""
    try:
        settings_path = state.settings_file or os.path.join(
            state.shared_root or state.output_root,
            'recommended_settings.yaml')
        payload = {
            'k_runs_per_setting': int(state.k_runs_per_setting),
            'k_jitter': int(state.k_jitter),
            'settings': [
                {'label': str(s.get('label', '')),
                 'params': dict(s.get('params', {}))}
                for s in state.recommended_settings
            ],
        }
        _atomic_yaml_dump(settings_path, payload, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save settings: {exc}\n')
        sys.stderr.flush()


def _save_notes():
    """Persist all fire notes to notes.yaml."""
    try:
        notes_path = os.path.join(state.output_root, 'notes.yaml')
        with state.lock:
            notes_data = {fn: fire.notes
                          for fn, fire in state.fires.items()
                          if fire.notes}
        _atomic_yaml_dump(notes_path, notes_data, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save notes: {exc}\n')
        sys.stderr.flush()


def _save_fire_state():
    """Persist per-fire state to fire_state.yaml so mapped fires survive restart."""
    # Refuse to overwrite a file we couldn't parse on boot. Without this
    # guard, a corrupt-but-recoverable fire_state.yaml would be clobbered
    # by the next save with the stripped init-from-GDF state, permanently
    # destroying ACCEPTED/MAPPED markers that were still in the old file.
    if state.fire_state_load_failed:
        return
    try:
        data = {}
        # Hold the lock across the entire snapshot so no fire attribute is
        # read while another thread is mutating it. state.lock is an RLock
        # so any inner helpers that re-acquire it remain safe.
        with state.lock:
            for fn, fire in state.fires.items():
                # Only persist fires with meaningful state beyond PENDING
                if (fire.status == FireStatus.PENDING and not fire.hidden
                        and not fire.cache_dir):
                    continue
                entry = {
                    'status': fire.status.value,
                    'hidden': fire.hidden,
                }
                if fire.cache_dir:
                    entry['cache_dir'] = fire.cache_dir
                if fire.crop_bin:
                    entry['crop_bin'] = fire.crop_bin
                if fire.hint_bin:
                    entry['hint_bin'] = fire.hint_bin
                if fire.perim_bin:
                    entry['perim_bin'] = fire.perim_bin
                if fire.viirs_bin:
                    entry['viirs_bin'] = fire.viirs_bin
                if fire.crop_w:
                    entry['crop_w'] = fire.crop_w
                    entry['crop_h'] = fire.crop_h
                if fire.padding_used:
                    entry['padding_used'] = fire.padding_used
                if fire.acc_start:
                    entry['acc_start'] = fire.acc_start
                    entry['acc_end'] = fire.acc_end
                if fire.perimeter_type:
                    entry['perimeter_type'] = fire.perimeter_type
                if fire.sample_size:
                    entry['sample_size'] = fire.sample_size
                if fire.available_views:
                    entry['available_views'] = list(fire.available_views)
                if fire.last_comparison:
                    entry['last_comparison'] = fire.last_comparison
                if fire.last_params:
                    entry['last_params'] = dict(fire.last_params)
                if fire.ml_area_ha >= 0:
                    entry['ml_area_ha'] = fire.ml_area_ha
                if fire.agreement_pct >= 0:
                    entry['agreement_pct'] = fire.agreement_pct
                if fire.previously_accepted:
                    entry['previously_accepted'] = True
                if fire.recommended_override:
                    entry['recommended_override'] = [
                        {'label': str(s.get('label', '')),
                         'params': dict(s.get('params', {}))}
                        for s in fire.recommended_override
                    ]
                if fire.last_preset:
                    entry['last_preset'] = str(fire.last_preset)
                if fire.last_cancel_reason:
                    entry['last_cancel_reason'] = str(
                        fire.last_cancel_reason)
                # Persist serial gallery state so the results gallery
                # survives restart. Without this, fire.serial_results
                # defaults back to [] on boot and the left-pane ML overlay
                # (restored from the canonical classified.bin) appears
                # without the per-run cards.
                if fire.serial_results:
                    entry['serial_results'] = [
                        {k: v for k, v in r.items()}
                        for r in fire.serial_results
                    ]
                if fire.serial_settings:
                    entry['serial_settings'] = [
                        {'label': str(s.get('label', '')),
                         'params': dict(s.get('params', {}))}
                        for s in fire.serial_settings
                    ]
                data[fn] = entry

        state_path = os.path.join(state.output_root, 'fire_state.yaml')
        _atomic_yaml_dump(state_path, data, mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save fire state: {exc}\n')
        sys.stderr.flush()


def _load_fire_state():
    """Restore per-fire state from fire_state.yaml after init_fires_from_gdf."""
    state_path = os.path.join(state.output_root, 'fire_state.yaml')
    if not os.path.isfile(state_path):
        return

    try:
        import yaml
        with open(state_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        # Rotate the unparseable file aside so the refuse-to-save guard
        # in _save_fire_state does not need the original path free, and
        # an operator has something to inspect/recover. Then flip the
        # load-failed flag to block future saves from clobbering the
        # on-disk original.
        backup = f'{state_path}.corrupt-{int(time.time())}'
        try:
            shutil.copy2(state_path, backup)
        except OSError:
            backup = '<copy failed>'
        sys.stderr.write(
            f'[load] CRITICAL: fire_state.yaml failed to parse ({exc}). '
            f'Copied aside to {backup}. Saves are now blocked; '
            f'investigate and restart before next save.\n')
        sys.stderr.flush()
        state.fire_state_load_failed = True
        return

    restored = 0
    for fn, entry in data.items():
        if fn not in state.fires:
            continue
        fire = state.fires[fn]

        # Restore hidden flag
        if entry.get('hidden'):
            fire.hidden = True

        # Restore per-fire recommended override (independent of cache_dir)
        override = entry.get('recommended_override')
        if isinstance(override, list) and override:
            clean = []
            for row in override:
                if not isinstance(row, dict):
                    continue
                clean.append({
                    'label': str(row.get('label', '') or ''),
                    'params': dict(row.get('params', {}) or {}),
                })
            if clean:
                fire.recommended_override = clean
        if entry.get('last_preset'):
            fire.last_preset = str(entry['last_preset'])
        if entry.get('last_cancel_reason'):
            fire.last_cancel_reason = str(entry['last_cancel_reason'])

        saved_status = entry.get('status', 'pending')

        # Don't downgrade: if init_fires_from_gdf already found ACCEPTED,
        # only restore supplementary fields, not status
        if fire.status == FireStatus.ACCEPTED and saved_status != 'accepted':
            # Just restore hidden and skip — the accepted state from disk
            # is authoritative
            continue

        # Restore cache paths — but only if they still exist on disk
        cache_dir = entry.get('cache_dir', '')
        if cache_dir and os.path.isdir(cache_dir):
            fire.cache_dir = cache_dir
            fire.crop_bin = entry.get('crop_bin', '')
            fire.hint_bin = entry.get('hint_bin', '')
            fire.perim_bin = entry.get('perim_bin', '')
            fire.viirs_bin = entry.get('viirs_bin', '')
            fire.crop_w = entry.get('crop_w', 0)
            fire.crop_h = entry.get('crop_h', 0)
            fire.padding_used = entry.get('padding_used', 0.0)
            fire.acc_start = entry.get('acc_start', '')
            fire.acc_end = entry.get('acc_end', '')
            fire.perimeter_type = entry.get('perimeter_type', '')
            fire.sample_size = entry.get('sample_size', 0)
            fire.available_views = entry.get('available_views', [])
            fire.last_comparison = entry.get('last_comparison', '')
            fire.last_params = entry.get('last_params', {})
            fire.ml_area_ha = entry.get('ml_area_ha', -1.0)
            fire.agreement_pct = entry.get('agreement_pct', -1.0)
            fire.previously_accepted = entry.get(
                'previously_accepted', False)

            # Restore serial gallery. Each entry's classified.bin must
            # still be on disk — drop entries whose files were deleted
            # out of band (manual cleanup, external wipe). The 'previous
            # accepted snapshot' run (is_previous=True) is restored
            # unconditionally; its backing files live in accepted output,
            # not the .web_cache directory.
            saved_serial = entry.get('serial_results')
            if isinstance(saved_serial, list) and saved_serial:
                clean_serial = []
                for r in saved_serial:
                    if not isinstance(r, dict):
                        continue
                    clf = r.get('classified', '')
                    if r.get('is_previous'):
                        clean_serial.append(dict(r))
                        continue
                    if clf and os.path.isfile(clf):
                        clean_serial.append(dict(r))
                if clean_serial:
                    fire.serial_results = clean_serial

            saved_settings = entry.get('serial_settings')
            if isinstance(saved_settings, list) and saved_settings:
                clean_settings = []
                for s in saved_settings:
                    if not isinstance(s, dict):
                        continue
                    clean_settings.append({
                        'label': str(s.get('label', '') or ''),
                        'params': dict(s.get('params', {}) or {}),
                    })
                if clean_settings:
                    fire.serial_settings = clean_settings

            # Fallback: if fire_state.yaml predates the serial_results
            # persistence change, reconstruct a partial gallery by
            # scanning the cache dir for per-run classified.bin files.
            # Setting labels and params are lost (they lived only in
            # memory), but run_id, file paths, agreement_pct, and
            # ml_area_ha can be recovered. Accept/Rebrush still work.
            if not fire.serial_results:
                try:
                    import re as _re
                    pat = _re.compile(
                        r'^' + _re.escape(fn)
                        + r'_serial_(\d+)_classified\.bin$')
                    scanned = []
                    for name in os.listdir(cache_dir):
                        m = pat.match(name)
                        if not m:
                            continue
                        rid = int(m.group(1))
                        clf = os.path.join(cache_dir, name)
                        comp = os.path.join(
                            cache_dir, f'{fn}_serial_{rid}.png')
                        entry_ = {
                            'run_id': rid,
                            'setting_idx': 0,
                            'run_idx': 0,
                            'setting_label': '',
                            'params': {},
                            'classified': clf,
                            'comparison': comp if os.path.isfile(comp)
                                          else '',
                            'agreement_pct': _compute_agreement(
                                fire, clf_path=clf),
                            'ml_area_ha': _compute_ml_area(fire, clf),
                        }
                        scanned.append(entry_)
                    if scanned:
                        scanned.sort(key=lambda r: r['run_id'])
                        fire.serial_results = scanned
                        sys.stderr.write(
                            f'[load] Rebuilt {len(scanned)} serial '
                            f'gallery entries for {fn} from cache scan '
                            f'(no persisted serial_results).\n')
                except Exception as exc:
                    sys.stderr.write(
                        f'[load] WARNING: serial gallery rebuild '
                        f'failed for {fn}: {exc}\n')
                    sys.stderr.flush()

            # Validate critical files exist before restoring status
            if saved_status in ('ready', 'mapped'):
                crop_ok = (fire.crop_bin
                           and os.path.isfile(fire.crop_bin))
                hint_ok = (fire.hint_bin
                           and os.path.isfile(fire.hint_bin))
                if crop_ok and hint_ok:
                    fire.status = FireStatus(saved_status)
                    restored += 1
                else:
                    fire.status = FireStatus.PENDING
            elif saved_status == 'accepted':
                fire.status = FireStatus.ACCEPTED
                restored += 1
            # MAPPING/PREPARING on disk means crashed mid-work — reset
            elif saved_status in ('mapping', 'preparing'):
                # Check if a mapped result exists in cache
                comp = os.path.join(
                    cache_dir, f'{fn}_comparison.png')
                if os.path.isfile(comp):
                    fire.status = FireStatus.MAPPED
                    fire.last_comparison = comp
                    restored += 1
                else:
                    fire.status = FireStatus.READY
        elif saved_status == 'accepted':
            fire.status = FireStatus.ACCEPTED

    if restored:
        sys.stderr.write(
            f'[load] Restored state for {restored} fire(s) '
            f'from fire_state.yaml\n')
        sys.stderr.flush()


def _save_active_year():
    """Persist the currently-active year so it survives restarts."""
    if not state.shared_root:
        return
    try:
        path = os.path.join(state.shared_root, 'active_year.yaml')
        _atomic_yaml_dump(path, {'active_year': int(state.active_year)},
                          mode=0o644)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save active year: {exc}\n')
        sys.stderr.flush()


def _switch_year(year: int) -> tuple[bool, str]:
    """Re-point active-year state to *year* without restarting the server.

    Fails fast if any long-running job is in flight (one-shot mapping,
    batch mapping, rebrush). Returns (ok, message). On success the caller
    should persist and notify clients to reload.
    """
    if not isinstance(year, int):
        return False, 'year must be an int'
    if year not in state.rasters_by_year:
        return False, f'unknown year {year}'
    if year == state.active_year:
        return True, 'already active'

    # Busy checks — refuse mid-job rather than corrupting in-flight state.
    with state.lock:
        if state.current_job is not None:
            return False, ('A mapping job is running '
                           f'(fire {state.current_job.get("fire_numbe")}). '
                           'Wait for it to finish or cancel it.')
        if state.batch_status and state.batch_status.get('running'):
            return False, ('A batch mapping is running '
                           f'(fire {state.batch_status.get("current_fire") or "?"}). '
                           'Cancel it from the fire list or wait.')
    # Rebrush workers run outside _gpu_lock — check explicitly so the
    # switch doesn't wipe fire_state.yaml out from under a running
    # class_brush.exe.
    with _rebrush_procs_lock:
        if _rebrush_procs:
            active = ', '.join(sorted(_rebrush_procs.keys())[:3])
            more = ('' if len(_rebrush_procs) <= 3
                    else f' (+{len(_rebrush_procs) - 3} more)')
            return False, (f'A rebrush is running on fire {active}{more}. '
                           'Cancel it or wait.')

    from batch_fire_mapping.run_fire_mapping import (
        get_raster_info, load_all_viirs)
    import geopandas as _gpd
    from shapely.geometry import box as _shapely_box

    new_raster = state.rasters_by_year[year]
    new_outdir = state.outdirs_by_year[year]
    new_viirs  = state.viirs_shp_dirs_by_year[year]

    # Persist current year's state before swapping so nothing is lost.
    _save_fire_state()

    try:
        crs_wkt, gt, W, H = get_raster_info(new_raster)
    except Exception as exc:
        return False, f'Failed to read raster: {exc}'

    # Re-project + spatial-filter the cached raw polygon GDF to the new
    # raster. Falling back to re-reading the shapefile if the cache is
    # missing (e.g. a stale AppState).
    raw = state.polygon_gdf_raw
    if raw is None:
        try:
            raw = _gpd.read_file(state.polygon_file)
            state.polygon_gdf_raw = raw
        except Exception as exc:
            return False, f'Failed to read polygons: {exc}'

    from batch_fire_mapping.run_fire_mapping import (
        _wkt_to_pyproj_crs, raster_native_extent)
    import pandas as _pd
    try:
        # Filter by FIRE_YEAR first (cheap, drops most rows before reproject)
        if 'FIRE_YEAR' in raw.columns:
            fy = _pd.to_numeric(raw['FIRE_YEAR'], errors='coerce')
            gdf = raw[fy == year].copy()
        else:
            gdf = raw.copy()
        if gdf.empty:
            return False, (f'No polygons with FIRE_YEAR={year} in the '
                           'shapefile.')
        gdf = gdf.to_crs(_wkt_to_pyproj_crs(crs_wkt))
        xmin, ymin, xmax, ymax = raster_native_extent(gt, W, H)
        gdf = gdf[gdf.geometry.intersects(
            _shapely_box(xmin, ymin, xmax, ymax))].copy()
    except Exception as exc:
        return False, f'Failed to filter polygons: {exc}'

    if gdf.empty:
        return False, (f'No polygons with FIRE_YEAR={year} overlap the '
                       f'year-{year} raster extent.')

    # Load VIIRS for the new year (traditional mode -> empty GDF)
    if state.perimeter_mode == 'traditional':
        viirs_gdf = _gpd.GeoDataFrame()
    else:
        try:
            viirs_gdf = load_all_viirs(new_viirs, crs_wkt)
        except Exception as exc:
            return False, f'Failed to load VIIRS: {exc}'

    # Atomic swap under the state lock. Rebuild fires from the new gdf;
    # _load_fire_state reads the new outdir's fire_state.yaml afterwards.
    with state.lock:
        state.active_year    = year
        state.raster_path    = new_raster
        state.output_root    = new_outdir
        state.viirs_shp_dir  = new_viirs
        state.raster_crs     = crs_wkt
        state.raster_gt      = gt
        state.raster_W       = W
        state.raster_H       = H
        state.gdf            = gdf
        state.viirs_gdf      = viirs_gdf
        state.fires          = {}
        state.init_fires_from_gdf()

    _load_fire_state()
    _save_active_year()

    sys.stderr.write(
        f'[switch_year] Active year -> {year}, {len(state.fires)} fire(s)\n')
    sys.stderr.flush()
    _push_notification(
        None, 'info',
        f'Active year switched to {year}',
        f'{len(state.fires)} fire(s) loaded. Reload the page to see the '
        'new fire list.',
        action={'url': '/', 'label': 'Reload fire list'})
    return True, 'ok'


def _compute_ml_area(fire: 'FireInfo',
                     clf_path: str = None) -> float:
    """Compute ML burned area in hectares from a classified raster.

    Returns area in ha or -1 if computation fails.
    """
    if clf_path is None:
        clf_path = os.path.join(
            fire.cache_dir,
            f'{fire.fire_numbe}_crop.bin_classified.bin')
    if not os.path.isfile(clf_path):
        return -1.0
    try:
        gt = state.raster_gt
        pixel_area_m2 = abs(gt[1] * gt[5])
        ds = gdal.Open(clf_path, gdal.GA_ReadOnly)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        burned_px = int(np.nansum(arr > 0))
        ml_area_ha = burned_px * pixel_area_m2 / 10000.0
        return round(ml_area_ha, 2)
    except Exception as exc:
        sys.stderr.write(
            f'[ml_area] WARNING: Failed to compute ML area: {exc}\n')
        return -1.0


def _overlay_mask_on_post(fire: 'FireInfo', raster_path: str,
                          out_name: str, color: tuple):
    """Overlay a binary raster on the post-fire preview.

    *color* is (r, g, b) floats 0-1 for the tint.
    Produces a pixel-aligned PNG at the same dimensions as post.png.

    When the overlay raster has different dimensions from the current
    crop (e.g. a previously accepted classification after re-cropping
    with different padding), uses GDAL geotransforms to place it at
    the correct geographic position rather than naively stretching.
    """
    try:
        post_path = os.path.join(fire.cache_dir, 'previews', 'post.png')
        if not os.path.isfile(post_path) or not os.path.isfile(raster_path):
            return

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.image import imread, imsave
        from scipy.ndimage import zoom as scipy_zoom

        post = imread(post_path)
        if post.ndim == 2:
            post = np.stack([post] * 3, axis=2)

        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        arr = ds.GetRasterBand(1).ReadAsArray()
        old_gt = ds.GetGeoTransform()
        ds = None

        ph, pw = post.shape[:2]
        ah, aw = arr.shape

        if ah != ph or aw != pw:
            aligned = False
            # Try geospatial alignment using crop geotransform.
            # Both rasters are crops of the same source, so pixel
            # sizes match — we just need to compute the offset.
            if fire.crop_bin and os.path.isfile(fire.crop_bin):
                try:
                    ds_crop = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
                    new_gt = ds_crop.GetGeoTransform()
                    new_w = ds_crop.RasterXSize
                    new_h = ds_crop.RasterYSize
                    ds_crop = None

                    if (old_gt and new_gt
                            and abs(old_gt[1] - new_gt[1]) < 1e-6
                            and abs(old_gt[5] - new_gt[5]) < 1e-6):
                        # Pixel sizes match — compute offset
                        off_x = round(
                            (old_gt[0] - new_gt[0]) / new_gt[1])
                        off_y = round(
                            (old_gt[3] - new_gt[3]) / new_gt[5])

                        # Place old raster in crop-sized array
                        arr_aligned = np.zeros(
                            (new_h, new_w), dtype=arr.dtype)
                        src_y0 = max(0, -off_y)
                        src_x0 = max(0, -off_x)
                        dst_y0 = max(0, off_y)
                        dst_x0 = max(0, off_x)
                        copy_h = min(ah - src_y0, new_h - dst_y0)
                        copy_w = min(aw - src_x0, new_w - dst_x0)
                        if copy_h > 0 and copy_w > 0:
                            arr_aligned[
                                dst_y0:dst_y0 + copy_h,
                                dst_x0:dst_x0 + copy_w,
                            ] = arr[
                                src_y0:src_y0 + copy_h,
                                src_x0:src_x0 + copy_w,
                            ]

                        # Scale to match preview PNG dimensions
                        # (preview may be downsampled from crop)
                        if new_h != ph or new_w != pw:
                            arr_aligned = scipy_zoom(
                                arr_aligned.astype(np.float32),
                                (ph / new_h, pw / new_w), order=0)

                        arr = arr_aligned
                        aligned = True
                except Exception:
                    pass

            if not aligned:
                # Fallback: naive resize (same-extent rasters)
                arr = scipy_zoom(
                    arr.astype(np.float32),
                    (ph / ah, pw / aw), order=0)

        mask = arr > 0
        result = post[:, :, :3].copy()
        r, g, b = color
        result[mask, 0] = np.clip(result[mask, 0] * 0.3 + r * 0.7, 0, 1)
        result[mask, 1] = np.clip(result[mask, 1] * 0.3 + g * 0.7, 0, 1)
        result[mask, 2] = np.clip(result[mask, 2] * 0.3 + b * 0.7, 0, 1)

        out_path = os.path.join(fire.cache_dir, 'previews', f'{out_name}.png')
        imsave(out_path, np.clip(result, 0, 1))

        if (out_name not in fire.available_views
                and not out_name.startswith('serial_')):
            fire.available_views.append(out_name)
    except Exception as exc:
        sys.stderr.write(
            f'[overlay] WARNING: Failed to generate {out_name} '
            f'overlay: {exc}\n')


def _generate_result_preview(fire: 'FireInfo'):
    """Generate pixel-aligned overlay previews after mapping."""
    clf_path = os.path.join(
        fire.cache_dir,
        f'{fire.fire_numbe}_crop.bin_classified.bin')
    _overlay_mask_on_post(fire, clf_path, 'result', (0.9, 0.1, 0.0))

    # Also generate hint overlay if hint raster exists
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))


def _compute_agreement(fire: 'FireInfo',
                       clf_path: str | None = None) -> float:
    """Compute overlap % between ML classification and hint perimeter.

    When *clf_path* is None, reads the main crop's classified.bin;
    callers can pass a per-run classified.bin for serial-run agreement.
    Returns percentage (0-100) or -1 if computation fails.

    When clf and hint have different shapes (e.g. rebrush on a serial
    run whose padding differs from the current fire.hint_bin extent —
    recommended_settings sweeps can span multiple paddings), aligns
    them via GeoTransform and computes IoU over the common overlap
    rectangle. Without this, every rebrush of a cross-padding run
    collapses to agreement=-1 → Accept button disappears.
    """
    try:
        if clf_path is None:
            clf_path = os.path.join(
                fire.cache_dir,
                f'{fire.fire_numbe}_crop.bin_classified.bin')
        hint_path = fire.hint_bin
        if not clf_path or not hint_path:
            return -1.0
        if not os.path.isfile(clf_path) or not os.path.isfile(hint_path):
            return -1.0

        ds_clf = ds_hint = None
        try:
            ds_clf = gdal.Open(clf_path, gdal.GA_ReadOnly)
            ds_hint = gdal.Open(hint_path, gdal.GA_ReadOnly)
            if ds_clf is None or ds_hint is None:
                return -1.0

            clf = ds_clf.GetRasterBand(1).ReadAsArray()
            hint = ds_hint.GetRasterBand(1).ReadAsArray()
            clf_gt = ds_clf.GetGeoTransform()
            hint_gt = ds_hint.GetGeoTransform()
        finally:
            ds_clf = ds_hint = None

        if clf.shape != hint.shape:
            # Cross-extent: crops are from the same source raster so
            # pixel sizes must match — only the origin / extent differ.
            # Refuse if pixel sizes don't line up (can't align sensibly).
            if not (clf_gt and hint_gt
                    and abs(clf_gt[1] - hint_gt[1]) < 1e-6
                    and abs(clf_gt[5] - hint_gt[5]) < 1e-6):
                return -1.0
            # Offset of clf's origin expressed in hint's pixel frame.
            off_x = round((clf_gt[0] - hint_gt[0]) / hint_gt[1])
            off_y = round((clf_gt[3] - hint_gt[3]) / hint_gt[5])
            ch, cw = clf.shape
            hh, hw = hint.shape
            # Intersection rectangle in hint's pixel coordinates.
            com_x0 = max(0, off_x)
            com_y0 = max(0, off_y)
            com_x1 = min(hw, off_x + cw)
            com_y1 = min(hh, off_y + ch)
            if com_x1 <= com_x0 or com_y1 <= com_y0:
                return -1.0
            hint = hint[com_y0:com_y1, com_x0:com_x1]
            clf = clf[
                com_y0 - off_y:com_y1 - off_y,
                com_x0 - off_x:com_x1 - off_x,
            ]
            if clf.shape != hint.shape:
                return -1.0

        clf_mask = clf > 0
        hint_mask = hint > 0
        union = np.sum(clf_mask | hint_mask)
        if union == 0:
            return 0.0
        intersection = np.sum(clf_mask & hint_mask)
        return round(float(intersection / union) * 100, 1)
    except Exception as exc:
        sys.stderr.write(
            f'[agreement] WARNING: Failed to compute agreement: {exc}\n')
        return -1.0


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


_batch_thread = None


def _batch_map_worker(fire_numbes: list[str],
                      session_hash: str | None = None):
    """Process fires sequentially, delegating each to ``_serial_map_worker``.

    This mirrors what the fire page's "Map Fire with settings" button
    does: a full N recommended settings × K replicates sweep per fire,
    populating ``fire.serial_results`` so the gallery is visible when
    the user opens the fire later. Replaces the previous single-shot
    path (recommended[0] only, no gallery), which made batch-mapped
    fires indistinguishable from failures because the gallery stayed
    empty.

    Threading contract:
      * ``_serial_map_worker`` acquires ``_gpu_lock`` per run internally,
        so this function must NOT wrap the call in ``_gpu_lock`` (would
        deadlock).
      * ``_serial_map_worker`` manages ``state.current_job`` per run.
      * Between-fires cancel: checked via ``_batch_cancel``. In-flight
        cancel is driven by ``handle_api_batch_cancel`` setting the
        running fire's ``serial_canceled`` flag.
    """
    import traceback

    with state.lock:
        state.batch_status = {
            'running': True,
            'total': len(fire_numbes),
            'completed': 0,
            'current_fire': '',
            'errors': [],
        }

    sys.stderr.write(
        f'[batch] Starting batch: {len(fire_numbes)} fire(s)\n')
    sys.stderr.flush()

    k_runs = max(1, min(10, int(state.k_runs_per_setting)))
    k_jitter = max(0, int(state.k_jitter))

    for fire_numbe in fire_numbes:
        if _batch_cancel.is_set():
            sys.stderr.write('[batch] Cancelled by user.\n')
            sys.stderr.flush()
            break

        fire = state.fires.get(fire_numbe)
        if not fire or fire.status in (
                FireStatus.ACCEPTED, FireStatus.MAPPING):
            with state.lock:
                state.batch_status['completed'] += 1
            continue

        settings = _get_recommended_settings(fire)
        if not settings:
            _set_fire_status(
                fire, FireStatus.ERROR,
                'No recommended settings configured')
            with state.lock:
                state.batch_status['errors'].append(fire_numbe)
                state.batch_status['completed'] += 1
            sys.stderr.write(
                f'[batch] [{fire_numbe}] SKIPPED: no recommended '
                f'settings.\n')
            sys.stderr.flush()
            continue

        with state.lock:
            state.batch_status['current_fire'] = fire_numbe
            # Prime the same state the /serial_map handler sets before
            # spawning its thread. _serial_map_worker re-initialises
            # some of these (idempotent), but setting MAPPING here
            # avoids a transient window where the fire list shows a
            # stale status.
            fire.serial_prev_status = fire.status
            fire.status = FireStatus.MAPPING
            fire.serial_results = []
            fire.serial_settings = [_clone_setting(s) for s in settings]
            fire.serial_canceled = False
            fire.console_log.clear()

        sys.stderr.write(
            f'[batch] [{fire_numbe}] Starting sweep: '
            f'{len(settings)} setting(s) × {k_runs} run(s)\n')
        sys.stderr.flush()

        try:
            _serial_map_worker(fire_numbe, settings, k_runs, k_jitter,
                                session_hash=session_hash)
        except Exception as exc:
            _set_fire_status(fire, FireStatus.ERROR, str(exc))
            with state.lock:
                state.current_job = None
            sys.stderr.write(
                f'[batch] [{fire_numbe}] EXCEPTION:\n'
                f'{traceback.format_exc()}\n')
            sys.stderr.flush()

        fire = state.fires.get(fire_numbe)
        if fire and fire.status == FireStatus.ERROR:
            with state.lock:
                state.batch_status['errors'].append(fire_numbe)
            sys.stderr.write(
                f'[batch] [{fire_numbe}] FAILED '
                f'({fire.error_msg or "see fire console"})\n')
        elif fire and fire.status == FireStatus.MAPPED:
            sys.stderr.write(
                f'[batch] [{fire_numbe}] MAPPED '
                f'(agreement={fire.agreement_pct}%, '
                f'{len(fire.serial_results)} run(s) in gallery)\n')
        sys.stderr.flush()

        with state.lock:
            state.batch_status['completed'] += 1

    with state.lock:
        state.batch_status['running'] = False
        state.batch_status['current_fire'] = ''
        total = int(state.batch_status.get('total', 0))
        completed = int(state.batch_status.get('completed', 0))
        n_errors = len(state.batch_status.get('errors', []))
    sys.stderr.write(
        f'[batch] Complete: {completed}/{total} fires, '
        f'{n_errors} error(s)\n')
    sys.stderr.flush()
    _push_notification(
        session_hash,
        'warning' if n_errors else 'success',
        'Batch mapping complete',
        f'{completed - n_errors}/{total} fires mapped successfully'
        + (f' ({n_errors} error(s))' if n_errors else '') + '.',
        action={'url': '/', 'label': 'Open fire list'})


def _jitter_hdbscan(base: int, run_idx: int, step: int) -> int:
    """Return a jittered hdbscan_min_samples for replicate `run_idx`.

    Fan-out pattern: base, base+step, base-step, base+2*step, base-2*step, ...
    Guaranteed >= 1. step=0 disables jitter (returns base).
    """
    if step <= 0 or run_idx == 0:
        return max(1, int(base))
    level = (run_idx + 1) // 2
    sign = 1 if run_idx % 2 == 1 else -1
    return max(1, int(base) + sign * level * int(step))


def _serial_map_worker(fire_numbe: str, settings: list[dict],
                        k_runs: int, k_jitter: int,
                        session_hash: str | None = None):
    """Run N settings × K HDBSCAN replicates for one fire.

    For each setting, the expensive deterministic part (t-SNE + RF) runs
    once on its first replicate and is cached in a per-setting .npz.
    Replicates 2..K load the cached state and only re-run HDBSCAN with
    a jittered hdbscan_min_samples value (fan-out pattern).
    """
    import traceback

    fire = state.fires[fire_numbe]
    # A fresh sweep discards anything left from a previously cancelled
    # run — both the in-memory gallery and the serial_* files on disk
    # that back it. Without the disk cleanup, run_ids from the prior
    # sweep that aren't reused by this one (e.g. prior went to 5, new
    # only reaches 3) would leak as orphan files in cache_dir.
    if fire.cache_dir and os.path.isdir(fire.cache_dir):
        prefix = f'{fire_numbe}_serial_'
        for f in os.listdir(fire.cache_dir):
            if f.startswith(prefix):
                try:
                    os.remove(os.path.join(fire.cache_dir, f))
                except OSError:
                    pass
        prev_dir = os.path.join(fire.cache_dir, 'previews')
        if os.path.isdir(prev_dir):
            for f in os.listdir(prev_dir):
                if f.startswith('serial_'):
                    try:
                        os.remove(os.path.join(prev_dir, f))
                    except OSError:
                        pass
    fire.serial_results = []
    fire.serial_settings = [_clone_setting(s) for s in settings]
    fire.serial_canceled = False
    fire.serial_accept_promoted = False
    # Drop any Event left behind by a prior accept — the cleanup
    # below would otherwise wait on it and stall the worker.
    fire.serial_accept_event = None
    fire.console_log.clear()
    n_settings = len(settings)
    k_runs = max(1, int(k_runs))
    k_jitter = max(0, int(k_jitter))
    n_total = n_settings * k_runs

    # Progress tracker — wired to _stream_subprocess in each replicate.
    # run_id advances as the sweep progresses; stage transitions are
    # detected from CLI stdout.
    tracker = _ProgressTracker(fire, total_runs=n_total, run_id=1,
                               pipeline='full')

    sys.stderr.write(
        f'[serial] Starting {n_settings} setting(s) × {k_runs} run(s) '
        f'for {fire_numbe}\n')
    sys.stderr.flush()

    # Snapshot previously accepted results as Run 0
    if fire.previously_accepted:
        try:
            old_agreement = fire.agreement_pct
            old_ml_area = fire.ml_area_ha
            old_params = dict(fire.last_params) if fire.last_params else {}

            # Search both cache_dir and canonical output dir
            canon_dir = os.path.join(state.output_root, fire_numbe)
            search_dirs = [fire.cache_dir]
            if os.path.isdir(canon_dir):
                search_dirs.append(canon_dir)

            # Find existing classified raster
            old_clf = None
            for _dir in search_dirs:
                for _pat in (f'{fire_numbe}_crop.bin_classified.bin',
                             f'{fire_numbe}_crop_classified.bin',
                             f'{fire_numbe}_classified.bin'):
                    _cand = os.path.join(_dir, _pat)
                    if os.path.isfile(_cand):
                        old_clf = _cand
                        break
                if old_clf:
                    break
            if old_clf is None:
                for _dir in search_dirs:
                    for _cand in glob.glob(os.path.join(
                            _dir, '*classified*.bin')):
                        if 'serial_' not in os.path.basename(_cand):
                            old_clf = _cand
                            break
                    if old_clf:
                        break

            # Copy classified raster as serial_0
            s0_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_0_classified.bin')
            if old_clf and os.path.isfile(old_clf):
                shutil.copy2(old_clf, s0_clf)
                old_hdr = os.path.splitext(old_clf)[0] + '.hdr'
                if not os.path.isfile(old_hdr):
                    old_hdr = old_clf + '.hdr'
                if os.path.isfile(old_hdr):
                    shutil.copy2(
                        old_hdr,
                        os.path.splitext(s0_clf)[0] + '.hdr')

            # Copy comparison PNG as serial_0
            s0_comp = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_0.png')
            for _dir in search_dirs:
                old_comp = os.path.join(
                    _dir, f'{fire_numbe}_comparison.png')
                if os.path.isfile(old_comp):
                    shutil.copy2(old_comp, s0_comp)
                    break

            # Copy brush comparison as serial_0
            s0_brush = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_0_brush.png')
            for _dir in search_dirs:
                old_brush = os.path.join(
                    _dir, f'{fire_numbe}_brush_comparison.png')
                if os.path.isfile(old_brush):
                    shutil.copy2(old_brush, s0_brush)
                    break

            # Generate overlay for Run 0
            if os.path.isfile(s0_clf):
                _overlay_mask_on_post(
                    fire, s0_clf, 'serial_0', (0.9, 0.1, 0.0))

            fire.serial_results.append({
                'run_id': 0,
                'params': old_params,
                'agreement_pct': old_agreement,
                'ml_area_ha': old_ml_area,
                'comparison': s0_comp,
                'classified': s0_clf,
                'is_previous': True,
            })
            fire.console_log.append(
                '=== Run 0: Previously accepted result ===')
            fire.console_log.append(
                f'  Agreement: {old_agreement}%'
                f', ML area: {old_ml_area} ha')
        except Exception:
            sys.stderr.write(
                f'[serial] Warning: could not snapshot previous '
                f'result for {fire_numbe}\n')
            sys.stderr.flush()

    # Global monotonic counter for run_id (run 0 = previous; 1..N*K = new)
    run_id = 0

    for setting_idx, setting in enumerate(settings):
        if fire.serial_canceled:
            break
        setting_label = str(setting.get('label', f'setting_{setting_idx}'))
        base_params = dict(setting.get('params', {}))
        try:
            base_hms = int(base_params.get('hdbscan_min_samples', 20))
        except (TypeError, ValueError):
            base_hms = 20
        try:
            padding = float(base_params.get('padding', state.padding))
        except (TypeError, ValueError):
            padding = float(state.padding)
        # Per-setting t-SNE+RF cache — each setting has its own .npz so
        # different padding / sample_rate / embed_bands / tsne_* don't
        # collide.
        state_file = os.path.join(
            fire.cache_dir or '',
            f'{fire_numbe}_serial_state_s{setting_idx}.npz')

        fire.console_log.append(
            f'=== Setting {setting_idx + 1}/{n_settings}: '
            f'{setting_label} ===')
        full_keys = []
        for k in ('padding', 'embed_bands', 'tsne_perplexity',
                  'sample_rate', 'hdbscan_min_samples'):
            v = base_params.get(k)
            if v is not None and v != '':
                full_keys.append(f'{k}={v}')
        if full_keys:
            fire.console_log.append(f'  {", ".join(full_keys)}')

        setting_stopped = False

        for replicate in range(k_runs):
            if setting_stopped or fire.serial_canceled:
                break
            run_id += 1
            params = dict(base_params)
            jittered = _jitter_hdbscan(base_hms, replicate, k_jitter)
            params['hdbscan_min_samples'] = jittered

            fire.console_log.append(
                f'-- Run {run_id}/{n_total} '
                f'(setting {setting_idx + 1}, replicate {replicate + 1}'
                f'/{k_runs}, hdbscan_min_samples={jittered}) --')

            try:
                with _gpu_lock:
                    # Cancel window (A): check serial_canceled BEFORE
                    # calling _prepare_fire_sync. Without this, a
                    # cancel/accept arriving between settings with
                    # different paddings would still run a full prep
                    # (briefly flipping status to PREPARING) before
                    # the worker noticed the cancel — user-visible as
                    # a "back to preparing" flicker after Accept.
                    if fire.serial_canceled:
                        break
                    # Re-prepare if padding changed or cache files missing.
                    # Per-setting padding: re-prep each time padding differs
                    # from fire.padding_used. Replicates within the same
                    # setting share the prep.
                    if replicate == 0:
                        needs_prepare = (
                            fire.padding_used != padding
                            or not fire.cache_dir
                            or not os.path.isdir(fire.cache_dir)
                            or not fire.crop_bin
                            or not os.path.isfile(fire.crop_bin)
                            or not fire.hint_bin
                            or not os.path.isfile(fire.hint_bin)
                        )
                        if needs_prepare:
                            fire.console_log.append(
                                f'  Re-preparing (padding '
                                f'{fire.padding_used} → {padding}) ...')
                            _prepare_fire_sync(fire_numbe, padding)
                            fire = state.fires[fire_numbe]
                            if fire.status == FireStatus.ERROR:
                                fire.serial_results.append({
                                    'run_id': run_id,
                                    'setting_idx': setting_idx,
                                    'run_idx': replicate,
                                    'setting_label': setting_label,
                                    'params': params,
                                    'agreement_pct': -1,
                                    'error': fire.error_msg,
                                })
                                # Cache_dir was wiped — recompute state_file
                                # path for next setting.
                                setting_stopped = True
                                continue
                            # cache_dir path may have been rebuilt;
                            # update state_file for this setting.
                            state_file = os.path.join(
                                fire.cache_dir,
                                f'{fire_numbe}_serial_state_s'
                                f'{setting_idx}.npz')

                    # If the user accepted a completed run while we
                    # were waiting on the GPU lock, skip this replicate
                    # so we don't overwrite the accepted status / cache.
                    if fire.serial_canceled:
                        break
                    with state.lock:
                        fire.status = FireStatus.MAPPING
                        fire.last_params = params
                        state.current_job = {
                            'fire_numbe': (
                                f'{fire_numbe} (run {run_id}/{n_total})'),
                            'client_ip': 'serial',
                            'started_at':
                                datetime.datetime.now().isoformat(
                                    timespec='seconds'),
                        }

                    # Replicate 0: full pipeline + save per-setting state.
                    # Replicates 1..K-1: load cached state (HDBSCAN only).
                    is_first_of_setting = (replicate == 0)
                    cmd = _build_mapping_cmd(
                        fire, params,
                        save_state=(
                            state_file if is_first_of_setting else None),
                        load_state=(
                            None if is_first_of_setting else state_file),
                    )
                    if not is_first_of_setting:
                        fire.console_log.append(
                            '  (resuming from cached t-SNE + RF)')

                    def _on_line(text, _fn=fire_numbe, _rid=run_id,
                                 _fire=fire):
                        _fire.console_log.append(text)
                        sys.stderr.write(
                            f'[serial] [{_fn}#{_rid}] {text}\n')

                    tracker.mark_run_start(
                        run_id,
                        pipeline=('full' if is_first_of_setting
                                  else 'resume'))
                    rc, killed = _stream_subprocess(
                        cmd, state.project_root, _on_line,
                        tracker=tracker, fire_numbe=fire_numbe)
                    tracker.mark_run_end()
                    with state.lock:
                        state.current_job = None
                    sys.stderr.flush()

                    if killed:
                        fire.console_log.append(
                            f'[watchdog] killed after '
                            f'{_SUBPROCESS_SILENCE_TIMEOUT}s of silence')

                    # Subprocess was SIGTERMed by an accept/cancel
                    # handler (rc is non-zero, cancel flag is set, and
                    # the watchdog didn't fire). Don't log as FAILED
                    # or append a phantom gallery card — the cancel
                    # cleanup below will wipe serial_results anyway,
                    # and a "Run N FAILED" line is misleading since
                    # the user's accept was the cause.
                    if (rc != 0 and not killed
                            and fire.serial_canceled):
                        fire.console_log.append(
                            f'Run {run_id} terminated by cancel/accept.')
                        break

                    if rc == 0:
                        agr = _compute_agreement(fire)

                        src_comp = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_comparison.png')
                        serial_comp = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_serial_{run_id}.png')
                        if os.path.isfile(src_comp):
                            shutil.copy2(src_comp, serial_comp)

                        src_brush = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_brush_comparison.png')
                        serial_brush = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_serial_{run_id}_brush.png')
                        if os.path.isfile(src_brush):
                            shutil.copy2(src_brush, serial_brush)

                        src_clf = None
                        for _pat in (
                                f'{fire_numbe}_crop.bin_classified.bin',
                                f'{fire_numbe}_crop_classified.bin',
                                f'{fire_numbe}_classified.bin'):
                            _cand = os.path.join(fire.cache_dir, _pat)
                            if os.path.isfile(_cand):
                                src_clf = _cand
                                break
                        if src_clf is None:
                            for _cand in glob.glob(os.path.join(
                                    fire.cache_dir, '*classified*.bin')):
                                # Skip the pre-brush backup siblings —
                                # we want the canonical (brushed) copy.
                                if _cand.endswith('_raw.bin'):
                                    continue
                                src_clf = _cand
                                break

                        serial_clf = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_serial_{run_id}_classified.bin')
                        if src_clf and os.path.isfile(src_clf):
                            shutil.copy2(src_clf, serial_clf)
                            src_hdr = (
                                os.path.splitext(src_clf)[0] + '.hdr')
                            if not os.path.isfile(src_hdr):
                                src_hdr = src_clf + '.hdr'
                            if os.path.isfile(src_hdr):
                                shutil.copy2(
                                    src_hdr,
                                    os.path.splitext(serial_clf)[0]
                                    + '.hdr')
                            # Also preserve the pre-brush backup so a
                            # per-run rebrush can start from the raw.
                            src_raw = (os.path.splitext(src_clf)[0]
                                       + '_raw.bin')
                            if os.path.isfile(src_raw):
                                serial_raw = (
                                    os.path.splitext(serial_clf)[0]
                                    + '_raw.bin')
                                shutil.copy2(src_raw, serial_raw)
                                src_raw_hdr = (
                                    os.path.splitext(src_raw)[0] + '.hdr')
                                if not os.path.isfile(src_raw_hdr):
                                    src_raw_hdr = src_raw + '.hdr'
                                if os.path.isfile(src_raw_hdr):
                                    shutil.copy2(
                                        src_raw_hdr,
                                        os.path.splitext(serial_raw)[0]
                                        + '.hdr')

                        clf_for_run = serial_clf if os.path.isfile(
                            serial_clf) else src_clf

                        run_ml_area = _compute_ml_area(
                            fire, clf_for_run) if clf_for_run else -1.0

                        if clf_for_run:
                            _overlay_mask_on_post(
                                fire, clf_for_run, f'serial_{run_id}',
                                (0.9, 0.1, 0.0))

                        serial_overlay = os.path.join(
                            fire.cache_dir, 'previews',
                            f'serial_{run_id}.png')
                        result_overlay = os.path.join(
                            fire.cache_dir, 'previews', 'result.png')
                        if os.path.isfile(serial_overlay):
                            shutil.copy2(serial_overlay, result_overlay)
                            if 'result' not in fire.available_views:
                                fire.available_views.append('result')
                        else:
                            _generate_result_preview(fire)

                        fire.serial_results.append({
                            'run_id': run_id,
                            'setting_idx': setting_idx,
                            'run_idx': replicate,
                            'setting_label': setting_label,
                            'params': params,
                            'agreement_pct': agr,
                            'ml_area_ha': run_ml_area,
                            'comparison': serial_comp,
                            'classified': serial_clf,
                        })
                        fire.console_log.append(
                            f'Run {run_id} complete '
                            f'(agreement={agr}%, ML area={run_ml_area} ha)')
                    else:
                        fire.serial_results.append({
                            'run_id': run_id,
                            'setting_idx': setting_idx,
                            'run_idx': replicate,
                            'setting_label': setting_label,
                            'params': params,
                            'agreement_pct': -1,
                            'error': f'Exited with code {rc}',
                        })
                        fire.console_log.append(
                            f'Run {run_id} FAILED (exit code {rc})')
                        # Abandon this setting if the full pipeline
                        # failed — no cached state to resume from.
                        if is_first_of_setting:
                            fire.console_log.append(
                                '  Skipping remaining replicates for '
                                'this setting.')
                            setting_stopped = True

            except Exception as exc:
                with state.lock:
                    state.current_job = None
                fire.serial_results.append({
                    'run_id': run_id,
                    'setting_idx': setting_idx,
                    'run_idx': replicate,
                    'setting_label': setting_label,
                    'params': params,
                    'agreement_pct': -1,
                    'error': str(exc),
                })
                sys.stderr.write(
                    f'[serial] [{fire_numbe}#{run_id}] EXCEPTION:\n'
                    f'{traceback.format_exc()}\n')
                sys.stderr.flush()
                if replicate == 0:
                    setting_stopped = True

    # Cancel path. Two flavors:
    #
    # (A) Accept-initiated cancel (serial_accept_promoted=True): the
    #     accept handler already copied the chosen run into the main
    #     slot and flipped status to ACCEPTED. Gallery should clear —
    #     drop every serial_* file and empty serial_results.
    #
    # (B) User-initiated cancel (serial_accept_promoted=False): the
    #     user hit "stop, that's enough". Keep the gallery. If at
    #     least one run succeeded, promote the best into the main slot
    #     and land on MAPPED so the fire is usable. If nothing
    #     succeeded, revert to the pre-sweep status.
    #
    # _gpu_lock is held across the entire cleanup so concurrent accept
    # handlers cannot race our file writes / deletes on the same
    # cache_dir.
    if fire.serial_canceled:
        # If an accept handler is mid-run, wait for it to finish
        # writing the canonical output before we enter the cleanup
        # block. Without this, the worker and the accept handler race
        # for _gpu_lock — if the worker wins, it deletes the
        # per-run serial_* files before accept has copied them into
        # the main slot, and the canonical dir ends up empty. The
        # event is set by handle_api_serial_accept's finally, so it
        # unblocks on both success and exception. Timeout is a
        # generous 120s to survive disk-heavy accepts; if it expires,
        # fall through and let the existing _gpu_lock contention
        # arbitrate (previous behaviour).
        accept_event = fire.serial_accept_event
        if (fire.serial_accept_promoted
                and accept_event is not None):
            accept_event.wait(timeout=120)
        with _gpu_lock:
            accept_promoted = fire.serial_accept_promoted

            if accept_promoted:
                # (A) — full gallery wipe.
                for sr in list(fire.serial_results):
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

            # Free per-setting T-SNE+RF caches regardless of flavor.
            # They only accelerate intra-sweep replicates; the gallery
            # does not need them and they are the single biggest disk
            # win (30-100 MB each).
            for npz in glob.glob(os.path.join(
                    fire.cache_dir,
                    f'{fire_numbe}_serial_state_s*.npz')):
                try:
                    os.remove(npz)
                except OSError:
                    pass

            if accept_promoted:
                with state.lock:
                    # Accept handler set serial_prev_status = ACCEPTED;
                    # trust it.
                    revert = (fire.serial_prev_status
                              or FireStatus.ACCEPTED)
                    fire.status = revert
                    fire.serial_results = []
                    fire.serial_canceled = False
                    fire.serial_prev_status = None
                    fire.serial_accept_promoted = False
                    fire.progress = {}
                    state.current_job = None
                fire.console_log.append(
                    f'Serial mapping cancelled by accept — status set '
                    f'to {revert.value}.')
                _save_fire_state()
                sys.stderr.write(
                    f'[serial] {fire_numbe} accept-cancel → '
                    f'{revert.value}\n')
                sys.stderr.flush()
                return

            # (B) — preserve gallery; promote best successful run.
            successful = [r for r in fire.serial_results
                          if r.get('agreement_pct', -1) >= 0
                          and not r.get('is_previous')
                          and r.get('classified')
                          and os.path.isfile(r.get('classified', ''))]
            if successful:
                best = max(successful,
                           key=lambda r: r['agreement_pct'])
                try:
                    best_comp = best.get('comparison', '')
                    if best_comp and os.path.isfile(best_comp):
                        main_comp = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_comparison.png')
                        shutil.copy2(best_comp, main_comp)
                        fire.last_comparison = main_comp
                    best_clf = best.get('classified', '')
                    if best_clf and os.path.isfile(best_clf):
                        main_clf = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_crop.bin_classified.bin')
                        shutil.copy2(best_clf, main_clf)
                        best_hdr = (
                            os.path.splitext(best_clf)[0] + '.hdr')
                        if not os.path.isfile(best_hdr):
                            best_hdr = best_clf + '.hdr'
                        if os.path.isfile(best_hdr):
                            shutil.copy2(
                                best_hdr,
                                os.path.splitext(main_clf)[0] + '.hdr')
                        _generate_result_preview(fire)
                except Exception:
                    pass
                with state.lock:
                    fire.agreement_pct = best['agreement_pct']
                    fire.ml_area_ha = best.get('ml_area_ha', -1.0)
                    fire.last_params = best['params']
                    # If the fire was already ACCEPTED before the sweep
                    # started, keep that — the canonical dir is still on
                    # disk and the user didn't explicitly un-accept.
                    # Otherwise land on MAPPED so the fire is usable.
                    prev = fire.serial_prev_status
                    if prev == FireStatus.ACCEPTED:
                        fire.status = FireStatus.ACCEPTED
                    else:
                        fire.status = FireStatus.MAPPED
                    fire.serial_canceled = False
                    fire.serial_prev_status = None
                    fire.serial_accept_promoted = False
                    fire.progress = {}
                    state.current_job = None
                fire.console_log.append(
                    f'Serial mapping cancelled — kept gallery '
                    f'({len(successful)} run(s)); best: run '
                    f'{best["run_id"]} '
                    f'(agreement={best["agreement_pct"]}%). '
                    f'Click Map Fire again to discard and re-sweep.')
                _save_fire_state()
                _push_notification(
                    session_hash, 'info',
                    f'Mapping cancelled — {fire_numbe}',
                    f'Kept {len(successful)} run(s); best agreement '
                    f'{best["agreement_pct"]}%.',
                    fire=fire_numbe,
                    action={'url': f'/fire/{fire_numbe}',
                            'label': 'Open fire'})
                sys.stderr.write(
                    f'[serial] {fire_numbe} user-cancel → '
                    f'{fire.status.value} ({len(successful)} kept)\n')
                sys.stderr.flush()
                return

            # No successful runs — nothing worth keeping. Revert to
            # pre-sweep status and clear serial_results so the UI
            # doesn't show a gallery of error cards for a fire that
            # now has no data.
            with state.lock:
                revert = fire.serial_prev_status or FireStatus.PENDING
                fire.status = revert
                fire.serial_results = []
                fire.serial_canceled = False
                fire.serial_prev_status = None
                fire.serial_accept_promoted = False
                fire.progress = {}
                state.current_job = None
            fire.console_log.append(
                f'Serial mapping cancelled — no successful runs, '
                f'status restored to {revert.value}.')
            _save_fire_state()
            _push_notification(
                session_hash, 'warning',
                f'Mapping cancelled — {fire_numbe}',
                'No successful runs. Status restored.',
                fire=fire_numbe)
        sys.stderr.write(
            f'[serial] {fire_numbe} user-cancel (empty) → '
            f'{revert.value}\n')
        sys.stderr.flush()
        return

    # Pick best result as the "current" mapping (exclude Run 0 / previous)
    successful = [r for r in fire.serial_results
                  if r.get('agreement_pct', -1) >= 0
                  and not r.get('is_previous')]
    if successful:
        best = max(successful, key=lambda r: r['agreement_pct'])
        fire.agreement_pct = best['agreement_pct']
        fire.ml_area_ha = best.get('ml_area_ha', -1.0)
        fire.last_params = best['params']

        # Copy best result as the "main" comparison
        best_comp = best.get('comparison', '')
        if best_comp and os.path.isfile(best_comp):
            main_comp = os.path.join(
                fire.cache_dir, f'{fire_numbe}_comparison.png')
            shutil.copy2(best_comp, main_comp)
            fire.last_comparison = main_comp

        best_clf = best.get('classified', '')
        if best_clf and os.path.isfile(best_clf):
            main_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_crop.bin_classified.bin')
            shutil.copy2(best_clf, main_clf)
            best_hdr = (
                os.path.splitext(best_clf)[0] + '.hdr')
            if not os.path.isfile(best_hdr):
                best_hdr = best_clf + '.hdr'
            if os.path.isfile(best_hdr):
                shutil.copy2(
                    best_hdr,
                    os.path.splitext(main_clf)[0] + '.hdr')
            _generate_result_preview(fire)

        fire.status = FireStatus.MAPPED
        fire.console_log.append(
            f'Serial mapping complete. Best: run {best["run_id"]} '
            f'(agreement={best["agreement_pct"]}%)')
    else:
        _set_fire_status(
            fire, FireStatus.ERROR, 'All serial runs failed')
        fire.console_log.append('All serial runs failed.')

    # Free per-setting T-SNE+RF caches. These accelerate intra-sweep
    # replicates but are dead weight after the sweep finishes (the
    # user will either accept a result or leave the gallery — neither
    # needs the .npz again). Single largest disk win per fire.
    for npz in glob.glob(os.path.join(
            fire.cache_dir,
            f'{fire_numbe}_serial_state_s*.npz')):
        try:
            os.remove(npz)
        except OSError:
            pass

    # Clear live progress — the sweep is done.
    with state.lock:
        fire.progress = {}
    _save_fire_state()
    _save_stage_timings()
    # Notify the initiating session.
    if successful:
        best_agr = max(r['agreement_pct'] for r in successful)
        _push_notification(
            session_hash, 'success',
            f'Mapping complete — {fire_numbe}',
            f'{len(successful)}/{n_total} run(s) succeeded. '
            f'Best agreement: {best_agr}%.',
            fire=fire_numbe,
            action={'url': f'/fire/{fire_numbe}', 'label': 'Open fire'})
    else:
        _push_notification(
            session_hash, 'error',
            f'Mapping failed — {fire_numbe}',
            'All runs failed. Check the fire console for details.',
            fire=fire_numbe,
            action={'url': f'/fire/{fire_numbe}', 'label': 'Open fire'})
    sys.stderr.write(
        f'[serial] {fire_numbe} done: {len(successful)}/{n_total} '
        f'successful\n')
    sys.stderr.flush()


def _save_ip_list():
    """Persist approved, blocked, and pending IPs to disk."""
    if not state.ip_file:
        return
    try:
        with state.lock:
            data = {
                'approved': dict(state.approved_ips),
                'blocked': dict(state.blocked_ips),
                'pending': dict(state.pending_ips),
            }
        _atomic_yaml_dump(state.ip_file, data)
    except Exception as exc:
        sys.stderr.write(
            f'[save] WARNING: Failed to save IP list: {exc}\n')


# =========================================================================
# Simple template rendering  (replaces Jinja2)
# =========================================================================

def _html_escape(s: str) -> str:
    """Escape HTML special characters."""
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;')
             .replace("'", '&#39;'))


def render_template(name: str, context: dict) -> str:
    """Replace ``{{ key }}`` placeholders in a template file.

    Values are HTML-escaped by default.  Use ``{{{ key }}}`` for raw
    (unescaped) insertion when the value is known-safe.
    """
    path = os.path.join(_HERE, 'templates', name)
    with open(path) as f:
        html = f.read()
    for key, val in context.items():
        html = html.replace('{{{ ' + key + ' }}}', str(val))   # raw
        html = html.replace('{{ ' + key + ' }}', _html_escape(str(val)))
    return html


# =========================================================================
# Preparation — runs synchronously (called from request thread)
# =========================================================================

def _prepare_fire_sync(fire_numbe: str, padding: float | None = None):
    """Prepare a fire for mapping: crop, VIIRS accumulate, hint, previews."""
    from batch_fire_mapping.run_fire_mapping import (
        raster_native_extent, crop_raster, rasterize_polygon,
    )
    from shapely.geometry import box as shapely_box

    fire = state.fires[fire_numbe]

    # Refuse only if another prepare is already running on this fire.
    # MAPPING is *not* a refusal condition: _serial_map_worker sets
    # status=MAPPING before calling this (see app.py:1090, 1384) and
    # then asks us to re-prep when padding changes or the crop/hint
    # files are missing. Treating MAPPING as "busy" here was the root
    # cause of batch-mapping failures on never-opened fires — the
    # guard no-op'd, crop_bin/hint_bin stayed empty, and the CLI
    # subprocess crashed on empty positional args.
    if fire.status == FireStatus.PREPARING:
        fire.error_msg = 'Cannot prepare: fire is currently preparing'
        return

    fire.status = FireStatus.PREPARING
    fire.error_msg = ""

    pad = padding if padding is not None else state.padding

    try:
        row = state.gdf[
            state.gdf['FIRE_NUMBE'].astype(str) == fire_numbe
        ].iloc[0]
    except (IndexError, KeyError):
        _set_fire_status(fire, FireStatus.ERROR,
                         f"Fire {fire_numbe} not found in shapefile")
        return

    # -- Parse FIRE_DATE --
    raw = row.get('FIRE_DATE', '')
    try:
        if hasattr(raw, 'date'):
            fire_date = datetime.datetime(raw.year, raw.month, raw.day)
        else:
            fire_date = datetime.datetime.strptime(
                str(raw).split()[0], '%Y-%m-%d')
    except (ValueError, AttributeError):
        _set_fire_status(fire, FireStatus.ERROR,
                         f"Cannot parse FIRE_DATE: {raw!r}")
        return

    acc_start = fire_date - datetime.timedelta(days=5)

    # -- Clip polygon to raster, compute crop bounds --
    gt = state.raster_gt
    W, H = state.raster_W, state.raster_H
    rx1, ry1, rx2, ry2 = raster_native_extent(gt, W, H)
    raster_box = shapely_box(rx1, ry1, rx2, ry2)

    clipped = row.geometry.intersection(raster_box)
    if clipped.is_empty:
        _set_fire_status(fire, FireStatus.ERROR,
                         "Fire polygon does not overlap the raster")
        return

    bounds = clipped.bounds  # (minx, miny, maxx, maxy)
    px_lo = int((bounds[0] - gt[0]) / gt[1])
    px_hi = int((bounds[2] - gt[0]) / gt[1])
    py_lo = int((bounds[3] - gt[3]) / gt[5])  # maxy -> top row
    py_hi = int((bounds[1] - gt[3]) / gt[5])  # miny -> bottom row

    fire_max_dim = max(px_hi - px_lo, py_hi - py_lo)
    p = max(1, int(round(pad * fire_max_dim)))
    px_lo = max(0, px_lo - p)
    px_hi = min(W - 1, px_hi + p)
    py_lo = max(0, py_lo - p)
    py_hi = min(H - 1, py_hi + p)

    if px_lo >= px_hi or py_lo >= py_hi:
        _set_fire_status(fire, FireStatus.ERROR,
                         "Crop box has zero area after clipping")
        return

    crop_xmin = gt[0] + px_lo * gt[1]
    crop_xmax = gt[0] + px_hi * gt[1]
    crop_ymax = gt[3] + py_lo * gt[5]
    crop_ymin = gt[3] + py_hi * gt[5]

    crop_w = px_hi - px_lo
    crop_h = py_hi - py_lo
    # Capture the old padding BEFORE mutating fire.padding_used so the
    # cache-wipe comparison below can detect a real change. Previously the
    # assignment happened first and the check was always False, so stale
    # crops survived a padding change and were reused with the new label.
    old_pad = fire.padding_used
    fire.crop_w = crop_w
    fire.crop_h = crop_h
    fire.padding_used = pad

    sample_size = int(round(crop_w * crop_h * state.sample_rate))
    sample_size = max(state.min_samples, min(state.max_samples, sample_size))
    fire.sample_size = sample_size

    # -- Create / clear cache directory --
    # Only wipe when padding actually changed; preserve existing results
    cache_dir = os.path.join(state.output_root, '.web_cache', fire_numbe)
    padding_changed = (old_pad != 0
                       and old_pad != pad
                       and os.path.isdir(cache_dir))
    if padding_changed:
        # Selective wipe: keep {fire}_serial_* (the gallery's
        # classified.bin + standalone comparison PNGs) and drop
        # everything else. A recommended-settings sweep whose
        # settings span multiple padding values re-enters this
        # branch between settings; the old shutil.rmtree wiped
        # every prior setting's gallery files off disk, leaving
        # in-memory fire.serial_results entries pointing at
        # non-existent files (ghost cards with no thumbnails).
        # Serial overlay PNGs under previews/ are tied to the
        # old post.png extent, so drop those — handle_api_serial_image
        # regenerates them on demand from the surviving
        # classified.bin via _overlay_mask_on_post, which aligns
        # across crop extents using GeoTransform.
        serial_prefix = f'{fire_numbe}_serial_'
        for fname in os.listdir(cache_dir):
            full_path = os.path.join(cache_dir, fname)
            if (os.path.isfile(full_path)
                    and not fname.startswith(serial_prefix)):
                try:
                    os.remove(full_path)
                except OSError:
                    pass
        previews_dir = os.path.join(cache_dir, 'previews')
        if os.path.isdir(previews_dir):
            shutil.rmtree(previews_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    fire.cache_dir = cache_dir

    # -- Crop raster --
    crop_bin = os.path.join(cache_dir, f'{fire_numbe}_crop.bin')
    if not crop_raster(state.raster_path, crop_bin,
                       crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        _set_fire_status(fire, FireStatus.ERROR, "GDAL crop failed")
        return
    fire.crop_bin = crop_bin

    crop_gt = (crop_xmin, gt[1], gt[2], crop_ymax, gt[4], gt[5])

    # -- Rasterize traditional perimeter --
    perim_bin = os.path.join(cache_dir, f'{fire_numbe}_perimeter.bin')
    try:
        rasterize_polygon(
            state.polygon_file, fire_numbe, state.raster_crs,
            crop_bin, perim_bin, geometry=clipped,
            crop_gt=crop_gt, crop_w=crop_w, crop_h=crop_h)
    except Exception:
        perim_bin = None
    fire.perim_bin = perim_bin or ''

    # -- VIIRS accumulation --
    viirs_bin = None
    acc_end = fire_date
    plot_start = acc_start.date()
    plot_end = fire_date.date()

    if (state.perimeter_mode == 'viirs'
            and state.viirs_gdf is not None
            and not state.viirs_gdf.empty):
        from viirs.utils.accumulate import accumulate
        from viirs.utils.rasterize import rasterize_shapefile

        inside = state.viirs_gdf[
            state.viirs_gdf.geometry.within(row.geometry)]

        if not inside.empty:
            acc_end = datetime.datetime.combine(
                inside['detection_date'].max(), datetime.time.min)
            inside_window = inside[
                inside['detection_datetime'] >= acc_start]
            if not inside_window.empty:
                plot_start = inside_window['detection_date'].min()
            plot_end = acc_end.date()

        try:
            acc_paths = accumulate(
                shp_dir=state.viirs_shp_dir,
                start_str=acc_start.strftime('%Y%m%d'),
                end_str=acc_end.strftime('%Y%m%d'),
                reference_raster=crop_bin,
                output_dir=cache_dir,
                final_only=True,
                bbox=(crop_xmin, crop_ymin, crop_xmax, crop_ymax),
            )
        except Exception:
            acc_paths = []

        if acc_paths:
            try:
                viirs_bin = rasterize_shapefile(
                    shp_path=acc_paths[-1],
                    ref_image=crop_bin,
                    output_dir=cache_dir,
                    buffer_m=375.0,
                )
                if viirs_bin:
                    ds = gdal.Open(viirs_bin, gdal.GA_ReadOnly)
                    arr = ds.GetRasterBand(1).ReadAsArray()
                    ds = None
                    if np.nansum(arr) == 0:
                        viirs_bin = None
            except Exception:
                viirs_bin = None

    # -- Select hint raster --
    if state.perimeter_mode == 'traditional':
        fire.hint_bin = perim_bin or ''
        fire.perimeter_type = 'traditional'
    elif viirs_bin:
        fire.hint_bin = viirs_bin
        fire.viirs_bin = viirs_bin
        fire.perimeter_type = 'viirs'
    elif perim_bin and os.path.exists(perim_bin):
        fire.hint_bin = perim_bin
        fire.perimeter_type = 'traditional'
    else:
        _set_fire_status(fire, FireStatus.ERROR,
                         "No classification hint available")
        return

    fire.acc_start = str(plot_start)
    fire.acc_end = str(plot_end)

    # -- Generate preview images --
    views = generate_all_previews(crop_bin, cache_dir, fire_numbe)
    fire.available_views = views

    # -- Copy results from canonical dir for previously accepted fires --
    canon_dir = os.path.join(state.output_root, fire_numbe)
    if os.path.isdir(canon_dir):
        copied = []
        for fname in os.listdir(canon_dir):
            src = os.path.join(canon_dir, fname)
            dst = os.path.join(cache_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(fname)
        if copied:
            sys.stderr.write(
                f'[prepare] [{fire_numbe}] Restored {len(copied)} '
                f'file(s) from accepted dir\n')
            sys.stderr.flush()

    # -- Find classified raster (try multiple naming patterns) --
    clf_path = None
    for pattern in (f'{fire_numbe}_crop.bin_classified.bin',
                    f'{fire_numbe}_crop_classified.bin',
                    f'{fire_numbe}_classified.bin'):
        candidate = os.path.join(cache_dir, pattern)
        if os.path.isfile(candidate):
            clf_path = candidate
            break
    if clf_path is None:
        # Last resort: any *classified*.bin
        for candidate in glob.glob(
                os.path.join(cache_dir, '*classified*.bin')):
            clf_path = candidate
            break

    # -- Generate overlay previews (always try both) --
    if clf_path and os.path.isfile(clf_path):
        # Point fire at the classified raster for overlay generation
        _overlay_mask_on_post(fire, clf_path, 'result', (0.9, 0.1, 0.0))
        if 'result' not in fire.available_views:
            fire.available_views.append('result')
        sys.stderr.write(
            f'[prepare] [{fire_numbe}] Generated ML classification '
            f'overlay from {os.path.basename(clf_path)}\n')
        sys.stderr.flush()
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))

    fire.status = FireStatus.READY
    _save_fire_state()


# =========================================================================
# Accept — runs synchronously
# =========================================================================

def _ensure_brush_comparison_in_cache(fire: 'FireInfo', cache_dir: str) -> None:
    """If the cache is missing a brush comparison PNG, try to render one
    from the pre- and post-brush masks available on disk.

    Inputs resolved in cache_dir:
      - brushed mask = ``{fire}_crop.bin_classified.bin`` (canonical;
        contains the brushed mask when brush succeeded, else the raw
        classification — the same data either way).
      - raw mask    = ``{fire}_crop.bin_classified_raw.bin`` (pre-brush
        backup; only exists when brush succeeded at least once).

    When both exist, renders a full before/after figure. When only the
    canonical mask exists, renders a figure where "After" falls back to
    the raw view and the title reflects the missing brush output. When
    neither exists, silently no-ops — the canonical dir just won't have
    a brush PNG, same as before.

    Best-effort: any rendering error is logged and swallowed so accept
    never fails because of a cosmetic figure.
    """
    fire_numbe = fire.fire_numbe
    out_path = os.path.join(cache_dir, f'{fire_numbe}_brush_comparison.png')
    if os.path.isfile(out_path):
        return

    brushed_path = os.path.join(
        cache_dir, f'{fire_numbe}_crop.bin_classified.bin')
    if not os.path.isfile(brushed_path):
        return

    raw_path = os.path.join(
        cache_dir, f'{fire_numbe}_crop.bin_classified_raw.bin')
    post_png = os.path.join(cache_dir, 'previews', 'post.png')
    if not os.path.isfile(post_png):
        return

    try:
        brushed = _read_envi_mask(brushed_path)
        if os.path.isfile(raw_path):
            raw = _read_envi_mask(raw_path)
            brushed_for_fig = brushed
        else:
            # No pre-brush backup on disk — we only have one mask. Show
            # it as "Before" and flag "After" as unavailable so the
            # figure is informative rather than misleadingly claiming
            # brushing happened.
            raw = brushed
            brushed_for_fig = None

        start = getattr(fire, 'acc_start', '') or ''
        end = getattr(fire, 'acc_end', '') or ''
        title = f'Fire: {fire_numbe}  —  class_brush comparison'
        if start or end:
            title += f'\nStart: {start}   |   End: {end}'
        _render_brush_comparison_png(
            raw, brushed_for_fig, post_png, out_path, title)
    except Exception as exc:
        sys.stderr.write(
            f'[accept] WARNING: brush comparison regen for '
            f'{fire_numbe}: {exc}\n')
        sys.stderr.flush()


def _accept_fire_sync(fire_numbe: str) -> str:
    """Copy results from cache to canonical dir, write params. Returns path."""
    fire = state.fires[fire_numbe]
    cache_dir = fire.cache_dir
    # Refuse to run with no cache_dir — glob.glob(os.path.join('',
    # '*.bin')) would silently fall through to the process CWD and
    # copy unrelated files into the canonical output dir.
    if not cache_dir or not os.path.isdir(cache_dir):
        raise RuntimeError(
            f'Cannot accept {fire_numbe}: cache_dir missing or invalid '
            f'({cache_dir!r}). Re-prepare the fire and try again.')
    if not state.output_root:
        raise RuntimeError(
            f'Cannot accept {fire_numbe}: output_root not configured.')
    fire_dir = os.path.join(state.output_root, fire_numbe)

    # Register this accept as in-progress so the background cache
    # sweeper treats cache_dir as hard-pinned for the duration.
    # Without this, _cache_sweep (which uses its own lock, not
    # _gpu_lock) could rmtree cache_dir mid-copy.
    with _accept_in_progress_lock:
        _accept_in_progress.add(fire_numbe)
    try:
        if os.path.isdir(fire_dir):
            shutil.rmtree(fire_dir)
        os.makedirs(fire_dir)

        # Safety net: ensure {fire}_brush_comparison.png exists in cache
        # before the copy, regenerating from the pre/post-brush masks on
        # disk if it's missing. Guarantees the canonical dir always has a
        # brush comparison figure, even for fires mapped before
        # class_brush.exe was available (where the CLI produced a
        # "FAILED" figure that may have been cleaned up) or where the
        # serial accept path didn't supply one.
        _ensure_brush_comparison_in_cache(fire, cache_dir)

        # Only canonical/final artifacts belong in the output dir. Per-run
        # serial artifacts ({fire}_serial_{rid}*) live in .web_cache and
        # must not leak into the final result. Same for rebrush backups
        # (*_raw.bin / *_raw.hdr) which are cache-only pre-brush snapshots.
        for pattern in ('*.bin', '*.hdr', '*.png', '*.shp', '*.dbf',
                         '*.shx', '*.prj', '*.cpg'):
            for f in glob.glob(os.path.join(cache_dir, pattern)):
                basename = os.path.basename(f)
                if '_serial_' in basename:
                    continue
                if basename.endswith('_raw.bin') or basename.endswith('_raw.hdr'):
                    continue
                shutil.copy2(f, fire_dir)

        # Compute ML area from the accepted dir
        clf_bin = os.path.join(
            fire_dir, f'{fire_numbe}_crop.bin_classified.bin')
        ml_area_val = _compute_ml_area(fire, clf_bin)
        ml_area_ha = ml_area_val if ml_area_val >= 0 else None
        ml_area_m2 = (ml_area_ha * 10000.0) if ml_area_ha is not None else None
        fire.ml_area_ha = ml_area_val

        # Write params YAML
        try:
            import yaml
            params_dict = {
                'fire': {
                    'fire_numbe': fire_numbe,
                    'fire_date': fire.fire_date,
                    'fire_size_ha': fire.fire_size_ha,
                    'ml_area_ha': ml_area_ha,
                    'ml_area_m2': ml_area_m2,
                    'agreement_pct': fire.agreement_pct,
                    'notes': fire.notes or '',
                },
                'run': {
                    'timestamp': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                    'source': 'web',
                },
                'inputs': {
                    'raster': state.raster_path,
                    'perimeter_type': fire.perimeter_type,
                },
                'crop': {
                    'padding': fire.padding_used,
                    'width_px': fire.crop_w,
                    'height_px': fire.crop_h,
                    'total_px': fire.crop_w * fire.crop_h,
                },
                'sampling': {
                    'sample_rate': state.sample_rate,
                    'actual_sample_size': fire.sample_size,
                },
                'accumulation': {
                    'start_date': fire.acc_start,
                    'end_date': fire.acc_end,
                },
            }
            if fire.last_params:
                # fire.last_params is a FLAT CLI-style dict (e.g.
                # 'hdbscan_min_samples', 'tsne_perplexity', 'embed_bands',
                # 'rf_n_estimators', 'brush_size'). The previous version
                # expected nested sub-dicts under 'tsne'/'hdbscan'/
                # 'random_forest' keys and silently wrote nothing, so
                # every accepted YAML (and the PDF built from it) lost
                # bands, t-SNE, RF, HDBSCAN, and brush settings. Group by
                # prefix so readers can pull a whole stage without string
                # parsing; unknown keys fall into 'misc'.
                _prefix_to_section = (
                    ('tsne_',    'tsne'),
                    ('hdbscan_', 'hdbscan'),
                    ('rf_',      'random_forest'),
                    ('brush_',   'brush'),
                )
                _explicit = {
                    'embed_bands':       'bands',
                    'point_threshold':   'brush',
                    'controlled_ratio':  'random_forest',
                    'contour_width':     'output',
                }
                # These are already represented in higher-level sections
                # (crop/sampling). Skip to avoid duplication/conflicting
                # values if the per-run override differs from the global.
                _skip = {'padding', 'sample_rate', 'min_samples', 'max_samples'}
                for k, v in fire.last_params.items():
                    if v is None or v == '':
                        continue
                    if k in _skip:
                        continue
                    section = None
                    for prefix, sec in _prefix_to_section:
                        if k.startswith(prefix):
                            section = sec
                            break
                    if section is None:
                        section = _explicit.get(k, 'misc')
                    params_dict.setdefault(section, {})[k] = v

            path = os.path.join(fire_dir, f'{fire_numbe}_params.yaml')
            _atomic_yaml_dump(path, params_dict, mode=0o644)
        except ImportError:
            pass

        # Update fire_status.yaml (atomic write). Hold the file lock across
        # the read-modify-write so concurrent accepts of different fires
        # don't lose each other's entries.
        try:
            import yaml
            status_path = os.path.join(state.output_root, 'fire_status.yaml')
            with _accept_file_lock:
                idx = {}
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        idx = yaml.safe_load(f) or {}
                idx[fire_numbe] = {
                    'status': 'accepted',
                    'timestamp': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                    'fire_dir': fire_dir,
                    'source': 'web',
                }
                _atomic_yaml_dump(status_path, idx)
        except Exception:
            pass

        # Clean up XML artefacts
        for xml in glob.glob(os.path.join(fire_dir, '*.xml')):
            try:
                os.remove(xml)
            except Exception:
                pass

        # Append to accepted_params.csv for parameter learning (deduplicate).
        # The full read-dedupe-rewrite-append sequence runs under the file
        # lock so concurrent accepts cannot interleave and corrupt the file.
        try:
            import csv
            csv_path = os.path.join(state.output_root, 'accepted_params.csv')
            with _accept_file_lock:
                # Read existing rows (if any), drop the row for this fire
                # (dedupe on re-accept), then write everything + the new row
                # in a single tmp-file + rename so a crash or disk-full
                # cannot truncate the CSV mid-write.
                existing = []
                if os.path.isfile(csv_path):
                    with open(csv_path, newline='') as cf:
                        reader = csv.DictReader(cf)
                        existing = [r for r in reader
                                    if r.get('fire_numbe') != fire_numbe]

                row_data = {
                    'fire_numbe': fire_numbe,
                    'fire_size_ha': fire.fire_size_ha,
                    'agreement_pct': fire.agreement_pct,
                    'padding': fire.padding_used,
                    'timestamp': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                }
                if fire.last_params:
                    for k, v in fire.last_params.items():
                        row_data[k] = v

                tmp_path = (
                    f'{csv_path}.{os.getpid()}.{threading.get_ident()}.tmp')
                try:
                    with open(tmp_path, 'w', newline='') as cf:
                        writer = csv.DictWriter(
                            cf, fieldnames=_CSV_FIELDNAMES,
                            extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(existing)
                        writer.writerow(row_data)
                    os.replace(tmp_path, csv_path)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
        except Exception as exc:
            sys.stderr.write(
                f'[save] WARNING: Failed to update accepted_params.csv: '
                f'{exc}\n')

        # Re-point last_comparison at the canonical copy. Until now
        # it points into cache_dir, which _cache_sweep is free to
        # reap once status flips to ACCEPTED — that would leave the
        # UI / PDF builder pointing at a deleted file.
        canon_comp = os.path.join(
            fire_dir, f'{fire_numbe}_comparison.png')
        if os.path.isfile(canon_comp):
            fire.last_comparison = canon_comp

        # Flip status + clear ephemeral tracking state under state.lock
        # so readers never observe a fire that is ACCEPTED but still
        # has a live progress snapshot. Per-run serial gallery cleanup
        # (fire.serial_results + on-disk serial_* files) is the
        # caller's responsibility — the mapping worker has the full
        # list and deletes the files in its cancel path; clearing the
        # list here would strand those files.
        with state.lock:
            fire.status = FireStatus.ACCEPTED
            fire.previously_accepted = False
            fire.progress = {}
            if state.current_job:
                cur = state.current_job.get('fire_numbe', '')
                if cur.split(' (run')[0].strip() == fire_numbe:
                    state.current_job = None
        _save_fire_state()
        return fire_dir
    finally:
        with _accept_in_progress_lock:
            _accept_in_progress.discard(fire_numbe)


# =========================================================================
# Build fire_mapping_cli.py command
# =========================================================================

def _build_mapping_cmd(fire: FireInfo, params: dict,
                       save_state: str = None,
                       load_state: str = None) -> list[str]:
    """Build the subprocess command for fire_mapping_cli.py.

    Raises ValueError if any parameter fails validation.
    """
    rate = params.get('sample_rate')
    rate = float(rate) if rate is not None else state.sample_rate
    min_s = params.get('min_samples')
    min_s = int(min_s) if min_s is not None else state.min_samples
    max_s = params.get('max_samples')
    max_s = int(max_s) if max_s is not None else state.max_samples
    sample_size = int(round(fire.crop_w * fire.crop_h * rate))
    sample_size = max(min_s, min(max_s, sample_size))

    # '-u' forces line-buffered stdout on the CLI child. Without it,
    # Python block-buffers to a pipe (~8 KB), so the web UI sees many
    # stage-transition lines arrive in one burst and the progress pills
    # appear to flip to "done" all at once. With '-u' each print flushes
    # immediately and the UI can animate each stage individually.
    cmd = [
        sys.executable,
        '-u',
        state.cli_script,
        '--sample_size', str(sample_size),
        fire.crop_bin,
        fire.hint_bin,
        '--fire_numbe', fire.fire_numbe,
        '--start_date', fire.acc_start,
        '--end_date', fire.acc_end,
    ]

    if fire.perim_bin and os.path.exists(fire.perim_bin):
        cmd += ['--perimeter', fire.perim_bin]

    if save_state:
        cmd += ['--save_state', save_state]
    if load_state:
        cmd += ['--load_state', load_state]

    flag_map = {
        'seed': '--seed',
        'rf_n_estimators': '--rf_n_estimators',
        'rf_max_depth': '--rf_max_depth',
        'rf_max_features': '--rf_max_features',
        'rf_random_state': '--rf_random_state',
        'controlled_ratio': '--controlled_ratio',
        'hdbscan_min_samples': '--hdbscan_min_samples',
        'tsne_perplexity': '--tsne_perplexity',
        'tsne_learning_rate': '--tsne_learning_rate',
        'tsne_max_iter': '--tsne_max_iter',
        'tsne_init': '--tsne_init',
        'tsne_n_components': '--tsne_n_components',
        'tsne_random_state': '--tsne_random_state',
        'contour_width': '--contour_width',
        'brush_size': '--brush_size',
        'point_threshold': '--point_threshold',
    }

    for key, flag in flag_map.items():
        val = params.get(key)
        if val is not None and str(val).strip():
            val = _validate_param(key, val)
            # Argparse int args choke on "15.0" — normalise whole floats
            if isinstance(val, float) and val == int(val):
                val = int(val)
            cmd += [flag, str(val)]

    # Boolean store_true flag — append only when truthy
    bas = params.get('brush_all_segments')
    if bas is not None and str(bas).strip() != '':
        if _validate_param('brush_all_segments', bas):
            cmd.append('--brush_all_segments')

    eb = params.get('embed_bands')
    if eb and str(eb).strip():
        eb = _validate_embed_bands(eb)
        if eb:
            cmd += ['--embed_bands', eb]

    return cmd


# =========================================================================
# class_brush rebrush helpers — reruns post-processing only, no t-SNE/RF.
# =========================================================================

def _class_brush_exe() -> str:
    """Locate the compiled class_brush.exe relative to project_root."""
    # state.project_root is wps-research/data/bill; class_brush.exe lives at
    # wps-research/cpp/class_brush.exe
    root = state.project_root
    repo_root = os.path.dirname(os.path.dirname(root))
    return os.path.join(repo_root, 'cpp', 'class_brush.exe')


def _read_envi_mask(path: str) -> np.ndarray:
    """Read an ENVI .bin classification as a boolean mask (first band)."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Cannot open {path}')
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return (arr > 0) & np.isfinite(arr)


def _write_envi_mask_like(mask: np.ndarray, out_path: str,
                           ref_path: str) -> None:
    """Write a boolean mask as float32 ENVI .bin, copying the reference
    file's .hdr geometry so downstream GDAL readers see identical
    dimensions/projection."""
    mask.astype(np.float32).tofile(out_path)
    # Copy the sibling .hdr from ref_path (handles both foo.hdr and
    # foo.bin.hdr naming conventions).
    ref_hdr = os.path.splitext(ref_path)[0] + '.hdr'
    if not os.path.isfile(ref_hdr):
        ref_hdr = ref_path + '.hdr'
    if os.path.isfile(ref_hdr):
        out_hdr = os.path.splitext(out_path)[0] + '.hdr'
        shutil.copy2(ref_hdr, out_hdr)


# Registry of running rebrush subprocesses, keyed by fire_numbe. The
# cancel endpoint uses this to terminate a running class_brush.exe.
_rebrush_procs: dict = {}
_rebrush_procs_lock = threading.Lock()


def _run_class_brush_only(clf_path: str, brush_size: int,
                          point_threshold: int,
                          all_segments: bool,
                          fire_numbe: str | None = None
                          ) -> tuple[np.ndarray | None, bool]:
    """Shell to class_brush.exe on an existing classification.

    Returns ``(brushed_mask, cancelled)``:
      - brushed_mask: boolean mask, or None if exe unavailable / produced
        nothing / was cancelled.
      - cancelled:    True iff a cancel signal terminated the subprocess.

    When ``fire_numbe`` is provided, the subprocess handle is registered
    in ``_rebrush_procs[fire_numbe]`` so ``handle_api_rebrush_cancel``
    can terminate it. Intermediate files are always cleaned up.
    """
    brush_exe = _class_brush_exe()
    if not os.path.isfile(brush_exe):
        return None, False

    cmd = [brush_exe]
    if all_segments:
        cmd.append('--all_segments')
    cmd += [clf_path, str(int(brush_size)), str(int(point_threshold))]

    proc = subprocess.Popen(
        cmd, cwd=os.path.dirname(clf_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if fire_numbe is not None:
        with _rebrush_procs_lock:
            _rebrush_procs[fire_numbe] = proc
    try:
        # communicate() drains stdout+stderr concurrently on internal
        # threads. Using proc.wait() here would deadlock once the OS
        # pipe buffer fills up (class_brush.exe emits one line per
        # component — easily >64KB for complex fires).
        proc.communicate()
        rc = proc.returncode
    finally:
        if fire_numbe is not None:
            with _rebrush_procs_lock:
                _rebrush_procs.pop(fire_numbe, None)

    # rc < 0 on POSIX when terminated by signal (e.g. SIGTERM = -15).
    cancelled = rc is not None and rc < 0

    comp_files = sorted(glob.glob(clf_path + '_comp_*.bin'))
    brushed = None
    if comp_files:
        if all_segments:
            for cf in comp_files:
                try:
                    m = _read_envi_mask(cf)
                    brushed = m if brushed is None else (brushed | m)
                except Exception:
                    continue
        else:
            largest_count = -1
            for cf in comp_files:
                try:
                    m = _read_envi_mask(cf)
                    c = int(m.sum())
                    if c > largest_count:
                        largest_count = c
                        brushed = m
                except Exception:
                    continue

    # Clean up component files + their headers
    for cf in comp_files:
        for p in (cf, os.path.splitext(cf)[0] + '.hdr'):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

    # Clean up C++ stage intermediaries
    for suffix in ('_flood4.bin', '_flood4.hdr',
                   '_flood4.bin_link.bin', '_flood4.bin_link.hdr',
                   '_flood4.bin_link.bin_recode.bin',
                   '_flood4.bin_link.bin_recode.hdr',
                   '_flood4.bin_link.bin_recode.bin_wheel.bin',
                   '_flood4.bin_link.bin_recode.bin_wheel.hdr'):
        p = clf_path + suffix
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    # Discard any partial output if the subprocess was cancelled.
    if cancelled:
        return None, True
    return brushed, False


def _align_mask_to_crop_frame(mask_path: str, crop_bin_path: str,
                              target_h: int, target_w: int):
    """Align a raster mask to the current crop's geographic frame and
    resize to ``(target_h, target_w)``.

    When the mask and the crop share pixel size but differ in extent
    (e.g. the classification was produced with a different padding
    epoch than the current crop), this places the mask in the crop's
    coordinate frame at the correct pixel offset computed from the
    geotransforms — the same technique ``_overlay_mask_on_post`` uses
    for the interactive web UI. That path is why the main-page overlay
    looks right; the PDF report used a naive stretch which visibly
    mis-scaled the overlay. Falls back to a naive zoom when
    geotransforms aren't available or pixel sizes don't match.

    Returns a boolean ``ndarray`` of shape ``(target_h, target_w)``,
    or ``None`` if the mask can't be loaded.
    """
    if not mask_path or not os.path.isfile(mask_path):
        return None
    try:
        from scipy.ndimage import zoom as scipy_zoom
        ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
        if ds is None:
            return None
        arr = ds.GetRasterBand(1).ReadAsArray()
        old_gt = ds.GetGeoTransform()
        ds = None
        if arr is None:
            return None

        ah, aw = arr.shape
        aligned = False

        if crop_bin_path and os.path.isfile(crop_bin_path):
            try:
                ds_crop = gdal.Open(crop_bin_path, gdal.GA_ReadOnly)
                if ds_crop is not None:
                    new_gt = ds_crop.GetGeoTransform()
                    new_w = ds_crop.RasterXSize
                    new_h = ds_crop.RasterYSize
                    ds_crop = None

                    if (old_gt and new_gt
                            and abs(old_gt[1] - new_gt[1]) < 1e-6
                            and abs(old_gt[5] - new_gt[5]) < 1e-6):
                        off_x = round(
                            (old_gt[0] - new_gt[0]) / new_gt[1])
                        off_y = round(
                            (old_gt[3] - new_gt[3]) / new_gt[5])
                        arr_aligned = np.zeros(
                            (new_h, new_w), dtype=arr.dtype)
                        src_y0 = max(0, -off_y)
                        src_x0 = max(0, -off_x)
                        dst_y0 = max(0, off_y)
                        dst_x0 = max(0, off_x)
                        copy_h = min(ah - src_y0, new_h - dst_y0)
                        copy_w = min(aw - src_x0, new_w - dst_x0)
                        if copy_h > 0 and copy_w > 0:
                            arr_aligned[
                                dst_y0:dst_y0 + copy_h,
                                dst_x0:dst_x0 + copy_w,
                            ] = arr[
                                src_y0:src_y0 + copy_h,
                                src_x0:src_x0 + copy_w,
                            ]
                        arr = arr_aligned
                        aligned = True
            except Exception:
                pass

        ah2, aw2 = arr.shape
        if (ah2, aw2) != (target_h, target_w):
            arr = scipy_zoom(
                arr.astype(np.uint8),
                (target_h / ah2, target_w / aw2), order=0)

        return arr.astype(bool)
    except Exception as exc:
        sys.stderr.write(
            f'[align_mask] WARNING: {mask_path}: {exc}\n')
        return None


def _render_comparison_png(fire: 'FireInfo', classified_path: str,
                           out_path: str) -> bool:
    """Regenerate the perimeter-style comparison PNG.

    Mirrors fire_mapping_cli.make_comparison_figure so callers (rebrush)
    can keep ``<fire>_comparison.png`` in sync with the current
    ``classified.bin`` without re-running the full mapping pipeline.
    Background is taken from ``cache_dir/previews/post.png``.

    Returns True on success, False if inputs are missing or render fails.
    """
    try:
        post_path = os.path.join(fire.cache_dir, 'previews', 'post.png')
        if not (os.path.isfile(post_path)
                and os.path.isfile(classified_path)):
            return False

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
        from matplotlib.lines import Line2D
        from scipy.ndimage import zoom as scipy_zoom, binary_dilation

        bg = imread(post_path)
        if bg.ndim == 2:
            bg = np.stack([bg] * 3, axis=2)
        bh, bw = bg.shape[:2]

        def _contour_rgba(mask, rgb):
            dil = binary_dilation(mask)
            boundary = dil & (~mask)
            rgba = np.zeros((bh, bw, 4), dtype=np.float32)
            rgba[..., 0] = rgb[0]
            rgba[..., 1] = rgb[1]
            rgba[..., 2] = rgb[2]
            rgba[..., 3] = boundary.astype(np.float32)
            return rgba

        # Use geotransform-aware placement so masks produced under a
        # different crop/padding epoch still land at the right position
        # relative to the current post-fire background.
        crop_bin = fire.crop_bin or ''
        clf_mask = _align_mask_to_crop_frame(
            classified_path, crop_bin, bh, bw)
        hint_path = fire.hint_bin or fire.viirs_bin or ''
        hint_mask = _align_mask_to_crop_frame(
            hint_path, crop_bin, bh, bw)
        perim_mask = _align_mask_to_crop_frame(
            fire.perim_bin, crop_bin, bh, bw)

        # If perimeter == hint file, drop the separate cyan outline.
        if (fire.perim_bin and hint_path
                and os.path.abspath(fire.perim_bin)
                == os.path.abspath(hint_path)):
            perim_mask = None

        pt = (fire.perimeter_type or '').lower()
        if pt == 'viirs':
            hint_label = 'VIIRS hint'
        elif pt in ('polygon_perimeter', 'polygon', 'traditional'):
            hint_label = 'Traditional perimeter (hint)'
        else:
            hint_label = 'Hint'

        def _iou(a, b):
            if a is None or b is None:
                return None
            inter = int(np.sum(a & b))
            union = int(np.sum(a | b))
            return inter / union if union > 0 else 0.0

        def _acc(a, b):
            if a is None or b is None:
                return None
            return float(np.sum(a == b)) / a.size if a.size else 0.0

        iou_cv = _iou(clf_mask, hint_mask)
        acc_cv = _acc(clf_mask, hint_mask)
        parts = []
        if iou_cv is not None:
            parts.append(f'IoU(ours/{hint_label})={iou_cv:.3f}')
        if acc_cv is not None:
            parts.append(f'Acc(ours/{hint_label})={acc_cv:.3f}')
        metrics = '  '.join(parts)
        if perim_mask is not None and clf_mask is not None:
            iou_cp = _iou(clf_mask, perim_mask)
            iou_vp = _iou(hint_mask, perim_mask) if hint_mask is not None else None
            extras = []
            if iou_cp is not None:
                extras.append(f'IoU(ours/perim)={iou_cp:.3f}')
            if iou_vp is not None:
                extras.append(f'IoU({hint_label}/perim)={iou_vp:.3f}')
            if extras:
                metrics += '\n' + '  '.join(extras)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(bg, interpolation='nearest', origin='upper')
        if clf_mask is not None:
            ax.imshow(_contour_rgba(clf_mask, (1.0, 0.0, 0.0)),
                      interpolation='nearest', origin='upper')
        if hint_mask is not None:
            ax.imshow(_contour_rgba(hint_mask, (0.0, 1.0, 0.0)),
                      interpolation='nearest', origin='upper')
        if perim_mask is not None:
            ax.imshow(_contour_rgba(perim_mask, (0.0, 1.0, 1.0)),
                      interpolation='nearest', origin='upper')

        ax.set_xlim(0, bw)
        ax.set_ylim(bh, 0)
        ax.set_xlabel('Column (px)', fontsize=8)
        ax.set_ylabel('Row (px)', fontsize=8)
        ax.tick_params(labelsize=7)

        handles = []
        if clf_mask is not None:
            handles.append(Line2D([0], [0], color='red', linewidth=2,
                                  label='Our mapping'))
        if hint_mask is not None:
            handles.append(Line2D([0], [0], color='lime', linewidth=2,
                                  label=hint_label))
        if perim_mask is not None:
            handles.append(Line2D([0], [0], color='cyan', linewidth=2,
                                  label='Traditional perimeter'))
        if handles:
            ax.legend(handles=handles, loc='lower right', fontsize=9,
                      framealpha=0.7, edgecolor='white')

        title = f'Fire: {fire.fire_numbe}'
        if fire.acc_start or fire.acc_end:
            title += (f'   |   Start: {fire.acc_start}'
                      f'   |   End: {fire.acc_end}')
        if metrics:
            title += f'\n{metrics}'
        ax.set_title(title, fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as exc:
        sys.stderr.write(
            f'[comparison] WARNING: regen failed: {exc}\n')
        sys.stderr.flush()
        return False


def _render_ml_classification_png(fire_numbe: str, post_path: str,
                                  classified_path: str, out_path: str,
                                  subtitle: str = '',
                                  hint_path: str = '',
                                  hint_label: str = 'Hint',
                                  crop_bin: str = '') -> bool:
    """Render an ML-classification view: post-fire background with the
    burned mask tinted red (filled). Optionally overlays a lime contour
    for the hint perimeter (VIIRS or traditional) so the reader can see
    the ground truth alongside the ML output.

    Ephemeral — used only for the PDF report's Pass-1 hero pages, never
    saved to a fire's permanent directory. ``subtitle`` is appended to
    the figure's title.

    Returns True on success.
    """
    try:
        if not (os.path.isfile(post_path)
                and os.path.isfile(classified_path)):
            return False

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
        from matplotlib.lines import Line2D
        from scipy.ndimage import zoom as scipy_zoom, binary_dilation

        bg = imread(post_path)
        if bg.ndim == 2:
            bg = np.stack([bg] * 3, axis=2)
        bh, bw = bg.shape[:2]

        # Geotransform-aware placement when crop_bin is known; falls
        # back to naive resize otherwise (preserves prior behavior for
        # any external callers that don't pass crop_bin).
        if crop_bin:
            aligned = _align_mask_to_crop_frame(
                classified_path, crop_bin, bh, bw)
            if aligned is None:
                return False
            mask = aligned
        else:
            ds = gdal.Open(classified_path, gdal.GA_ReadOnly)
            if ds is None:
                return False
            arr = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            ah, aw = arr.shape
            if (ah, aw) != (bh, bw):
                arr = scipy_zoom(arr.astype(np.uint8),
                                 (bh / ah, bw / aw), order=0)
            mask = arr > 0

        rgb = bg[:, :, :3].astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        r, g, b = 0.9, 0.1, 0.0
        result = rgb.copy()
        result[mask, 0] = np.clip(rgb[mask, 0] * 0.3 + r * 0.7, 0, 1)
        result[mask, 1] = np.clip(rgb[mask, 1] * 0.3 + g * 0.7, 0, 1)
        result[mask, 2] = np.clip(rgb[mask, 2] * 0.3 + b * 0.7, 0, 1)

        # Optional hint contour (lime). Same contour technique as
        # _render_comparison_png so outline style matches the detail
        # pages. Silently skipped if the hint raster is missing.
        hint_rgba = None
        if hint_path and os.path.isfile(hint_path):
            try:
                if crop_bin:
                    hmask = _align_mask_to_crop_frame(
                        hint_path, crop_bin, bh, bw)
                else:
                    ds_h = gdal.Open(hint_path, gdal.GA_ReadOnly)
                    hmask = None
                    if ds_h is not None:
                        harr = ds_h.GetRasterBand(1).ReadAsArray()
                        ds_h = None
                        hh, hw = harr.shape
                        if (hh, hw) != (bh, bw):
                            harr = scipy_zoom(
                                harr.astype(np.uint8),
                                (bh / hh, bw / hw), order=0)
                        hmask = harr > 0
                if hmask is not None and hmask.any():
                    dil = binary_dilation(hmask)
                    boundary = dil & (~hmask)
                    hint_rgba = np.zeros((bh, bw, 4), dtype=np.float32)
                    hint_rgba[..., 1] = 1.0  # lime
                    hint_rgba[..., 3] = boundary.astype(np.float32)
            except Exception:
                hint_rgba = None

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(result, interpolation='nearest', origin='upper')
        if hint_rgba is not None:
            ax.imshow(hint_rgba, interpolation='nearest', origin='upper')
        ax.set_xlim(0, bw)
        ax.set_ylim(bh, 0)
        ax.set_xlabel('Column (px)', fontsize=8)
        ax.set_ylabel('Row (px)', fontsize=8)
        ax.tick_params(labelsize=7)

        handles = [Line2D([0], [0], color='red', linewidth=6,
                          alpha=0.7, label='ML classification')]
        if hint_rgba is not None:
            handles.append(Line2D([0], [0], color='lime', linewidth=2,
                                  label=hint_label))
        ax.legend(handles=handles, loc='lower right', fontsize=9,
                  framealpha=0.7, edgecolor='white')

        title = f'Fire: {fire_numbe}  —  ML classification'
        if subtitle:
            title += f'\n{subtitle}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as exc:
        sys.stderr.write(
            f'[ml_overlay] WARNING: render failed: {exc}\n')
        sys.stderr.flush()
        return False


def _render_brush_comparison_png(raw: np.ndarray, brushed: np.ndarray | None,
                                 bg_path: str, out_path: str,
                                 title: str) -> None:
    """Draw a two-panel (raw vs brushed) contour figure on bg_path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from scipy.ndimage import zoom as scipy_zoom

    bg = imread(bg_path)
    bh, bw = bg.shape[:2]

    def _resize(mask):
        mh, mw = mask.shape
        if (mh, mw) == (bh, bw):
            return mask.astype(bool)
        zy = bh / mh
        zx = bw / mw
        return scipy_zoom(mask.astype(np.uint8),
                          (zy, zx), order=0).astype(bool)

    def _contour_rgba(mask_bg):
        from scipy.ndimage import binary_dilation
        mask_bg = mask_bg.astype(bool)
        dil = binary_dilation(mask_bg)
        boundary = dil & (~mask_bg)
        rgba = np.zeros((bh, bw, 4), dtype=np.float32)
        rgba[..., 0] = 1.0  # red
        rgba[..., 3] = boundary.astype(np.float32)
        return rgba

    raw_bg = _resize(raw)
    after_mask = _resize(brushed) if brushed is not None else raw_bg
    after_title = ('After class_brush\n(brushed)'
                   if brushed is not None
                   else 'After class_brush\n(no output)')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title, fontsize=10, fontweight='bold')
    for ax, m, t in [
        (axes[0], raw_bg,     'Before class_brush\n(raw classification)'),
        (axes[1], after_mask, after_title),
    ]:
        ax.imshow(bg, interpolation='nearest', origin='upper')
        ax.imshow(_contour_rgba(m), interpolation='nearest', origin='upper')
        ax.set_title(t, fontsize=9)
        ax.set_xlim(0, bw)
        ax.set_ylim(bh, 0)
        ax.set_xlabel('Column (px)', fontsize=8)
        ax.set_ylabel('Row (px)', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# HTTP request handler
# =========================================================================

class FireHandler(BaseHTTPRequestHandler):
    """Routes all HTTP requests for the fire mapping web interface."""

    # -- Routing tables (compiled once) --
    ROUTES_GET = [
        (re.compile(r'^/$'), 'handle_fire_list'),
        (re.compile(r'^/login$'), 'handle_login_page'),
        (re.compile(r'^/logout$'), 'handle_logout'),
        (re.compile(r'^/admin$'), 'handle_admin_page'),
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
        (re.compile(r'^/api/presets$'), 'handle_api_presets_get'),
        (re.compile(r'^/static/(?P<path>.+)$'), 'handle_static'),
    ]
    ROUTES_POST = [
        (re.compile(r'^/login$'), 'handle_login_post'),
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
        (re.compile(
            r'^/api/fire/(?P<fire_numbe>[^/]+)/preset$'),
         'handle_api_preset_post'),
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

    # -- Page handlers --

    def handle_fire_list(self):
        is_admin = (getattr(self, '_role', '') == 'admin')
        admin_link = ('<a href="/admin" class="btn" '
                      'style="font-size:11px;padding:3px 10px">'
                      'Admin</a>'
                      if is_admin else '')
        years_sorted = sorted(state.rasters_by_year)
        html = render_template('fire_list.html', {
            'raster_name': os.path.basename(state.raster_path),
            'polygon_name': os.path.basename(state.polygon_file),
            'n_fires': str(len(state.fires)),
            'admin_link': admin_link,
            'all_years_json': json.dumps(years_sorted),
            'active_year_json': json.dumps(int(state.active_year)),
            'is_admin_json': json.dumps(bool(is_admin)),
        })
        self._send_html(html)

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

    def handle_fire_page(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_html('Fire not found', 404)
            return
        fire = state.fires[fire_numbe]
        html = render_template('fire_mapping.html', {
            'fire_numbe': fire_numbe,
            'fire_numbe_json': json.dumps(fire_numbe),
            'fire_date': fire.fire_date,
            'fire_year': str(fire.fire_year),
            'fire_size_ha': str(fire.fire_size_ha),
            'fire_status': fire.status.value,
            'padding': str(state.padding),
            'sample_rate': str(state.sample_rate),
            'min_samples': str(state.min_samples),
            'max_samples': str(state.max_samples),
        })
        self._send_html(html)

    def handle_static(self, path):
        static_root = os.path.realpath(os.path.join(_HERE, 'static'))
        filepath = os.path.realpath(os.path.join(_HERE, 'static', path))
        if filepath != static_root and not filepath.startswith(
                static_root + os.sep):
            self.send_error(403)
            return
        self._send_file(filepath)

    # -- API handlers --

    def handle_api_fires(self):
        with state.lock:
            fires = [
                {
                    'fire_numbe': f.fire_numbe,
                    'fire_date': f.fire_date,
                    'fire_year': f.fire_year,
                    'fire_size_ha': f.fire_size_ha,
                    'status': f.status.value,
                    'previously_accepted': f.previously_accepted,
                    'agreement_pct': f.agreement_pct,
                    'ml_area_ha': f.ml_area_ha,
                    'notes': f.notes,
                    'has_override': bool(
                        getattr(f, 'recommended_override', None)),
                }
                for f in state.fires.values()
                if not f.hidden
            ]
        self._send_json(fires)

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

    # -- Batch mapping API --

    def handle_api_batch_map(self):
        global _batch_thread
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
            _batch_thread = threading.Thread(
                target=_batch_map_worker,
                args=(fire_numbes, sess_hash),
                daemon=True)
            _batch_thread.start()
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

    # -- Fire API --

    def handle_api_prepare(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return

        fire = state.fires[fire_numbe]
        body = self._read_body()
        if body is None:
            return
        padding = body.get('padding')

        prev_status = fire.status

        needs_prepare = fire.status in (
            FireStatus.PENDING, FireStatus.ERROR)
        # Re-prepare accepted/mapped fires with no previews yet
        if not fire.available_views and fire.status in (
                FireStatus.ACCEPTED, FireStatus.MAPPED):
            needs_prepare = True
        padding_changed = (padding is not None
                           and fire.padding_used != float(padding))
        if padding_changed:
            needs_prepare = True
        if fire.status == FireStatus.PREPARING:
            self._send_json({'status': 'preparing'})
            return

        if needs_prepare:
            with _gpu_lock:
                _prepare_fire_sync(fire_numbe, padding)
            fire = state.fires[fire_numbe]
            if fire.status == FireStatus.READY and prev_status in (
                    FireStatus.ACCEPTED, FireStatus.MAPPED):
                if padding_changed:
                    # User changed params — treat as fresh start
                    fire.previously_accepted = (
                        prev_status == FireStatus.ACCEPTED)
                    fire.last_comparison = ''
                else:
                    # Initial page load — just regenerating previews
                    fire.status = prev_status

        fire = state.fires[fire_numbe]
        if fire.status == FireStatus.ERROR:
            self._send_json(
                {'status': 'error', 'error': fire.error_msg}, 500)
            return

        # Check for comparison images (cache dir only for re-prepared
        # fires; cache + canonical for untouched accepted fires)
        has_comparison = False
        has_brush = False
        canonical = os.path.join(state.output_root, fire_numbe)

        if fire.last_comparison and os.path.isfile(fire.last_comparison):
            has_comparison = True
        elif not fire.previously_accepted:
            for d in (fire.cache_dir, canonical):
                if not d:
                    continue
                comp = os.path.join(d, f'{fire_numbe}_comparison.png')
                if os.path.isfile(comp):
                    fire.last_comparison = comp
                    has_comparison = True
                    break

        if not fire.previously_accepted:
            for d in (fire.cache_dir, canonical):
                if not d:
                    continue
                brush = os.path.join(
                    d, f'{fire_numbe}_brush_comparison.png')
                if os.path.isfile(brush):
                    has_brush = True
                    break

        self._send_json({
            'status': fire.status.value,
            'views': fire.available_views,
            'crop_w': fire.crop_w,
            'crop_h': fire.crop_h,
            'sample_size': fire.sample_size,
            'perimeter_type': fire.perimeter_type,
            'acc_start': fire.acc_start,
            'acc_end': fire.acc_end,
            'has_comparison': has_comparison,
            'has_brush_comparison': has_brush,
            'previously_accepted': fire.previously_accepted,
            'ml_area_ha': fire.ml_area_ha,
        })

    def handle_api_preview(self, fire_numbe, view):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        # Validate view name — alphanumeric/underscore/hyphen only
        if not re.fullmatch(r'[A-Za-z0-9_-]+', view):
            self.send_error(400, 'Invalid view name')
            return
        fire = state.fires[fire_numbe]
        png = os.path.join(fire.cache_dir, 'previews', f'{view}.png')

        # On-the-fly generation for serial overlays
        if not os.path.exists(png):
            m = re.match(r'^serial_(\d+)$', view)
            if m:
                rid = m.group(1)
                # Try multiple naming patterns for the classified raster
                for _pat in (f'{fire_numbe}_serial_{rid}_classified.bin',
                             f'{fire_numbe}_crop.bin_classified.bin'):
                    _cand = os.path.join(fire.cache_dir, _pat)
                    if os.path.isfile(_cand):
                        _overlay_mask_on_post(
                            fire, _cand, view, (0.9, 0.1, 0.0))
                        if os.path.exists(png):
                            break

        if not os.path.exists(png):
            self._send_json(
                {'error': f"Preview '{view}' not available"}, 404)
            return
        self._send_file(png, 'image/png')

    def handle_api_comparison(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        path = None
        if fire.last_comparison and os.path.isfile(fire.last_comparison):
            path = fire.last_comparison
        else:
            for d in (fire.cache_dir,
                      os.path.join(state.output_root, fire_numbe)):
                if not d:
                    continue
                candidate = os.path.join(
                    d, f'{fire_numbe}_comparison.png')
                if os.path.isfile(candidate):
                    path = candidate
                    break
        if not path:
            self._send_json({'error': 'No comparison available'}, 404)
            return
        self._send_file(path, 'image/png')

    def handle_api_brush_comparison(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        path = None
        for d in (fire.cache_dir,
                  os.path.join(state.output_root, fire_numbe)):
            if not d:
                continue
            candidate = os.path.join(
                d, f'{fire_numbe}_brush_comparison.png')
            if os.path.isfile(candidate):
                path = candidate
                break
        if not path:
            self._send_json(
                {'error': 'No brush comparison available'}, 404)
            return
        self._send_file(path, 'image/png')

    def handle_api_rebrush(self, fire_numbe):
        """Re-run class_brush on an existing classification; no re-map.

        POST body:
          brush_size (int)         — required
          point_threshold (int)    — required
          brush_all_segments (bool)— optional, default false
          run_id (int|null)        — optional; when set, rebrushes the
                                     serial run's classified.bin and
                                     updates its per-run brush PNG copy.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        if fire.status == FireStatus.MAPPING:
            self._send_json(
                {'error': 'Mapping in progress; rebrush disabled'}, 409)
            return
        body = self._read_body()
        if body is None:
            return

        try:
            bs  = _validate_param('brush_size',
                                  body.get('brush_size', 15))
            pt  = _validate_param('point_threshold',
                                  body.get('point_threshold', 10))
            bas = _validate_param('brush_all_segments',
                                  body.get('brush_all_segments', False))
        except ValueError as exc:
            self._send_json({'error': str(exc)}, 400)
            return

        run_id = body.get('run_id')
        clf_path = None         # canonical (overwritten with brushed)
        extra_brush_png = None
        if run_id not in (None, ''):
            serial_clf = os.path.join(
                fire.cache_dir,
                f'{fire_numbe}_serial_{run_id}_classified.bin')
            if os.path.isfile(serial_clf):
                clf_path = serial_clf
                extra_brush_png = os.path.join(
                    fire.cache_dir,
                    f'{fire_numbe}_serial_{run_id}_brush.png')
        if clf_path is None:
            for pat in (f'{fire_numbe}_crop.bin_classified.bin',
                        f'{fire_numbe}_crop_classified.bin',
                        f'{fire_numbe}_classified.bin'):
                cand = os.path.join(fire.cache_dir, pat)
                if os.path.isfile(cand):
                    clf_path = cand
                    break
        if clf_path is None:
            self._send_json(
                {'error': 'No classification found to rebrush'}, 404)
            return

        # Prefer the pre-brush sibling as brush input so we're not
        # re-brushing an already-brushed mask. Fall back to the canonical
        # file if no _raw backup exists (older runs predating the fix).
        raw_sibling = os.path.splitext(clf_path)[0] + '_raw.bin'
        source_path = raw_sibling if os.path.isfile(raw_sibling) else clf_path

        post_png = os.path.join(fire.cache_dir, 'previews', 'post.png')
        if not os.path.isfile(post_png):
            self._send_json(
                {'error': 'Post-fire preview missing; prepare the fire first'},
                404)
            return

        # Reject overlapping rebrushes for the same fire — the process
        # registry is single-slot per fire_numbe.
        with _rebrush_procs_lock:
            if fire_numbe in _rebrush_procs:
                self._send_json(
                    {'error': 'A rebrush is already running for this fire'},
                    409)
                return

        try:
            raw = _read_envi_mask(source_path)
            brushed, cancelled = _run_class_brush_only(
                source_path, int(bs), int(pt), bool(bas),
                fire_numbe=fire_numbe)

            if cancelled:
                self._send_json({
                    'status': 'cancelled',
                    'run_id': run_id,
                }, 200)
                return

            # Promote brushed result to the canonical classified.bin so
            # the ML-classification overlay, agreement%, and ML-area
            # metric reflect the new post-brush polygon. Preserve the
            # pre-brush raster as _raw.bin for subsequent rebrushes.
            # ORDER MATTERS: back up the pre-brush source BEFORE we
            # overwrite clf_path, otherwise the "raw backup" would
            # actually end up holding the brushed mask.
            if brushed is not None:
                raw_backup = os.path.splitext(clf_path)[0] + '_raw.bin'
                if source_path != raw_backup:
                    try:
                        shutil.copy2(source_path, raw_backup)
                        hdr_src = os.path.splitext(source_path)[0] + '.hdr'
                        if not os.path.isfile(hdr_src):
                            hdr_src = source_path + '.hdr'
                        if os.path.isfile(hdr_src):
                            shutil.copy2(
                                hdr_src,
                                os.path.splitext(raw_backup)[0] + '.hdr')
                    except OSError:
                        pass
                _write_envi_mask_like(brushed, clf_path, source_path)

            out_png = os.path.join(
                fire.cache_dir, f'{fire_numbe}_brush_comparison.png')
            start = getattr(fire, 'acc_start', '') or ''
            end   = getattr(fire, 'acc_end', '') or ''
            title = (f'Fire: {fire_numbe}  —  class_brush comparison '
                     f'(size={bs}, threshold={pt}, '
                     f'all_segments={bool(bas)})')
            if start or end:
                title += f'\nStart: {start}   |   End: {end}'
            _render_brush_comparison_png(raw, brushed, post_png,
                                         out_png, title)
            if extra_brush_png:
                try:
                    shutil.copy2(out_png, extra_brush_png)
                except OSError:
                    pass

            # Refresh the ML-classification overlay the UI actually
            # reads for this context. Serial runs read
            # previews/serial_{rid}.png; the main classification reads
            # previews/result.png. Also recompute the summary metrics
            # so the gallery/header reflect the new post-brush polygon.
            if brushed is not None:
                is_serial = run_id not in (None, '')
                overlay_name = (f'serial_{run_id}'
                                if is_serial else 'result')
                try:
                    _overlay_mask_on_post(
                        fire, clf_path, overlay_name, (0.9, 0.1, 0.0))
                    if not is_serial:
                        fire.agreement_pct = _compute_agreement(fire)
                        fire.ml_area_ha = _compute_ml_area(fire, clf_path)
                        # Persist the brush params onto last_params so a
                        # later accept writes them to the canonical
                        # params YAML and accepted_params.csv. Without
                        # this, an operator who rebrushes and then
                        # accepts gets a CSV row reflecting the original
                        # brush settings, not the final ones.
                        with state.lock:
                            if fire.last_params is None:
                                fire.last_params = {}
                            fire.last_params['brush_size'] = int(bs)
                            fire.last_params['point_threshold'] = int(pt)
                            fire.last_params['brush_all_segments'] = bool(bas)

                        # Regenerate the saved perimeter-style comparison
                        # PNG so the PDF report's detail pages and any
                        # viewer that reads <fire>_comparison.png reflect
                        # the rebrushed mask. Rebrush previously only
                        # updated previews/result.png + _brush_comparison,
                        # leaving _comparison.png stale and making the
                        # PDF look "horrible" despite a correct UI.
                        comp_path = os.path.join(
                            fire.cache_dir,
                            f'{fire_numbe}_comparison.png')
                        if _render_comparison_png(fire, clf_path, comp_path):
                            fire.last_comparison = comp_path
                            # For already-accepted fires, also push the
                            # refreshed PNG into the canonical fire dir
                            # so a PDF generated without re-accepting
                            # still shows the current mask.
                            if fire.status == FireStatus.ACCEPTED:
                                canon = os.path.join(
                                    state.output_root, fire_numbe)
                                if os.path.isdir(canon):
                                    try:
                                        shutil.copy2(
                                            comp_path,
                                            os.path.join(
                                                canon,
                                                f'{fire_numbe}_comparison.png'))
                                    except OSError:
                                        pass
                    else:
                        new_area = _compute_ml_area(fire, clf_path)
                        new_agr = _compute_agreement(
                            fire, clf_path=clf_path)
                        try:
                            rid_match = int(run_id)
                        except (TypeError, ValueError):
                            rid_match = run_id
                        with state.lock:
                            for sr in fire.serial_results:
                                if sr.get('run_id') == rid_match:
                                    sr['ml_area_ha'] = new_area
                                    sr['agreement_pct'] = new_agr
                                    # Update the run's own params dict
                                    # so a subsequent serial-accept of
                                    # this run promotes the new brush
                                    # values into fire.last_params (see
                                    # handle_api_serial_accept).
                                    p = sr.get('params')
                                    if p is None:
                                        p = {}
                                        sr['params'] = p
                                    p['brush_size'] = int(bs)
                                    p['point_threshold'] = int(pt)
                                    p['brush_all_segments'] = bool(bas)
                                    break
                    _save_fire_state()
                except Exception:
                    pass
        except Exception as exc:
            self._send_json({'error': f'Rebrush failed: {exc}'}, 500)
            return

        brushed_px = int(brushed.sum()) if brushed is not None else 0
        raw_px     = int(raw.sum())
        _push_notification(
            self._session_hash(), 'success',
            f'Rebrush done — {fire_numbe}',
            f'raw={raw_px:,}px → brushed={brushed_px:,}px.',
            fire=fire_numbe)
        self._send_json({
            'status': 'ok',
            'brush_size': int(bs),
            'point_threshold': int(pt),
            'brush_all_segments': bool(bas),
            'run_id': run_id,
            'raw_px': raw_px,
            'brushed_px': brushed_px,
        })

    def handle_api_rebrush_cancel(self, fire_numbe):
        """Terminate a running class_brush.exe for this fire, if any."""
        fire_numbe = unquote(fire_numbe)
        with _rebrush_procs_lock:
            proc = _rebrush_procs.get(fire_numbe)
        if proc is None:
            self._send_json({'status': 'idle'}, 200)
            return
        try:
            proc.terminate()
        except Exception:
            pass
        self._send_json({'status': 'cancelling'}, 200)

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

    def handle_api_status(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        f = state.fires[fire_numbe]
        with state.lock:
            payload = {'status': f.status.value, 'error': f.error_msg}
        self._send_json(payload)

    def handle_api_console(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        f = state.fires[fire_numbe]

        # Check for comparison images
        has_comparison = bool(
            f.last_comparison and os.path.isfile(f.last_comparison))
        has_brush = False
        if not has_comparison:
            canonical = os.path.join(state.output_root, fire_numbe)
            for d in (f.cache_dir, canonical):
                if not d:
                    continue
                if os.path.isfile(
                        os.path.join(d, f'{fire_numbe}_comparison.png')):
                    has_comparison = True
                    break
        for d in (f.cache_dir,
                  os.path.join(state.output_root, fire_numbe)):
            if not d:
                continue
            if os.path.isfile(
                    os.path.join(
                        d, f'{fire_numbe}_brush_comparison.png')):
                has_brush = True
                break

        # Snapshot mutable lists under lock to avoid iteration-during-mutation
        with state.lock:
            console_lines = list(f.console_log)
            raw_serial = list(f.serial_results)
            settings_used = [
                {'label': str(s.get('label', '')),
                 'params': dict(s.get('params', {}))}
                for s in f.serial_settings
            ]

        # Clean serial results for JSON
        serial_results = []
        for r in raw_serial:
            serial_results.append({
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

        # Is class_brush.exe currently running for this fire? The
        # frontend uses this to re-adopt a rebrush that started before
        # a page refresh (or from a different browser tab).
        with _rebrush_procs_lock:
            rebrush_running = fire_numbe in _rebrush_procs

        self._send_json({
            'status': f.status.value,
            'previously_accepted': f.previously_accepted,
            'lines': console_lines,
            'last_params': f.last_params,
            'agreement_pct': f.agreement_pct,
            'ml_area_ha': f.ml_area_ha,
            'available_views': list(f.available_views),
            'serial_results': serial_results,
            'settings_used': settings_used,
            'k_runs_per_setting': int(state.k_runs_per_setting),
            'has_comparison': has_comparison,
            'has_brush_comparison': has_brush,
            'rebrush_running': rebrush_running,
        })

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

    def handle_api_notes(self, fire_numbe):
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        body = self._read_body()
        if body is None:
            return
        state.fires[fire_numbe].notes = body.get('notes', '')
        _save_notes()
        self._send_json({'status': 'saved'})

    _VALID_FN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_. -]*$')

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

    def handle_api_remove(self, fire_numbe):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        # Refuse to hide a fire while a worker is actively using its
        # cache_dir — removing the cache out from under the running
        # subprocess causes spurious failures.
        if fire.status in (FireStatus.MAPPING, FireStatus.PREPARING):
            self._send_json(
                {'error': f'Cannot remove while {fire.status.value}'}, 409)
            return
        # Drop the .web_cache/<FIRE>/ directory so memory doesn't leak
        # on hide/unhide cycles. The canonical output dir (if the fire
        # was accepted) is preserved separately. Re-preparing on unhide
        # will rebuild the cache from scratch.
        cache_dir = fire.cache_dir
        with state.lock:
            fire.hidden = True
            fire.cache_dir = ''
            fire.crop_bin = ''
            fire.hint_bin = ''
            fire.perim_bin = ''
            fire.viirs_bin = ''
            fire.available_views = []
            fire.last_comparison = ''
        if cache_dir and os.path.isdir(cache_dir):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception:
                pass
        _save_fire_state()
        self._send_json({'status': 'removed'})

    def handle_api_unhide(self, fire_numbe):
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        state.fires[fire_numbe].hidden = False
        _save_fire_state()
        self._send_json({'status': 'restored'})

    def handle_api_fires_hidden(self):
        """Return list of hidden fires (for admin restore UI)."""
        if getattr(self, '_role', '') != 'admin':
            self._send_json({'error': 'Admin only'}, 403)
            return
        fires = [
            {
                'fire_numbe': f.fire_numbe,
                'fire_date': f.fire_date,
                'fire_year': f.fire_year,
                'fire_size_ha': f.fire_size_ha,
                'status': f.status.value,
            }
            for f in state.fires.values()
            if f.hidden
        ]
        self._send_json(fires)

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

    # -- Progress / queue / notifications / cache / abort / presets --

    def handle_api_progress(self, fire_numbe):
        """Live progress snapshot for a fire currently being mapped.

        Empty object when the fire is not in a running state. The poller
        in the UI uses this to render the stage-aware progress bar.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        snap = _progress_snapshot(fire)
        # Always include status so UI can decide whether to hide the bar.
        snap['status'] = fire.status.value
        # Also expose queue context — "you are waiting behind N".
        with state.lock:
            current = (dict(state.current_job)
                       if state.current_job else None)
            waiting = [dict(w) for w in state.waiting_jobs]
        snap['queue_current'] = current
        snap['queue_waiting'] = len(waiting)
        self._send_json(snap)

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

    def handle_api_fire_abort(self, fire_numbe):
        """Unified cancel — signals whichever job is currently active.

        Routes to rebrush cancel if a class_brush.exe is running, else
        falls through to the serial-mapping cancel semantics. Returns a
        structured summary the UI can trust without knowing the job
        type. Also records ``fire.last_cancel_reason`` for audit.
        """
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return
        fire = state.fires[fire_numbe]
        body = self._read_body() or {}
        reason = str(body.get('reason', '') or '').strip()[:500]
        user = getattr(self, '_username', '') or ''
        with state.lock:
            fire.last_cancel_reason = (
                f'{datetime.datetime.now().isoformat(timespec="seconds")}'
                f'|{user}|{reason}' if reason else '')
        _save_fire_state()

        actions = []
        with _rebrush_procs_lock:
            proc = _rebrush_procs.get(fire_numbe)
        if proc is not None:
            try:
                proc.terminate()
                actions.append('rebrush_cancel_requested')
            except Exception:
                pass
        if fire.status == FireStatus.MAPPING:
            fire.serial_canceled = True
            # SIGTERM the CLI so _gpu_lock releases in seconds, not
            # minutes. Worker's cleanup waits on serial_accept_event
            # only when serial_accept_promoted is True (which /abort
            # does not set), so the user-cancel path here does not
            # stall on the event.
            _terminate_serial_proc(fire_numbe)
            actions.append('mapping_cancel_requested')

        if not actions:
            self._send_json(
                {'status': 'idle', 'actions': [],
                 'message': 'No running job on this fire.'}, 200)
            return
        self._send_json({'status': 'cancelling', 'actions': actions})

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

    # -- Mapping with SSE streaming --

    def handle_api_map(self, fire_numbe):
        """Run fire_mapping_cli.py and stream output as SSE."""
        global _gpu_queue
        fire_numbe = unquote(fire_numbe)
        if fire_numbe not in state.fires:
            self._send_json({'error': 'Fire not found'}, 404)
            return

        fire = state.fires[fire_numbe]
        body = self._read_body()
        if body is None:
            return
        params = body.get('params', {})

        if fire.status not in (
                FireStatus.READY, FireStatus.MAPPED, FireStatus.ACCEPTED):
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
            queue_pos = _gpu_queue
            _gpu_queue += 1
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
                with state.lock:
                    # Move from waiting to current
                    if job_entry in state.waiting_jobs:
                        state.waiting_jobs.remove(job_entry)
                    state.current_job = {
                        **job_entry,
                        'started_at': datetime.datetime.now().isoformat(
                            timespec='seconds'),
                    }
                    if fire.status == FireStatus.ACCEPTED:
                        fire.previously_accepted = True
                    fire.status = FireStatus.MAPPING
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
                _gpu_queue = max(0, _gpu_queue - 1)
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

def create_server(host: str = '0.0.0.0',
                  port: int = 8765) -> ThreadingHTTPServer:
    """Create and return the threaded HTTP server."""
    server = ThreadingHTTPServer((host, port), FireHandler)
    return server
