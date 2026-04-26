"""Stage-aware progress tracking — stage markers + running-median ETA.

Stateful helpers read shared state through the module-level ``state``
attribute, set by :func:`init` at server boot.
"""

import os
import sys
import time

from .state import AppState, FireStatus
from .io_utils import _atomic_yaml_dump

state: AppState = None


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


def init(app_state: AppState):
    global state
    state = app_state


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
