"""Mapping worker family.

The two top-level entry points are:

* ``_batch_map_worker`` — drives a list of fires sequentially, delegating
  each to ``_serial_map_worker``. Spawned by ``handle_api_batch_map``.
* ``_serial_map_worker`` — runs N recommended settings × K HDBSCAN
  replicates for a single fire, with per-setting t-SNE+RF caching.
  Spawned by ``handle_api_serial_map`` and called inline by the batch
  worker.

Decomposed phase helpers (in call order):
  ``_serial_setup`` → ``_serial_snapshot_run0`` → ``_serial_run_replicate``
  (looped) → ``_serial_handle_cancel`` → ``_serial_finalize``.

Init-time wiring: everything mutable or shared with ``app.py`` (state,
``_gpu_lock``, ``_batch_cancel``, and a handful of small helpers that
reference module-level globals back in ``app.py``) is bound by
``init(app_state, helpers)``. Functions that live in sibling modules
and read their own state are imported directly.
"""

import datetime
import glob
import os
import shutil
import sys
import traceback

from .state import AppState, FireStatus
from .mapping import (
    _compute_agreement, _compute_ml_area, _overlay_mask_on_post,
    _generate_result_preview,
)
from .mapping_cmd import _build_mapping_cmd
from .prepare import _prepare_fire_sync
from .persistence import _save_fire_state
from .progress import _ProgressTracker, _save_stage_timings
from .notifications import _push_notification

# Bound by ``init`` from app.init_app — these capture module-level
# state that lives in ``app.py`` because it coordinates locks and
# registries across multiple modules.
state: AppState = None
_gpu_lock = None
_batch_cancel = None
_SUBPROCESS_SILENCE_TIMEOUT = None
_set_fire_status = None
_get_recommended_settings = None
_clone_setting = None
_stream_subprocess = None


def init(app_state, helpers):
    """Bind shared globals and helper callables.

    ``helpers`` is the same dict ``_wire_handlers`` builds — we accept
    the superset and pick out what we need so the call site in
    ``init_app`` stays uniform with the handler-mixin wiring.
    """
    global state, _gpu_lock, _batch_cancel, _SUBPROCESS_SILENCE_TIMEOUT
    global _set_fire_status, _get_recommended_settings, _clone_setting
    global _stream_subprocess
    state = app_state
    _gpu_lock = helpers['_gpu_lock']
    _batch_cancel = helpers['_batch_cancel']
    _SUBPROCESS_SILENCE_TIMEOUT = helpers['_SUBPROCESS_SILENCE_TIMEOUT']
    _set_fire_status = helpers['_set_fire_status']
    _get_recommended_settings = helpers['_get_recommended_settings']
    _clone_setting = helpers['_clone_setting']
    _stream_subprocess = helpers['_stream_subprocess']


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


def _serial_setup(fire, fire_numbe: str, settings: list[dict],
                  k_runs: int, k_jitter: int):
    """Wipe leftover serial_* artifacts, reset fire.serial_* fields, and
    return the per-sweep counters and the progress tracker."""
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
    return n_settings, k_runs, k_jitter, n_total, tracker


def _serial_snapshot_run0(fire, fire_numbe: str):
    """If a previously accepted result exists, copy it into the gallery
    as Run 0 so the user can compare new sweeps against the prior best."""
    if not fire.previously_accepted:
        return
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


def _serial_run_replicate(fire, fire_numbe: str, *, setting_idx: int,
                          setting_label: str, base_params: dict,
                          padding: float, state_file: str,
                          replicate: int, run_id: int, n_total: int,
                          params: dict, tracker) -> tuple[bool, bool, str]:
    """Execute a single replicate inside the per-setting loop.

    Returns ``(setting_stopped, broke_out, new_state_file)``. ``broke_out``
    is True when the caller should ``break`` out of the inner replicate
    loop (e.g. because the user cancelled mid-replicate). ``new_state_file``
    is the (possibly rebuilt) per-setting .npz path; the caller must
    keep using it for the rest of the setting.

    Holds ``_gpu_lock`` for the duration of the replicate so concurrent
    accepts can't race the file writes/deletes on cache_dir.
    """
    setting_stopped = False
    broke_out = False
    with _gpu_lock:
        # Cancel window (A): check serial_canceled BEFORE
        # calling _prepare_fire_sync. Without this, a
        # cancel/accept arriving between settings with
        # different paddings would still run a full prep
        # (briefly flipping status to PREPARING) before
        # the worker noticed the cancel — user-visible as
        # a "back to preparing" flicker after Accept.
        if fire.serial_canceled:
            return setting_stopped, True, state_file
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
                    return setting_stopped, False, state_file
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
            return setting_stopped, True, state_file
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
            return setting_stopped, True, state_file

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
    return setting_stopped, broke_out, state_file


def _serial_handle_cancel(fire, fire_numbe: str,
                          session_hash: str | None) -> bool:
    """Run cancel cleanup and return True if a cancel was processed
    (caller should return without finalising).

    Two flavors:

    (A) Accept-initiated cancel (serial_accept_promoted=True): the
        accept handler already copied the chosen run into the main
        slot and flipped status to ACCEPTED. Gallery should clear —
        drop every serial_* file and empty serial_results.

    (B) User-initiated cancel (serial_accept_promoted=False): the
        user hit "stop, that's enough". Keep the gallery. If at
        least one run succeeded, promote the best into the main slot
        and land on MAPPED so the fire is usable. If nothing
        succeeded, revert to the pre-sweep status.

    ``_gpu_lock`` is held across the entire cleanup so concurrent
    accept handlers cannot race our file writes / deletes on the
    same cache_dir.
    """
    if not fire.serial_canceled:
        return False
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
            return True

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
            return True

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
    return True


def _serial_finalize(fire, fire_numbe: str, n_total: int,
                     session_hash: str | None):
    """Pick the best successful run, promote it into the main slot,
    free per-setting npz caches, persist state, and notify the
    initiating session."""
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


def _serial_map_worker(fire_numbe: str, settings: list[dict],
                        k_runs: int, k_jitter: int,
                        session_hash: str | None = None):
    """Run N settings × K HDBSCAN replicates for one fire.

    For each setting, the expensive deterministic part (t-SNE + RF) runs
    once on its first replicate and is cached in a per-setting .npz.
    Replicates 2..K load the cached state and only re-run HDBSCAN with
    a jittered hdbscan_min_samples value (fan-out pattern).
    """
    fire = state.fires[fire_numbe]
    n_settings, k_runs, k_jitter, n_total, tracker = _serial_setup(
        fire, fire_numbe, settings, k_runs, k_jitter)
    _serial_snapshot_run0(fire, fire_numbe)

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
                setting_stopped, broke_out, state_file = (
                    _serial_run_replicate(
                        fire, fire_numbe,
                        setting_idx=setting_idx,
                        setting_label=setting_label,
                        base_params=base_params,
                        padding=padding,
                        state_file=state_file,
                        replicate=replicate,
                        run_id=run_id,
                        n_total=n_total,
                        params=params,
                        tracker=tracker))
                if broke_out:
                    break

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

    if _serial_handle_cancel(fire, fire_numbe, session_hash):
        return

    _serial_finalize(fire, fire_numbe, n_total, session_hash)
