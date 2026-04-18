"""Worker thread that runs the N x M parameter grid for selected fires.

Design notes
------------
* Per fire: flatten (set, run) pairs, then group by padding, then by
  t-SNE+RF signature. Within a (padding, signature) group the first run
  writes an .npz cache of the t-SNE embedding and RF-transformed image;
  remaining runs in the group load the cache and re-run HDBSCAN only.
  This matches the optimisation already used by ``_serial_map_worker``
  in app.py.
* Each padding value gets its own ``snapshot`` directory under
  ``.analyzer_cache/<FIRE>/p_<padding>/``. We first call the user's
  ``_prepare_fire_sync`` (which prepares ``.web_cache/<FIRE>/``) and then
  copy the relevant files into the snapshot. This keeps the analyzer's
  crop/hint rasters isolated from subsequent user activity on the same
  fire -- a user re-cropping later can't disturb analyzer work in
  progress, because we've snapshotted the inputs.
* GPU lock is acquired per run (same pattern as serial mapping), not
  for the whole fire -- this lets a user squeeze in mapping requests
  between analyzer runs without waiting for all N*M to finish.
* Per-run outputs live in ``snap_p*/set_<S>_run_<R>/`` with the CLI's
  native filenames. Agreement + params are mirrored to sidecar files
  so the run can be skipped on resume.
* The worker respects ``astate.cancel_event`` between runs and between
  fires. Cancelling leaves partial results intact on disk (resumable).
"""

import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

from .analyzer_state import (
    AnalyzerFireInfo, AnalyzerRun, AnalyzerStatus,
    TSNE_RF_CACHE_KEYS,
)


# =========================================================================
# Planning helpers
# =========================================================================

def _padding_key(padding: float) -> str:
    """Stable string key for a padding value, suitable as a dir name."""
    return f'p_{float(padding):.4f}'


def _tsne_rf_hash(params: dict) -> str:
    """Hash the subset of params that invalidate the t-SNE+RF cache.

    Two param sets with the same hash can share one .npz intermediate
    state file; their HDBSCAN-only runs are just a re-invocation with
    --load_state. The ``padding`` key is part of TSNE_RF_CACHE_KEYS so
    different paddings always produce distinct hashes.
    """
    parts = []
    for k in TSNE_RF_CACHE_KEYS:
        v = params.get(k, '')
        parts.append(f'{k}={v}')
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:12]


def _jitter_hdbscan(base: int, run_idx: int, step: int) -> int:
    """Return a jittered hdbscan_min_samples for replicate `run_idx`.

    Fan-out pattern: base, base+step, base-step, base+2*step, base-2*step, ...
    Guaranteed >= 1. step=0 disables jitter (returns base).
    """
    if step <= 0 or run_idx == 0:
        return max(1, int(base))
    # run_idx 1 -> +1*step, 2 -> -1*step, 3 -> +2*step, 4 -> -2*step, ...
    level = (run_idx + 1) // 2
    sign = 1 if run_idx % 2 == 1 else -1
    return max(1, int(base) + sign * level * int(step))


def _flatten_plan(param_sets: list, m_runs: int, jitter_step: int = 1) -> list:
    """Return [{set_idx, run_idx, padding, signature, params}, ...]

    Sorted so that runs sharing a (padding, signature) are contiguous.
    Within a set, the M replicates differ by a small jitter on
    hdbscan_min_samples (see _jitter_hdbscan) so GPU HDBSCAN produces
    meaningful variation even when it is otherwise deterministic on
    this input. The jittered value doesn't affect the t-SNE+RF cache
    signature so all M runs still reuse the same .npz.
    """
    plan = []
    for s_idx, params in enumerate(param_sets):
        try:
            padding = float(params.get('padding', 0.1))
        except (TypeError, ValueError):
            padding = 0.1
        try:
            base_hms = int(params.get('hdbscan_min_samples', 20))
        except (TypeError, ValueError):
            base_hms = 20
        # The signature is computed from the BASE params — hdbscan_min_samples
        # is not in TSNE_RF_CACHE_KEYS, so jittering it doesn't change the sig.
        sig = _tsne_rf_hash(params)
        for r_idx in range(m_runs):
            jittered = _jitter_hdbscan(base_hms, r_idx, jitter_step)
            run_params = dict(params)
            run_params['hdbscan_min_samples'] = jittered
            plan.append({
                'set_idx': s_idx,
                'run_idx': r_idx,
                'padding': padding,
                'signature': sig,
                'params': run_params,
            })
    plan.sort(key=lambda x: (
        x['padding'], x['signature'], x['set_idx'], x['run_idx']))
    return plan


# =========================================================================
# Snapshot prepare — uses user's _prepare_fire_sync then copies
# =========================================================================

_SNAPSHOT_FILES = (
    # crop + header
    '{fn}_crop.bin', '{fn}_crop.hdr', '{fn}_crop.bin.hdr',
    # perimeter rasterised from shapefile
    '{fn}_perimeter.bin', '{fn}_perimeter.hdr', '{fn}_perimeter.bin.hdr',
)


def _prepare_snapshot(fire_numbe: str, padding: float, snapshot_dir: str,
                      app_state, info: AnalyzerFireInfo) -> dict:
    """Ensure a snapshot exists at snapshot_dir; returns the snapshot metadata.

    If the snapshot crop.bin is already present, just return the existing
    metadata (resume). Otherwise delegate crop/hint generation to the
    user's ``_prepare_fire_sync`` (which writes into .web_cache/<FIRE>/)
    and copy the relevant files into snapshot_dir.
    """
    from .app import _prepare_fire_sync, state as _app_state, _gpu_lock
    from .state import FireStatus

    crop_dst = os.path.join(snapshot_dir, f'{fire_numbe}_crop.bin')
    meta_path = os.path.join(snapshot_dir, 'snapshot.json')

    if os.path.isfile(crop_dst) and os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            pass  # fall through and re-prepare

    os.makedirs(snapshot_dir, exist_ok=True)

    _log(info, f'Preparing snapshot at padding={padding} ...')

    # Hold the GPU lock for BOTH prepare and the file copy — otherwise a
    # concurrent user prepare at a different padding could wipe
    # ``.web_cache/<FIRE>/`` out from under us mid-copy.
    with _gpu_lock:
        _prepare_fire_sync(fire_numbe, padding)
        real_fire = _app_state.fires[fire_numbe]
        if real_fire.status == FireStatus.ERROR:
            raise RuntimeError(f'prepare failed: {real_fire.error_msg}')

        src_cache = real_fire.cache_dir

        # Copy crop + hdrs
        for pat in _SNAPSHOT_FILES:
            name = pat.format(fn=fire_numbe)
            src = os.path.join(src_cache, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(snapshot_dir, name))

        # Copy VIIRS-derived rasters (naming varies)
        for src in glob.glob(os.path.join(src_cache, 'VIIRS_*')):
            shutil.copy2(src, os.path.join(
                snapshot_dir, os.path.basename(src)))

        # Copy preview PNGs for thumbnail generation later
        src_prev = os.path.join(src_cache, 'previews')
        dst_prev = os.path.join(snapshot_dir, 'previews')
        if os.path.isdir(src_prev):
            os.makedirs(dst_prev, exist_ok=True)
            for fname in os.listdir(src_prev):
                src = os.path.join(src_prev, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(dst_prev, fname))

        # Figure out which hint raster we ended up with
        if real_fire.hint_bin:
            hint_name = os.path.basename(real_fire.hint_bin)
            hint_dst = os.path.join(snapshot_dir, hint_name)
            if not os.path.isfile(hint_dst):
                shutil.copy2(real_fire.hint_bin, hint_dst)
            hint_hdr = os.path.splitext(real_fire.hint_bin)[0] + '.hdr'
            if not os.path.isfile(hint_hdr):
                hint_hdr = real_fire.hint_bin + '.hdr'
            if os.path.isfile(hint_hdr):
                shutil.copy2(hint_hdr, os.path.join(
                    snapshot_dir, os.path.basename(hint_hdr)))
        else:
            raise RuntimeError('prepare produced no hint raster')

    # Save metadata
    meta = {
        'fire_numbe': fire_numbe,
        'padding': padding,
        'crop_bin': f'{fire_numbe}_crop.bin',
        'hint_bin': os.path.basename(real_fire.hint_bin),
        'perim_bin': (
            f'{fire_numbe}_perimeter.bin'
            if os.path.isfile(os.path.join(
                snapshot_dir, f'{fire_numbe}_perimeter.bin'))
            else ''),
        'perimeter_type': real_fire.perimeter_type,
        'crop_w': real_fire.crop_w,
        'crop_h': real_fire.crop_h,
        'acc_start': real_fire.acc_start,
        'acc_end': real_fire.acc_end,
        'fire_date': real_fire.fire_date,
        'fire_year': real_fire.fire_year,
        'fire_size_ha': real_fire.fire_size_ha,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    _log(info, f'  snapshot ready ({meta["crop_w"]}x{meta["crop_h"]} px, '
               f'hint={meta["perimeter_type"]})')
    return meta


def _build_shadow_fire(fire_numbe: str, snapshot_dir: str, meta: dict):
    """Build a FireInfo pointing at snapshot files for use by helpers.

    ``_build_mapping_cmd``, ``_compute_agreement``, ``_compute_ml_area``
    and ``_overlay_mask_on_post`` all take a FireInfo. Construct a
    detached one — not registered in state.fires — so the analyzer never
    mutates user-visible state.
    """
    from .state import FireInfo
    sf = FireInfo(
        fire_numbe=fire_numbe,
        fire_date=meta.get('fire_date', ''),
        fire_year=int(meta.get('fire_year', 0) or 0),
        fire_size_ha=float(meta.get('fire_size_ha', 0.0) or 0.0),
    )
    sf.cache_dir = snapshot_dir
    sf.crop_bin = os.path.join(snapshot_dir, meta['crop_bin'])
    sf.hint_bin = os.path.join(snapshot_dir, meta['hint_bin'])
    if meta.get('perim_bin'):
        sf.perim_bin = os.path.join(snapshot_dir, meta['perim_bin'])
    sf.crop_w = int(meta.get('crop_w', 0))
    sf.crop_h = int(meta.get('crop_h', 0))
    sf.acc_start = meta.get('acc_start', '')
    sf.acc_end = meta.get('acc_end', '')
    sf.perimeter_type = meta.get('perimeter_type', '')
    sf.padding_used = float(meta.get('padding', 0.0))
    return sf


# =========================================================================
# Single run
# =========================================================================

def _run_dir_path(snapshot_dir: str, set_idx: int, run_idx: int) -> str:
    return os.path.join(snapshot_dir, f'set_{set_idx:02d}_run_{run_idx:02d}')


def _sidecar_path(snapshot_dir: str, set_idx: int, run_idx: int) -> str:
    return os.path.join(
        _run_dir_path(snapshot_dir, set_idx, run_idx), 'agreement.json')


def _write_sidecar(run_dir: str, data: dict):
    os.makedirs(run_dir, exist_ok=True)
    tmp = os.path.join(run_dir, 'agreement.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, os.path.join(run_dir, 'agreement.json'))


def _write_params_yaml(run_dir: str, params: dict, meta: dict):
    import yaml
    path = os.path.join(run_dir, 'params.yaml')
    data = {
        'grid': {
            'set_idx': meta.get('set_idx'),
            'run_idx': meta.get('run_idx'),
            'signature': meta.get('signature'),
        },
        'params': dict(params),
        'outcome': {
            'agreement_pct': meta.get('agreement_pct'),
            'ml_area_ha': meta.get('ml_area_ha'),
        },
        'fire': {
            'fire_numbe': meta.get('fire_numbe'),
            'fire_date': meta.get('fire_date'),
            'fire_year': meta.get('fire_year'),
            'fire_size_ha': meta.get('fire_size_ha'),
            'padding_used': meta.get('padding_used'),
        },
        'run_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _execute_run(item: dict, snapshot_dir: str, meta: dict,
                 fire_numbe: str, info: AnalyzerFireInfo,
                 app_state, astate) -> AnalyzerRun:
    """Run one grid cell: subprocess, move outputs, compute metrics."""
    from .app import (_build_mapping_cmd, _compute_agreement,
                      _compute_ml_area, _overlay_mask_on_post, _gpu_lock,
                      _stream_subprocess)

    s_idx = item['set_idx']
    r_idx = item['run_idx']
    sig = item['signature']
    params = item['params']

    run_dir = _run_dir_path(snapshot_dir, s_idx, r_idx)
    os.makedirs(run_dir, exist_ok=True)

    # Resume: if the sidecar is already there, read and return.
    sidecar = os.path.join(run_dir, 'agreement.json')
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as f:
                saved = json.load(f)
            run = AnalyzerRun(
                set_idx=s_idx, run_idx=r_idx, params=dict(params),
                padding_used=float(meta.get('padding', 0.0)),
                agreement_pct=float(saved.get('agreement_pct', -1)),
                ml_area_ha=float(saved.get('ml_area_ha', -1)),
                status='done',
                thumb_rel=saved.get('thumb_rel', ''),
                comparison_rel=saved.get('comparison_rel', ''),
                classified_rel=saved.get('classified_rel', ''),
                timestamp=saved.get('timestamp', ''),
            )
            _log(info, f'  set {s_idx} run {r_idx}: RESUMED from disk '
                        f'(agreement={run.agreement_pct}%)')
            return run
        except Exception:
            pass  # corrupt sidecar — re-run

    shadow = _build_shadow_fire(fire_numbe, snapshot_dir, meta)

    # Decide full vs HDBSCAN-only based on .npz cache presence.
    state_file = os.path.join(snapshot_dir, f'tsne_rf_{sig}.npz')
    needs_full = not os.path.isfile(state_file)

    # Mark mapping in progress (admin dashboard visibility)
    app_state.current_job = {
        'fire_numbe': f'{fire_numbe} (analyzer s{s_idx}r{r_idx})',
        'client_ip': 'analyzer',
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }

    try:
        cmd = _build_mapping_cmd(
            shadow, params,
            save_state=state_file if needs_full else None,
            load_state=state_file if not needs_full else None,
        )
    except ValueError as exc:
        app_state.current_job = None
        return AnalyzerRun(
            set_idx=s_idx, run_idx=r_idx, params=dict(params),
            padding_used=shadow.padding_used,
            status='error', error_msg=f'bad params: {exc}')

    _log(info,
         f'  set {s_idx} run {r_idx}: '
         f'{"FULL PIPELINE" if needs_full else "HDBSCAN only"} '
         f'(sig={sig}, hdbscan_min_samples='
         f'{params.get("hdbscan_min_samples", "default")})')

    t0 = time.time()
    try:
        with _gpu_lock:
            if astate.cancel_event.is_set():
                app_state.current_job = None
                return AnalyzerRun(
                    set_idx=s_idx, run_idx=r_idx, params=dict(params),
                    padding_used=shadow.padding_used,
                    status='skipped', error_msg='cancelled')

            def _on_line(text):
                _log(info, f'    | {text}')
            rc, killed = _stream_subprocess(
                cmd, app_state.project_root, _on_line)
    finally:
        app_state.current_job = None

    elapsed = time.time() - t0

    if killed:
        _log(info, f'  set {s_idx} run {r_idx}: KILLED by watchdog '
                   f'(no output for 1800s, {elapsed:.1f}s)')
        return AnalyzerRun(
            set_idx=s_idx, run_idx=r_idx, params=dict(params),
            padding_used=shadow.padding_used,
            status='error',
            error_msg='watchdog: no CLI output for 1800s')

    if rc != 0:
        _log(info, f'  set {s_idx} run {r_idx}: FAILED (rc={rc}, '
                   f'{elapsed:.1f}s)')
        return AnalyzerRun(
            set_idx=s_idx, run_idx=r_idx, params=dict(params),
            padding_used=shadow.padding_used,
            status='error',
            error_msg=f'fire_mapping_cli exited with code {rc}')

    # The CLI writes outputs into dirname(crop_bin) = snapshot_dir using
    # the expected filenames. Compute metrics BEFORE we move the files,
    # since _compute_agreement reads the classified raster from the
    # canonical location ``<cache_dir>/<fn>_crop.bin_classified.bin``.
    cli_clf = os.path.join(
        snapshot_dir, f'{fire_numbe}_crop.bin_classified.bin')
    cli_comp = os.path.join(snapshot_dir, f'{fire_numbe}_comparison.png')
    cli_brush = os.path.join(
        snapshot_dir, f'{fire_numbe}_brush_comparison.png')

    agreement = -1.0
    ml_area = -1.0
    if os.path.isfile(cli_clf):
        try:
            agreement = _compute_agreement(shadow)
            ml_area = _compute_ml_area(shadow, cli_clf)
        except Exception as exc:
            _log(info, f'    (metric calc failed: {exc})')

    # Thumbnail via overlay helper (uses shadow.cache_dir/previews/post.png).
    thumb_rel = ''
    if os.path.isfile(cli_clf):
        thumb_name = f'set_{s_idx:02d}_run_{r_idx:02d}_thumb'
        try:
            _overlay_mask_on_post(
                shadow, cli_clf, thumb_name, (0.9, 0.1, 0.0))
        except Exception as exc:
            _log(info, f'    (thumbnail failed: {exc})')

    # Move per-run outputs into run_dir so subsequent runs don't collide.
    clf_dst = ''
    comp_dst = ''
    if os.path.isfile(cli_clf):
        clf_dst = os.path.join(run_dir, 'classified.bin')
        shutil.move(cli_clf, clf_dst)
        clf_hdr = os.path.splitext(cli_clf)[0] + '.hdr'
        if not os.path.isfile(clf_hdr):
            clf_hdr = cli_clf + '.hdr'
        if os.path.isfile(clf_hdr):
            shutil.move(clf_hdr, os.path.join(run_dir, 'classified.hdr'))
    if os.path.isfile(cli_comp):
        comp_dst = os.path.join(run_dir, 'comparison.png')
        shutil.move(cli_comp, comp_dst)
    if os.path.isfile(cli_brush):
        shutil.move(cli_brush, os.path.join(run_dir, 'brush_comparison.png'))

    # Move thumbnail into run_dir
    thumb_src = os.path.join(
        snapshot_dir, 'previews',
        f'set_{s_idx:02d}_run_{r_idx:02d}_thumb.png')
    if os.path.isfile(thumb_src):
        thumb_dst = os.path.join(run_dir, 'thumb.png')
        shutil.move(thumb_src, thumb_dst)
        thumb_rel = os.path.relpath(thumb_dst, snapshot_dir)

    ts = datetime.datetime.now().isoformat(timespec='seconds')

    # Write sidecar + params.yaml
    _write_sidecar(run_dir, {
        'set_idx': s_idx,
        'run_idx': r_idx,
        'signature': sig,
        'agreement_pct': agreement,
        'ml_area_ha': ml_area,
        'thumb_rel': thumb_rel,
        'comparison_rel': ('comparison.png'
                           if os.path.isfile(os.path.join(run_dir, 'comparison.png'))
                           else ''),
        'classified_rel': ('classified.bin'
                           if os.path.isfile(os.path.join(run_dir, 'classified.bin'))
                           else ''),
        'timestamp': ts,
        'elapsed_sec': round(elapsed, 2),
    })
    _write_params_yaml(run_dir, params, {
        'set_idx': s_idx, 'run_idx': r_idx, 'signature': sig,
        'agreement_pct': agreement, 'ml_area_ha': ml_area,
        'fire_numbe': fire_numbe,
        'fire_date': meta.get('fire_date'),
        'fire_year': meta.get('fire_year'),
        'fire_size_ha': meta.get('fire_size_ha'),
        'padding_used': shadow.padding_used,
    })

    _log(info,
         f'  set {s_idx} run {r_idx}: DONE '
         f'(agreement={agreement}%, ML={ml_area}ha, {elapsed:.1f}s)')

    return AnalyzerRun(
        set_idx=s_idx, run_idx=r_idx, params=dict(params),
        padding_used=shadow.padding_used,
        agreement_pct=agreement, ml_area_ha=ml_area,
        status='done',
        thumb_rel=thumb_rel,
        comparison_rel='comparison.png' if os.path.isfile(
            os.path.join(run_dir, 'comparison.png')) else '',
        classified_rel='classified.bin' if os.path.isfile(
            os.path.join(run_dir, 'classified.bin')) else '',
        timestamp=ts,
    )


# =========================================================================
# Per-fire orchestration
# =========================================================================

def _process_fire(fire_numbe: str, config, app_state, astate):
    """Process the full grid for one fire."""
    # Ensure AnalyzerFireInfo exists.
    with astate.lock:
        info = astate.fires.get(fire_numbe)
        if info is None:
            info = AnalyzerFireInfo(
                fire_numbe=fire_numbe,
                cache_dir=os.path.join(astate.cache_root, fire_numbe),
                canonical_dir=os.path.join(astate.analyzer_root, fire_numbe),
            )
            astate.fires[fire_numbe] = info
        info.status = AnalyzerStatus.ANALYZING
        info.error_msg = ''
        info.console_log.clear()
        # Drop any previous pending/error runs; keep accepted ones.
        info.runs = [r for r in info.runs if r.accepted]

    os.makedirs(info.cache_dir, exist_ok=True)

    jitter = getattr(config, 'm_run_jitter', 1)
    plan = _flatten_plan(
        config.param_sets, config.m_runs_per_set, jitter_step=jitter)
    _log(info,
         f'=== {fire_numbe}: {len(plan)} run(s) planned '
         f'({len(config.param_sets)} sets x {config.m_runs_per_set}, '
         f'HDBSCAN jitter=±{jitter}) ===')

    # Group by padding — each group shares one snapshot.
    by_padding = {}
    for item in plan:
        by_padding.setdefault(item['padding'], []).append(item)

    run_count_done = 0

    for padding, items in by_padding.items():
        if astate.cancel_event.is_set():
            break
        snapshot_dir = os.path.join(info.cache_dir, _padding_key(padding))
        try:
            meta = _prepare_snapshot(
                fire_numbe, padding, snapshot_dir, app_state, info)
        except Exception as exc:
            _log(info, f'!! Snapshot prepare FAILED: {exc}')
            astate.batch_status['errors'].append({
                'fire': fire_numbe,
                'error': f'snapshot prepare at padding={padding}: {exc}',
            })
            continue

        # Sub-group by signature so the .npz cache is hit for runs 2..k.
        by_sig = {}
        for item in items:
            by_sig.setdefault(item['signature'], []).append(item)

        for sig, sig_items in by_sig.items():
            if astate.cancel_event.is_set():
                break
            sig_items.sort(key=lambda x: (x['set_idx'], x['run_idx']))
            for item in sig_items:
                if astate.cancel_event.is_set():
                    break

                # Skip grid cells that already carry an accepted result
                # at this (set_idx, run_idx). Re-running would create a
                # duplicate entry and waste GPU. Admin can unaccept to
                # re-run.
                with astate.lock:
                    existing = next(
                        (r for r in info.runs
                         if r.set_idx == item['set_idx']
                         and r.run_idx == item['run_idx']
                         and r.accepted),
                        None)
                if existing is not None:
                    _log(info,
                         f'  set {item["set_idx"]} run {item["run_idx"]}: '
                         f'SKIPPED (already accepted as '
                         f'{existing.accept_id})')
                    run_count_done += 1
                    astate.batch_status['completed_runs'] += 1
                    continue

                try:
                    run = _execute_run(
                        item, snapshot_dir, meta,
                        fire_numbe, info, app_state, astate)
                except Exception as exc:
                    _log(info, f'  set {item["set_idx"]} run {item["run_idx"]}: '
                               f'EXCEPTION {exc}')
                    sys.stderr.write(
                        f'[analyzer] [{fire_numbe}] set {item["set_idx"]} '
                        f'run {item["run_idx"]} exception:\n'
                        f'{traceback.format_exc()}\n')
                    sys.stderr.flush()
                    run = AnalyzerRun(
                        set_idx=item['set_idx'],
                        run_idx=item['run_idx'],
                        params=dict(item['params']),
                        padding_used=float(padding),
                        status='error',
                        error_msg=str(exc),
                    )

                # Merge into info.runs (replace any non-accepted with same idx)
                with astate.lock:
                    info.runs = [
                        r for r in info.runs
                        if r.accepted or not
                        (r.set_idx == run.set_idx and r.run_idx == run.run_idx)
                    ]
                    info.runs.append(run)
                run_count_done += 1
                astate.batch_status['completed_runs'] += 1

    # Finalize fire status
    with astate.lock:
        if astate.cancel_event.is_set():
            info.status = AnalyzerStatus.PARTIAL
        else:
            done = [r for r in info.runs
                    if r.status == 'done' and not r.accepted]
            expected = len(plan)
            have = sum(1 for r in info.runs
                       if r.status == 'done' and not r.accepted)
            if have >= expected:
                info.status = AnalyzerStatus.ANALYZED
            elif have or any(r.accepted for r in info.runs):
                info.status = AnalyzerStatus.PARTIAL
            else:
                info.status = AnalyzerStatus.ERROR
                if not info.error_msg:
                    info.error_msg = 'no runs completed'

    _log(info,
         f'=== {fire_numbe}: finished {run_count_done}/{len(plan)} run(s) '
         f'-> status={info.status.value} ===')
    _save_fire_runs(info, astate)


# =========================================================================
# Top-level driver
# =========================================================================

def _do_analysis(app_state, astate):
    config = astate.config

    total_runs = (len(config.selected_fires)
                  * len(config.param_sets)
                  * config.m_runs_per_set)
    astate.batch_status = {
        'total_runs': total_runs,
        'completed_runs': 0,
        'total_fires': len(config.selected_fires),
        'completed_fires': 0,
        'current_fire': '',
        'errors': [],
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'finished_at': '',
        'cancelled': False,
    }

    sys.stderr.write(
        f'[analyzer] Starting: {len(config.selected_fires)} fire(s), '
        f'{len(config.param_sets)} set(s), M={config.m_runs_per_set} '
        f'-> {total_runs} total run(s)\n')
    sys.stderr.flush()

    for fire_numbe in config.selected_fires:
        if astate.cancel_event.is_set():
            astate.batch_status['cancelled'] = True
            break
        if fire_numbe not in app_state.fires:
            astate.batch_status['errors'].append({
                'fire': fire_numbe, 'error': 'unknown fire'})
            continue
        astate.batch_status['current_fire'] = fire_numbe
        try:
            _process_fire(fire_numbe, config, app_state, astate)
        except Exception as exc:
            sys.stderr.write(
                f'[analyzer] [{fire_numbe}] FATAL:\n'
                f'{traceback.format_exc()}\n')
            sys.stderr.flush()
            astate.batch_status['errors'].append({
                'fire': fire_numbe, 'error': str(exc)})
            with astate.lock:
                info = astate.fires.get(fire_numbe)
                if info:
                    info.status = AnalyzerStatus.ERROR
                    info.error_msg = str(exc)
        astate.batch_status['completed_fires'] += 1

    astate.batch_status['current_fire'] = ''
    astate.batch_status['finished_at'] = (
        datetime.datetime.now().isoformat(timespec='seconds'))
    sys.stderr.write(
        f'[analyzer] Done. '
        f'completed_fires={astate.batch_status["completed_fires"]}, '
        f'errors={len(astate.batch_status["errors"])}\n')
    sys.stderr.flush()


def analyzer_worker_thread(app_state, astate):
    """Thread entry point. Run the analysis; always clear running flag."""
    try:
        _do_analysis(app_state, astate)
    except Exception:
        sys.stderr.write(
            f'[analyzer] worker crashed:\n{traceback.format_exc()}\n')
        sys.stderr.flush()
    finally:
        with astate.lock:
            astate.running = False
            astate.worker_thread = None
        astate.cancel_event.clear()


def start_worker(app_state, astate) -> bool:
    """Start the worker thread. Returns True on success, False if already running."""
    with astate.lock:
        if astate.running:
            return False
        astate.running = True
        astate.cancel_event.clear()
        t = threading.Thread(
            target=analyzer_worker_thread,
            args=(app_state, astate),
            daemon=True,
            name='analyzer-worker',
        )
        astate.worker_thread = t
    t.start()
    return True


# =========================================================================
# Cache scan — rebuild runs list from disk (on startup, on demand)
# =========================================================================

def scan_cache_for_fire(fire_numbe: str, astate) -> list:
    """Inspect .analyzer_cache/<FIRE>/ for un-accepted run sidecars."""
    runs = []
    fire_cache = os.path.join(astate.cache_root, fire_numbe)
    if not os.path.isdir(fire_cache):
        return runs
    for padding_dir in os.listdir(fire_cache):
        full_pdir = os.path.join(fire_cache, padding_dir)
        if not os.path.isdir(full_pdir) or not padding_dir.startswith('p_'):
            continue
        # Extract padding value from dir name
        try:
            padding = float(padding_dir.split('_', 1)[1])
        except (ValueError, IndexError):
            padding = 0.0
        for run_dir in os.listdir(full_pdir):
            full_rdir = os.path.join(full_pdir, run_dir)
            if not os.path.isdir(full_rdir):
                continue
            sidecar = os.path.join(full_rdir, 'agreement.json')
            if not os.path.isfile(sidecar):
                continue
            try:
                with open(sidecar) as f:
                    data = json.load(f)
            except Exception:
                continue
            # Load params too
            params = {}
            py = os.path.join(full_rdir, 'params.yaml')
            if os.path.isfile(py):
                try:
                    import yaml
                    with open(py) as f:
                        pdoc = yaml.safe_load(f) or {}
                    params = pdoc.get('params', {})
                except Exception:
                    pass
            runs.append(AnalyzerRun(
                set_idx=int(data.get('set_idx', -1)),
                run_idx=int(data.get('run_idx', -1)),
                params=dict(params),
                padding_used=padding,
                agreement_pct=float(data.get('agreement_pct', -1)),
                ml_area_ha=float(data.get('ml_area_ha', -1)),
                status='done',
                thumb_rel=data.get('thumb_rel', ''),
                comparison_rel=data.get('comparison_rel', ''),
                classified_rel=data.get('classified_rel', ''),
                timestamp=data.get('timestamp', ''),
            ))
    return runs


def _save_fire_runs(info: AnalyzerFireInfo, astate):
    """Persist the current runs list to cache_dir/runs.yaml for fast reload."""
    try:
        import yaml
        path = os.path.join(info.cache_dir, 'runs.yaml')
        data = [
            {
                'set_idx': r.set_idx,
                'run_idx': r.run_idx,
                'params': dict(r.params),
                'padding_used': r.padding_used,
                'agreement_pct': r.agreement_pct,
                'ml_area_ha': r.ml_area_ha,
                'status': r.status,
                'error_msg': r.error_msg,
                'thumb_rel': r.thumb_rel,
                'comparison_rel': r.comparison_rel,
                'classified_rel': r.classified_rel,
                'accepted': r.accepted,
                'accept_id': r.accept_id,
                'accepted_at': r.accepted_at,
                'timestamp': r.timestamp,
            }
            for r in info.runs
        ]
        os.makedirs(info.cache_dir, exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, path)
    except Exception as exc:
        sys.stderr.write(
            f'[analyzer] WARNING: Failed to save runs for {info.fire_numbe}: {exc}\n')


# =========================================================================
# Console logging helper (buffers into AnalyzerFireInfo + stderr tee)
# =========================================================================

def _log(info: AnalyzerFireInfo, text: str):
    info.console_log.append(text)
    sys.stderr.write(f'[analyzer] [{info.fire_numbe}] {text}\n')
    sys.stderr.flush()
