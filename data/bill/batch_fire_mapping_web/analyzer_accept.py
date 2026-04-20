"""Accept/unaccept logic for the parameter analyzer.

Accepting a run copies its outputs from ``.analyzer_cache/<FIRE>/p_*/set_*_run_*``
into ``analyzing_parameters/<FIRE>/run_XXXX/`` and appends a row to
``analyzer_accepted.csv``. The biggest-padded accepted crop is kept in
``analyzing_parameters/<FIRE>/<FIRE>_crop_max.bin`` so the overlay viewer
has a single canvas for all accepted perimeters. It only ever grows
(unaccept never shrinks it) -- a bigger backdrop is harmless.
"""

import csv
import datetime
import os
import re
import shutil
import sys
import threading


_csv_lock = threading.Lock()

from .analyzer_state import (
    AnalyzerFireInfo, AnalyzerRun, AnalyzerStatus,
    ANALYZER_CSV_FIELDNAMES,
)
from .analyzer_worker import _padding_key, _run_dir_path


_SIZE_BUCKETS = [
    (0, 10), (10, 50), (50, 100), (100, 500),
    (500, 1000), (1000, 5000), (5000, 10**12),
]


def _size_bucket(ha: float) -> tuple:
    for lo, hi in _SIZE_BUCKETS:
        if lo <= ha < hi:
            return (lo, hi)
    return _SIZE_BUCKETS[-1]


def _region_zone(fire_numbe: str) -> tuple:
    if not fire_numbe:
        return ('', '')
    region = fire_numbe[0].upper() if fire_numbe[0].isalpha() else ''
    zone = ''
    if len(fire_numbe) >= 2 and region:
        zone = region + fire_numbe[1]
    return (region, zone)


_ACCEPT_ID_RE = re.compile(r'^run_(\d{4,})$')


def _next_accept_id(canon_dir: str) -> str:
    """Return the next unused run_XXXX identifier in the canonical dir."""
    max_n = 0
    if os.path.isdir(canon_dir):
        for name in os.listdir(canon_dir):
            m = _ACCEPT_ID_RE.match(name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f'run_{max_n + 1:04d}'


def _locate_run_files(fire_numbe: str, run: AnalyzerRun, astate) -> str:
    """Return the run_dir path inside the analyzer cache for this run.

    Raises FileNotFoundError if the directory (or its agreement.json) is
    missing -- that usually means someone wiped the cache or the run
    never actually completed.
    """
    snap_dir = os.path.join(
        astate.cache_root, fire_numbe, _padding_key(run.padding_used))
    run_dir = _run_dir_path(snap_dir, run.set_idx, run.run_idx)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'Run directory not found: {run_dir}')
    if not os.path.isfile(os.path.join(run_dir, 'agreement.json')):
        raise FileNotFoundError(
            f'Run sidecar missing at {run_dir}/agreement.json')
    return run_dir, snap_dir


def _save_manifest(canon_dir: str, info: AnalyzerFireInfo):
    """Write manifest.yaml listing all accepts + the biggest-padding crop."""
    import yaml
    accepts = [r.accept_id for r in info.runs if r.accepted and r.accept_id]
    data = {
        'fire_numbe': info.fire_numbe,
        'saved_padding': info.saved_max_padding,
        'saved_crop': info.saved_max_crop_rel,
        'accepts': accepts,
        'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    path = os.path.join(canon_dir, 'manifest.yaml')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp, path)


def _write_accept_params_yaml(dest_dir: str, run: AnalyzerRun,
                              info: AnalyzerFireInfo,
                              accept_id: str, app_state):
    """Write params.yaml inside the canonical run_XXXX/ dir."""
    import yaml
    size_lo, size_hi = _size_bucket(_size_bucket_size(info, app_state))
    region, zone = _region_zone(info.fire_numbe)
    data = {
        'fire': {
            'fire_numbe': info.fire_numbe,
            'fire_year': _fire_attr(info, app_state, 'fire_year'),
            'fire_date': _fire_attr(info, app_state, 'fire_date'),
            'fire_size_ha': _fire_attr(info, app_state, 'fire_size_ha'),
            'region': region,
            'zone': zone,
            'size_bucket_lo': size_lo,
            'size_bucket_hi': size_hi,
        },
        'grid': {
            'set_idx': run.set_idx,
            'run_idx': run.run_idx,
        },
        'params': dict(run.params),
        'outcome': {
            'agreement_pct': run.agreement_pct,
            'ml_area_ha': run.ml_area_ha,
            'padding_used': run.padding_used,
        },
        'accept_id': accept_id,
        'accepted_at': run.accepted_at,
    }
    path = os.path.join(dest_dir, 'params.yaml')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp, path)


def _fire_attr(info: AnalyzerFireInfo, app_state, attr):
    """Read a fire attribute from the user-facing FireInfo for provenance."""
    f = app_state.fires.get(info.fire_numbe)
    if f is None:
        return ''
    return getattr(f, attr, '')


def _size_bucket_size(info: AnalyzerFireInfo, app_state) -> float:
    f = app_state.fires.get(info.fire_numbe)
    return float(getattr(f, 'fire_size_ha', 0.0) if f else 0.0)


def _csv_row(info: AnalyzerFireInfo, run: AnalyzerRun,
             accept_id: str, app_state) -> dict:
    """Build one row for analyzer_accepted.csv."""
    size_lo, size_hi = _size_bucket(_size_bucket_size(info, app_state))
    region, zone = _region_zone(info.fire_numbe)
    # Snapshot crop dims (best-effort — read from meta if available)
    crop_w = 0
    crop_h = 0
    perimeter_type = ''
    from .analyzer_worker import _padding_key
    snap_dir = os.path.join(
        app_state.analyzer.cache_root,
        info.fire_numbe,
        _padding_key(run.padding_used))
    meta_path = os.path.join(snap_dir, 'snapshot.json')
    if os.path.isfile(meta_path):
        import json
        try:
            with open(meta_path) as f:
                m = json.load(f)
            crop_w = int(m.get('crop_w', 0))
            crop_h = int(m.get('crop_h', 0))
            perimeter_type = str(m.get('perimeter_type', ''))
        except Exception:
            pass

    row = {
        'fire_numbe': info.fire_numbe,
        'accept_id': accept_id,
        'accepted_at': run.accepted_at,
        'fire_year': _fire_attr(info, app_state, 'fire_year'),
        'fire_size_ha': _fire_attr(info, app_state, 'fire_size_ha'),
        'fire_region': region,
        'fire_zone': zone,
        'size_bucket_lo': size_lo,
        'size_bucket_hi': size_hi,
        'perimeter_type': perimeter_type,
        'crop_w': crop_w,
        'crop_h': crop_h,
        'set_idx': run.set_idx,
        'run_idx': run.run_idx,
        'agreement_pct': run.agreement_pct,
        'ml_area_ha': run.ml_area_ha,
    }
    # Parameter columns
    for k in ('padding', 'sample_rate', 'min_samples', 'max_samples',
               'seed', 'embed_bands',
               'tsne_perplexity', 'tsne_learning_rate',
               'tsne_max_iter', 'tsne_init', 'tsne_n_components',
               'tsne_random_state',
               'controlled_ratio', 'hdbscan_min_samples',
               'rf_n_estimators', 'rf_max_depth', 'rf_max_features',
               'rf_random_state', 'contour_width'):
        v = run.params.get(k)
        if v is not None and v != '':
            row[k] = v
    return row


def _append_csv_row(astate, row: dict):
    """Append a row atomically: read existing rows, add the new one,
    and write the whole file via tmp + os.replace. Slower than a raw
    append, but a crash cannot leave a half-written or header-less CSV.
    """
    csv_path = astate.csv_file
    with _csv_lock:
        existing = []
        if os.path.isfile(csv_path):
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                existing = list(reader)
        tmp = csv_path + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=ANALYZER_CSV_FIELDNAMES,
                extrasaction='ignore')
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(row)
        os.replace(tmp, csv_path)


def _remove_csv_row(astate, fire_numbe: str, accept_id: str):
    """Rewrite analyzer_accepted.csv without the matching row."""
    csv_path = astate.csv_file
    with _csv_lock:
        if not os.path.isfile(csv_path):
            return
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = [
                r for r in reader
                if not (r.get('fire_numbe') == fire_numbe
                        and r.get('accept_id') == accept_id)
            ]
        tmp = csv_path + '.tmp'
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=ANALYZER_CSV_FIELDNAMES,
                extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, csv_path)


# =========================================================================
# Accept / Unaccept
# =========================================================================

def accept_run(fire_numbe: str, set_idx: int, run_idx: int,
               astate, app_state) -> dict:
    """Promote one grid run to the canonical analyzer dir.

    Returns a dict describing the new accept (accept_id, crop_changed).
    Raises ValueError on bad arguments, FileNotFoundError if the cache
    directory no longer has the run's outputs.
    """
    # Claim the run atomically under the lock so two concurrent accepts
    # cannot both pass the `not r.accepted` check. We allocate accept_id
    # and set run.accepted before releasing the lock; subsequent file
    # work happens outside the lock. If any of it fails, the rollback
    # block restores run.accepted=False so the run can be re-claimed.
    canon_dir = os.path.join(astate.analyzer_root, fire_numbe)
    os.makedirs(canon_dir, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    with astate.lock:
        info = astate.fires.get(fire_numbe)
        if info is None:
            raise ValueError(f'Unknown fire {fire_numbe}')
        run = next((r for r in info.runs
                    if r.set_idx == set_idx and r.run_idx == run_idx
                    and not r.accepted), None)
        if run is None:
            raise ValueError(
                f'No un-accepted run at set={set_idx}, run={run_idx}')
        if run.status != 'done':
            raise ValueError(
                f'Cannot accept a run in state {run.status!r}')
        accept_id = _next_accept_id(canon_dir)
        dest = os.path.join(canon_dir, accept_id)
        # makedirs without exist_ok=True: if two threads ever race to the
        # same accept_id (shouldn't happen under the lock, but cheap
        # defense-in-depth) the second raises and we roll back.
        os.makedirs(dest)
        run.accepted = True
        run.accept_id = accept_id
        run.accepted_at = now

    try:
        run_dir, snap_dir = _locate_run_files(fire_numbe, run, astate)
        # Copy per-run outputs
        for name in ('classified.bin', 'classified.hdr',
                     'comparison.png', 'brush_comparison.png', 'thumb.png'):
            src = os.path.join(run_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, name))
    except Exception:
        with astate.lock:
            run.accepted = False
            run.accept_id = ''
            run.accepted_at = ''
        shutil.rmtree(dest, ignore_errors=True)
        raise

    # Provenance YAML
    _write_accept_params_yaml(dest, run, info, accept_id, app_state)

    # Update biggest-padding crop backdrop if this padding is larger.
    crop_changed = False
    with astate.lock:
        if run.padding_used > info.saved_max_padding:
            src_crop = os.path.join(snap_dir, f'{fire_numbe}_crop.bin')
            if os.path.isfile(src_crop):
                dst_crop = os.path.join(
                    canon_dir, f'{fire_numbe}_crop_max.bin')
                shutil.copy2(src_crop, dst_crop)
                # Header file can live at either <base>.hdr or <full>.hdr
                for hdr_src in (
                        os.path.splitext(src_crop)[0] + '.hdr',
                        src_crop + '.hdr'):
                    if os.path.isfile(hdr_src):
                        shutil.copy2(hdr_src, os.path.splitext(dst_crop)[0] + '.hdr')
                        break
                # Also copy the post-fire preview as the visual backdrop
                src_post = os.path.join(snap_dir, 'previews', 'post.png')
                if os.path.isfile(src_post):
                    shutil.copy2(src_post, os.path.join(
                        canon_dir, f'{fire_numbe}_post_max.png'))
                info.saved_max_padding = float(run.padding_used)
                info.saved_max_crop_rel = f'{fire_numbe}_crop_max.bin'
                crop_changed = True

    _save_manifest(canon_dir, info)
    _append_csv_row(astate, _csv_row(info, run, accept_id, app_state))

    with astate.lock:
        # If this was the first accept ever, bump status
        if info.status == AnalyzerStatus.PENDING:
            info.status = AnalyzerStatus.ANALYZED
        info.last_update = now

    sys.stderr.write(
        f'[analyzer] ACCEPT {fire_numbe} s{set_idx}r{run_idx} '
        f'-> {accept_id} (crop_grew={crop_changed})\n')
    sys.stderr.flush()

    return {
        'accept_id': accept_id,
        'crop_changed': crop_changed,
        'saved_max_padding': info.saved_max_padding,
    }


def unaccept_run(fire_numbe: str, accept_id: str,
                 astate, app_state) -> dict:
    """Remove a previously accepted run from the canonical dir."""
    with astate.lock:
        info = astate.fires.get(fire_numbe)
        if info is None:
            raise ValueError(f'Unknown fire {fire_numbe}')
        run = next((r for r in info.runs
                    if r.accepted and r.accept_id == accept_id), None)
        if run is None:
            raise ValueError(
                f'No accepted run with id {accept_id} on {fire_numbe}')

    canon_dir = os.path.join(astate.analyzer_root, fire_numbe)
    run_dir = os.path.join(canon_dir, accept_id)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)

    with astate.lock:
        run.accepted = False
        run.accept_id = ''
        run.accepted_at = ''
        accepted_left = [r for r in info.runs if r.accepted]

    _remove_csv_row(astate, fire_numbe, accept_id)

    if not accepted_left:
        # No accepted runs remain — clean up the canonical dir entirely.
        # (The worker cache under .analyzer_cache/ is left intact so the
        # admin can re-accept without re-running the grid.)
        if os.path.isdir(canon_dir):
            shutil.rmtree(canon_dir)
        with astate.lock:
            info.saved_max_padding = 0.0
            info.saved_max_crop_rel = ''
            info.last_update = datetime.datetime.now().isoformat(
                timespec='seconds')
    else:
        _save_manifest(canon_dir, info)

    sys.stderr.write(
        f'[analyzer] UNACCEPT {fire_numbe}/{accept_id} '
        f'(accepts_left={len(accepted_left)})\n')
    sys.stderr.flush()

    return {
        'accepts_left': len(accepted_left),
        'saved_max_padding': (
            info.saved_max_padding if accepted_left else 0.0),
    }
