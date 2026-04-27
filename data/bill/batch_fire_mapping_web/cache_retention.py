"""Cache retention — prune .web_cache by size + age, never touch canonical.

Stateful helpers read shared state through the module-level ``state``
attribute, set by :func:`init` at server boot. Cross-module dependencies
(in-flight fire sets, fire-state save callback) are also wired by
``init`` to keep this module free of import cycles.
"""

import os
import shutil
import sys
import threading
import time

from .state import AppState, FireStatus
from .io_utils import _atomic_yaml_dump

state: AppState = None

_cache_sweep_lock = threading.Lock()

# Wired by init():
#   _rebrush_procs            — dict of fire_numbe -> Popen for active rebrushes
#   _rebrush_procs_lock       — guards _rebrush_procs
#   _accept_in_progress       — set of fire_numbes mid-accept
#   _accept_in_progress_lock  — guards _accept_in_progress
#   _save_fire_state_cb       — callable, persists fire_state.yaml after a sweep
_rebrush_procs: dict = None
_rebrush_procs_lock: threading.Lock = None
_accept_in_progress: set = None
_accept_in_progress_lock: threading.Lock = None
_save_fire_state_cb = None


def init(app_state: AppState,
         rebrush_procs: dict, rebrush_procs_lock: threading.Lock,
         accept_in_progress: set,
         accept_in_progress_lock: threading.Lock,
         save_fire_state_cb):
    global state, _rebrush_procs, _rebrush_procs_lock
    global _accept_in_progress, _accept_in_progress_lock
    global _save_fire_state_cb
    state = app_state
    _rebrush_procs = rebrush_procs
    _rebrush_procs_lock = rebrush_procs_lock
    _accept_in_progress = accept_in_progress
    _accept_in_progress_lock = accept_in_progress_lock
    _save_fire_state_cb = save_fire_state_cb


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
            except OSError as exc:
                # AUDIT-M6: silently skipping unreadable files makes the
                # sweeper under-count bytes and overshoot max_gb.
                sys.stderr.write(
                    f'[cache] WARNING: stat {fp}: {exc}\n')
                sys.stderr.flush()
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
        # Post-accept rebrush leaves a gallery entry in cache that the
        # canonical dir does NOT have. Reaping the cache would silently
        # destroy the user's rebrush exploration. Soft-pin so manual
        # eviction is still possible but the auto-sweep stays off.
        rebrush_dirty_fires = {
            fn for fn, f in state.fires.items() if f.rebrush_dirty}
    hard_fires = {fn for fn, s in status_by_fire.items() if s in hard_pin}
    soft_fires = {fn for fn, s in status_by_fire.items() if s in soft_pin}
    soft_fires |= rebrush_dirty_fires
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
            _save_fire_state_cb()
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
