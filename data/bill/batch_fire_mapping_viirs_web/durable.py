"""Durable storage for AOI stacks, and recovery of lost fire identity.

Two jobs, both about surviving a cleared ramdisk.

**Mirroring.** Stacks are built on ``/ram`` because the clustering needs
the speed. ``/ram`` is tmpfs, so a reboot empties it and every fire's
imagery has to be rebuilt from the source mosaics -- and a dated
composite cannot be rebuilt at all once its day's mosaic has rolled off,
so that imagery is simply lost. A copy under the output root (real disk)
costs one sequential write and makes the ramdisk expendable again.

**Recovery.** A fire needs its bounding box and date range to be
re-prepared. Those live in ``fire_state.yaml``, and a save taken while a
fire was only partly loaded could drop them -- leaving a fire that
cannot be re-prepared even though its grid is recorded in every sidecar
beside its stacks. Those sidecars carry the geotransform and raster
size, which is exactly the bounding box, so the identity can be put
back rather than the fire re-created by hand.

Nothing here is on the critical path: mirroring runs in the background
and every failure is logged and skipped.
"""

import glob
import json
import os
import shutil
import sys
import threading
import time

from .state import AppState

state: AppState = None

# Copied alongside the stack. The sidecars are small and cheap, and a
# stack without its header cannot be opened at all.
_STACK_SUFFIXES = ('.bin', '.hdr', '_dates.json', '_overlays.json')

_mirror_lock = threading.Lock()
_mirror_thread = None


def init(app_state: AppState):
    global state
    state = app_state


def store_dir() -> str:
    """Where durable copies live: beside the other persistent state."""
    root = getattr(state, 'output_root', '') or ''
    return os.path.join(root, '.stacks') if root else ''


# ------------------------------------------------------------ mirroring

def _fire_stack_glob(fire) -> str:
    """Glob matching every stack file belonging to *fire*."""
    cb = getattr(fire, 'crop_bin', '') or ''
    base = os.path.basename(cb)
    import re
    m = re.match(r'^\d{8}_stack_(?P<safe>.+?)_(?P<h>[0-9a-fA-F]{6,})'
                 r'(_l2(_d\d{8})?)?\.bin$', base)
    if m:
        ram = os.path.dirname(cb)
        return os.path.join(
            ram, f'*_stack_{m.group("safe")}_{m.group("h")}*')
    return ''


def _is_stack_artifact(path: str) -> bool:
    """True for a stack or one of its sidecars -- never KGC scratch.

    The clustering writes gigabytes of neighbour tables beside the
    stacks. Copying those to disk would be pointless (they are derived
    and rebuildable) and ruinous for space, so the match is positive
    rather than a blacklist.
    """
    name = os.path.basename(path)
    if '.kgc' in name:
        return False
    return any(name.endswith(sfx) for sfx in _STACK_SUFFIXES)


def mirror_fire(fire, log=None) -> dict:
    """Copy this fire's stack files to the durable store.

    Incremental: a file already there with the same size is skipped, so
    repeat calls cost a stat per file.
    """
    out = {'copied': 0, 'skipped': 0, 'bytes': 0, 'failed': 0}
    dest = store_dir()
    pattern = _fire_stack_glob(fire)
    if not dest or not pattern:
        return out
    try:
        os.makedirs(dest, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f'[durable] cannot create {dest}: {exc}\n')
        return out

    for src in sorted(glob.glob(pattern)):
        if not os.path.isfile(src) or not _is_stack_artifact(src):
            continue
        dst = os.path.join(dest, os.path.basename(src))
        try:
            ssize = os.path.getsize(src)
            if os.path.isfile(dst) and os.path.getsize(dst) == ssize:
                out['skipped'] += 1
                continue
            # Copy to a temporary name first: a half-written mirror
            # that looks complete is worse than no mirror, because the
            # restore path would trust it.
            tmp = dst + '.part'
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            out['copied'] += 1
            out['bytes'] += ssize
        except OSError as exc:
            out['failed'] += 1
            sys.stderr.write(f'[durable] {os.path.basename(src)}: '
                             f'{exc}\n')
    if out['copied'] and log:
        log(f'[durable] mirrored {out["copied"]} file(s), '
            f'{out["bytes"] / 1048576:.0f} MB')
    return out


def mirror_all(log=None) -> dict:
    """Mirror every fire currently on the list.

    Only the current list: stale stacks from deleted fires or older
    code are neither copied nor cleaned up, because the question this
    answers is "what must survive a reboot", and that is exactly the
    fires an operator can still see.
    """
    total = {'copied': 0, 'skipped': 0, 'bytes': 0, 'failed': 0,
             'fires': 0}
    try:
        with state.lock:
            fires = list(state.fires.values())
    except Exception:
        return total
    for fire in fires:
        r = mirror_fire(fire)
        for k in ('copied', 'skipped', 'bytes', 'failed'):
            total[k] += r[k]
        total['fires'] += 1
    if total['copied']:
        msg = (f'[durable] mirrored {total["copied"]} file(s) '
               f'({total["bytes"] / 1048576:.0f} MB) for '
               f'{total["fires"]} fire(s)')
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
    return total


def mirror_in_background(delay_s: float = 2.0) -> bool:
    """Mirror after a short delay, coalescing repeated requests."""
    global _mirror_thread
    with _mirror_lock:
        if _mirror_thread is not None and _mirror_thread.is_alive():
            return False

        def _run():
            time.sleep(max(0.0, delay_s))
            try:
                mirror_all()
            except Exception as exc:
                sys.stderr.write(f'[durable] mirror failed: {exc}\n')

        _mirror_thread = threading.Thread(target=_run, daemon=True,
                                          name='durable-mirror')
        _mirror_thread.start()
        return True


# ------------------------------------------------------------ restoring

def restore_stack(ram_path: str, log=None) -> bool:
    """Bring one stack back from the durable store to the ramdisk.

    Called before a rebuild is attempted: copying a file back costs
    seconds, rebuilding costs minutes and, for a dated composite whose
    mosaic has rolled off, is impossible.
    """
    dest = store_dir()
    if not dest or not ram_path:
        return False
    base = os.path.basename(ram_path)
    src = os.path.join(dest, base)
    if not os.path.isfile(src):
        return False
    try:
        os.makedirs(os.path.dirname(ram_path), exist_ok=True)
        for sfx in _STACK_SUFFIXES:
            stem = base[:-4] if base.endswith('.bin') else base
            s = os.path.join(dest, stem + sfx) if sfx != '.bin' \
                else os.path.join(dest, stem + '.bin')
            d = os.path.join(os.path.dirname(ram_path),
                             os.path.basename(s))
            if os.path.isfile(s) and not os.path.isfile(d):
                tmp = d + '.part'
                shutil.copy2(s, tmp)
                os.replace(tmp, d)
        msg = f'[durable] restored {base} from the durable store'
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
        return os.path.isfile(ram_path)
    except OSError as exc:
        sys.stderr.write(f'[durable] restore {base}: {exc}\n')
        return False


def restore_fire(fire, log=None) -> int:
    """Bring back every durable stack belonging to *fire*."""
    dest = store_dir()
    cb = getattr(fire, 'crop_bin', '') or ''
    if not dest or not cb:
        return 0
    import re
    m = re.match(r'^\d{8}_stack_(?P<safe>.+?)_(?P<h>[0-9a-fA-F]{6,})'
                 r'(_l2(_d\d{8})?)?\.bin$', os.path.basename(cb))
    if not m:
        return 0
    ram = os.path.dirname(cb)
    n = 0
    for src in sorted(glob.glob(os.path.join(
            dest, f'*_stack_{m.group("safe")}_{m.group("h")}*'))):
        d = os.path.join(ram, os.path.basename(src))
        if os.path.isfile(d):
            continue
        try:
            tmp = d + '.part'
            shutil.copy2(src, tmp)
            os.replace(tmp, d)
            n += 1
        except OSError as exc:
            sys.stderr.write(f'[durable] restore: {exc}\n')
    if n:
        msg = (f'[durable] {fire.fire_numbe}: restored {n} file(s) '
               f'from the durable store')
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
    return n


# ------------------------------------------------------------- recovery

def _bbox_from_sidecar(path: str):
    """(xmin, ymin, xmax, ymax) from a sidecar's grid, or None."""
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        gt = d.get('gt')
        w = int(d.get('width') or 0)
        h = int(d.get('height') or 0)
        if not (isinstance(gt, (list, tuple)) and len(gt) == 6
                and w > 0 and h > 0):
            return None
        x0, px, _a, y0, _b, py = (float(v) for v in gt)
        x1 = x0 + px * w
        y1 = y0 + py * h
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    except (OSError, ValueError, TypeError):
        return None


def recover_identity(fire, log=None) -> bool:
    """Put back a bounding box and date range lost from the record.

    The grid is written into every ``_overlays.json`` beside a stack,
    and into the preview ``geo.json`` in the fire's cache. Either gives
    the bounding box exactly. Without this a fire whose imagery and
    grid are both on disk still could not be re-prepared, and had to be
    re-created by hand.
    """
    changed = []

    if not getattr(fire, 'bbox_native', None):
        candidates = []
        cb = getattr(fire, 'crop_bin', '') or ''
        safe = ''
        import re
        m = re.match(r'^\d{8}_stack_(?P<safe>.+?)_[0-9a-fA-F]{6,}',
                     os.path.basename(cb))
        if m:
            safe = m.group('safe')
        for root in (os.path.dirname(cb) or '/ram', store_dir()):
            if not root or not os.path.isdir(root):
                continue
            pat = (f'*_stack_{safe}_*_overlays.json' if safe
                   else '*_overlays.json')
            candidates.extend(sorted(glob.glob(os.path.join(root, pat)),
                                     reverse=True))
        cache = getattr(fire, 'cache_dir', '') or ''
        if cache and os.path.isdir(cache):
            candidates.extend(sorted(glob.glob(os.path.join(
                cache, 'previews*', 'geo.json')), reverse=True))
        for c in candidates:
            bbox = _bbox_from_sidecar(c)
            if bbox:
                fire.bbox_native = bbox
                changed.append(f'bbox from {os.path.basename(c)}')
                break

    # A date range can be reconstructed from the year: the accumulation
    # window has always run from 1 January to the imagery date, which is
    # what the header shows. Better a usable default than a fire that
    # cannot be prepared at all.
    year = getattr(fire, 'fire_year', 0) or 0
    if year and not getattr(fire, 'viirs_start_date', ''):
        fire.viirs_start_date = f'{int(year)}0101'
        changed.append('start date from the fire year')
    if year and not getattr(fire, 'viirs_end_date', ''):
        end = ''
        cb = getattr(fire, 'crop_bin', '') or ''
        base = os.path.basename(cb)
        if len(base) >= 8 and base[:8].isdigit():
            end = base[:8]
        if not end:
            end = time.strftime('%Y%m%d')
        fire.viirs_end_date = end
        changed.append(f'end date {end}')

    if changed:
        msg = (f'[recover] {fire.fire_numbe}: '
               + '; '.join(changed))
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
        return True
    return False


def recover_all(log=None) -> int:
    """Recover identity for every fire missing it. Returns the count."""
    try:
        with state.lock:
            fires = list(state.fires.values())
    except Exception:
        return 0
    n = 0
    for fire in fires:
        need = (not getattr(fire, 'bbox_native', None)
                or not getattr(fire, 'viirs_start_date', '')
                or not getattr(fire, 'viirs_end_date', ''))
        if need and recover_identity(fire, log=log):
            n += 1
    if n:
        try:
            from .persistence import _save_fire_state
            _save_fire_state()
        except Exception:
            pass
    return n
