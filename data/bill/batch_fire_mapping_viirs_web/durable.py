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

def fire_prefix(fire) -> str:
    """The `<name>_<hash>` that every one of this fire's files carries.

    Derived from the fire's NAME, not from crop_bin.

    crop_bin is empty on a fire that failed to load and stale on one
    whose paths moved -- precisely the fires that need mirroring and
    restoring most. Deriving the prefix from it meant those fires were
    silently skipped: nothing was copied to disk, so nothing could be
    restored, so every restart rebuilt them and reported "being
    prepared" for imagery that was supposed to be safe.
    """
    try:
        from .aoi_stack import aoi_identity_hash, sanitize_identifier
        safe = sanitize_identifier(fire.fire_numbe)
        h = aoi_identity_hash(fire.fire_numbe,
                              getattr(state, 'shared_root', '') or '')
        return f'{safe}_{h}'
    except Exception as exc:
        sys.stderr.write(f'[durable] cannot derive prefix for '
                         f'{getattr(fire, "fire_numbe", "?")}: {exc}\n')
        return ''


def ram_dir_for(fire) -> str:
    """Where this fire's stacks live on the ramdisk."""
    cb = getattr(fire, 'crop_bin', '') or ''
    d = os.path.dirname(cb)
    if d and os.path.isdir(d):
        return d
    try:
        from .aoi_stack import RAM_DIR
        return RAM_DIR
    except Exception:
        return '/ram'


def _fire_stack_glob(fire) -> str:
    """Glob matching every stack file belonging to *fire*."""
    pfx = fire_prefix(fire)
    if not pfx:
        return ''
    return os.path.join(ram_dir_for(fire), f'*_stack_{pfx}*')


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

        # Never mirror a stack that is still being written.
        #
        # The destination was already written via a temporary name, so
        # a half-written MIRROR was impossible -- but a half-written
        # SOURCE was not. Copying a stack mid-build produced a
        # plausible-looking file that GDAL then refused as "not
        # recognized as being in a supported file format", and the next
        # start-up restored that corruption over the ramdisk.
        if src.endswith('.bin'):
            try:
                from .aoi_stack import stack_is_valid
                if not stack_is_valid(src):
                    out['skipped'] += 1
                    continue
            except Exception:
                pass

        dst = os.path.join(dest, os.path.basename(src))
        try:
            ssize = os.path.getsize(src)
            smtime = os.path.getmtime(src)
            if os.path.isfile(dst) and os.path.getsize(dst) == ssize:
                out['skipped'] += 1
                continue
            tmp = dst + '.part'
            shutil.copy2(src, tmp)
            # Did the source change under us while we copied? Then what
            # we have is a torn read; discard it rather than publish it.
            if (os.path.getsize(src) != ssize
                    or os.path.getmtime(src) != smtime):
                os.remove(tmp)
                out['skipped'] += 1
                sys.stderr.write(
                    f'[durable] {os.path.basename(src)} changed during '
                    f'the copy; not mirrored this time\n')
                continue
            os.replace(tmp, dst)
            out['copied'] += 1
            out['bytes'] += ssize
        except OSError as exc:
            out['failed'] += 1
            sys.stderr.write(f'[durable] {os.path.basename(src)}: '
                             f'{exc}\n')
    sys.stderr.write(
        '[persist] mirror %s: %d copied, %d skipped, %d failed '
        '(%.0f MB)\n'
        % (getattr(fire, 'fire_numbe', '?'), out['copied'],
           out['skipped'], out['failed'], out['bytes'] / 1048576.0))
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
        # Re-check membership: this list was taken when the mirror
        # started, and a fire deleted since must not have its files
        # copied back to the durable store after the purge.
        try:
            if fire.fire_numbe not in state.fires:
                continue
        except Exception:
            pass
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

def _grid_of(path: str):
    """(width, height, gt) for a stack, read from its sidecar.

    The identity hash covers the fire's NAME and the server instance,
    not its bounding box -- so a fire deleted and recreated under the
    same name shares the hash with its predecessor's stacks. The grid
    does not lie: a stack whose size and geotransform match the AOI now
    on the books IS that AOI's, and one that does not belongs to a
    different incident that merely shared a name.
    """
    # The RASTER is the authority on its own grid.
    #
    # The sidecar is written beside the stack and is not always
    # rewritten when the stack is: a retired-and-rebuilt product keeps
    # the old _overlays.json, so the enumeration judged a perfectly
    # good new stack by the grid of the one it replaced -- and withheld
    # it from the selector with no error anywhere, because the build
    # itself had succeeded. Read the file when it is there.
    if path.endswith('.bin') and os.path.isfile(path):
        try:
            from osgeo import gdal
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is not None:
                gt = ds.GetGeoTransform()
                w, h = ds.RasterXSize, ds.RasterYSize
                ds = None
                if gt and w > 0 and h > 0:
                    return w, h, [float(v) for v in gt]
        except Exception:
            pass

    stem = os.path.splitext(path)[0]
    for side in (stem + '_overlays.json', stem + '_dates.json'):
        try:
            with open(side, encoding='utf-8') as f:
                d = json.load(f)
            gt = d.get('gt')
            w = int(d.get('width') or 0)
            h = int(d.get('height') or 0)
            if w > 0 and h > 0 and isinstance(gt, (list, tuple)) \
                    and len(gt) == 6:
                return w, h, [float(v) for v in gt]
        except (OSError, ValueError, TypeError):
            continue
    return None


def grids_match(a, b, tol: float = 0.51) -> bool:
    """Same raster grid? Sizes exactly, origin and pixel within tol."""
    if not a or not b:
        return False
    if a[0] != b[0] or a[1] != b[1]:
        return False
    ga, gb = a[2], b[2]
    if abs(ga[1] - gb[1]) > 1e-6 or abs(ga[5] - gb[5]) > 1e-6:
        return False
    return abs(ga[0] - gb[0]) <= tol and abs(ga[3] - gb[3]) <= tol


def grid_matches_bbox(fire, grid, slack_px: float = 1.5) -> bool:
    """Does this grid actually cover the fire's recorded AOI?

    The authoritative answer, because it comes from bbox_native --
    which the operator drew and which every other check derives from.
    A stack built while the bounding box was wrong carries a perfectly
    correct FILENAME and completely different ground; only the
    geotransform gives it away.
    """
    bb = getattr(fire, 'bbox_native', None)
    if not bb or not grid:
        return True                 # nothing to check against
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in bb)
        w, h, gt = grid
        px = abs(float(gt[1])) or 20.0
        py = abs(float(gt[5])) or px
        tol_x, tol_y = px * slack_px, py * slack_px
        if abs(float(gt[0]) - xmin) > tol_x:
            return False
        if abs(float(gt[3]) - ymax) > tol_y:
            return False
        if abs(w * px - (xmax - xmin)) > tol_x * 2:
            return False
        if abs(h * py - (ymax - ymin)) > tol_y * 2:
            return False
        return True
    except (TypeError, ValueError):
        return True


def reference_grid(fire):
    """The grid this fire's AOI is on now, or None if unknowable."""
    cb = getattr(fire, 'crop_bin', '') or ''
    if cb:
        g = _grid_of(cb)
        if g:
            return g
    cache = getattr(fire, 'cache_dir', '') or ''
    if cache:
        for gj in sorted(glob.glob(os.path.join(
                cache, 'previews*', 'geo.json')), reverse=True):
            try:
                with open(gj, encoding='utf-8') as f:
                    d = json.load(f)
                gt = d.get('gt')
                w = int(d.get('width') or 0)
                h = int(d.get('height') or 0)
                if w > 0 and h > 0 and gt and len(gt) == 6:
                    return w, h, [float(v) for v in gt]
            except (OSError, ValueError, TypeError):
                continue
    return None


def durable_products(fire) -> list:
    """Stacks in the durable store that belong to THIS AOI.

    Grid-checked, so a same-named predecessor's layers are never
    adopted. Returns absolute paths inside the store.

    When the grid cannot be established at all -- a fire with no
    readable stack and no preview geo.json -- nothing is returned
    rather than everything: recovering the wrong incident's imagery is
    far worse than recovering none.
    """
    dest = store_dir()
    pref = fire_prefix(fire)
    if not dest or not pref or not os.path.isdir(dest):
        return []
    ref = reference_grid(fire)
    if not ref:
        sys.stderr.write(
            '[persist] %s: NO REFERENCE GRID (crop_bin=%r) -- not '
            'adopting any durable stack, because nothing can prove '
            'they belong to this AOI. Every dated product will be '
            'missing from the selector until this fire has one '
            'readable stack with a sidecar.\n'
            % (getattr(fire, 'fire_numbe', '?'),
               getattr(fire, 'crop_bin', '')))
        return []
    out, rejected = [], 0
    for cand in sorted(glob.glob(os.path.join(
            dest, '*_stack_%s*.bin' % pref)), reverse=True):
        base = os.path.basename(cand)
        if '.kgc' in base:
            continue

        # Say WHY, for every candidate.
        #
        # A product silently missing from the selector is impossible to
        # diagnose from the outside: the file is on disk and the menu
        # does not list it. One line per decision turns that into a
        # five-second answer.
        # Either ENVI header convention counts: <stem>.hdr or
        # <name>.bin.hdr. Both are present in this data, and a raster
        # written with the second was being treated as headerless.
        _h = (os.path.splitext(cand)[0] + '.hdr')
        if not os.path.isfile(_h) and os.path.isfile(cand + '.hdr'):
            _h = cand + '.hdr'
        if not os.path.isfile(_h):
            rejected += 1
            sys.stderr.write(
                '[persist] %s: skip %s -- no .hdr beside it\n'
                % (getattr(fire, 'fire_numbe', '?'), base))
            continue
        g = _grid_of(cand)
        if not g:
            rejected += 1
            sys.stderr.write(
                '[persist] %s: skip %s -- no sidecar, so its grid is '
                'unknown\n' % (getattr(fire, 'fire_numbe', '?'), base))
            continue
        if not grids_match(ref, g):
            rejected += 1
            sys.stderr.write(
                '[persist] %s: skip %s -- grid %dx%d at (%.1f, %.1f) '
                'differs from this AOI %dx%d at (%.1f, %.1f)\n'
                % (getattr(fire, 'fire_numbe', '?'), base,
                   g[0], g[1], g[2][0], g[2][3],
                   ref[0], ref[1], ref[2][0], ref[2][3]))
            continue
        if not grid_matches_bbox(fire, g):
            rejected += 1
            _bb = getattr(fire, 'bbox_native', None) or (0, 0, 0, 0)
            sys.stderr.write(
                '[persist] %s: skip %s -- grid origin (%.1f, %.1f) '
                'does not cover the recorded bbox (%.1f, %.1f, %.1f, '
                '%.1f)\n'
                % (getattr(fire, 'fire_numbe', '?'), base,
                   g[2][0], g[2][3], _bb[0], _bb[1], _bb[2], _bb[3]))
            continue
        out.append(cand)
    sys.stderr.write(
        '[persist] %s: %d durable product(s) match this AOI, '
        '%d rejected\n'
        % (getattr(fire, 'fire_numbe', '?'), len(out), rejected))
    return out


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
        # Validate what we just put back. A durable copy made by an
        # older build could itself be torn; restoring it would turn one
        # bad file into two.
        try:
            from .aoi_stack import stack_is_valid
            if not stack_is_valid(ram_path):
                sys.stderr.write(
                    f'[durable] restored {base} is not readable; '
                    f'discarding it so the builder makes a fresh one\n')
                for sfx in ('.bin', '.hdr'):
                    bad = os.path.splitext(ram_path)[0] + sfx
                    try:
                        os.remove(bad)
                    except OSError:
                        pass
                # The durable copy is bad too -- drop it, or every
                # restart repeats this.
                try:
                    os.remove(src)
                except OSError:
                    pass
                return False
        except Exception:
            pass
        msg = f'[durable] restored {base} from the durable store'
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
        return os.path.isfile(ram_path)
    except OSError as exc:
        sys.stderr.write(f'[durable] restore {base}: {exc}\n')
        return False


def restore_fire(fire, log=None) -> int:
    """Bring back every durable stack belonging to *fire*.

    Also repairs ``crop_bin`` when it points at a file that no longer
    exists: a fire whose stacks were restored but whose pointer still
    named a purged path would be treated as having no imagery at all.
    """
    dest = store_dir()
    pfx = fire_prefix(fire)
    if not dest or not pfx or not os.path.isdir(dest):
        return 0
    ram = ram_dir_for(fire)

    # Only stacks whose GRID matches this AOI.
    #
    # The identity hash covers the fire's name and the server instance,
    # not its bounding box, so a fire deleted and recreated under the
    # same name shares the hash with its predecessor's stacks.
    # Restoring those would silently hand this AOI another incident's
    # imagery. The grid settles it.
    keep = {os.path.splitext(os.path.basename(p))[0]
            for p in durable_products(fire)}

    n = 0
    for src in sorted(glob.glob(os.path.join(dest, f'*_stack_{pfx}*'))):
        base = os.path.basename(src)
        stem = base
        for sfx in ('.bin', '.hdr', '_dates.json', '_overlays.json'):
            if stem.endswith(sfx):
                stem = stem[:-len(sfx)]
                break
        if keep and stem not in keep:
            continue
        d = os.path.join(ram, base)
        if os.path.isfile(d):
            continue
        try:
            os.makedirs(ram, exist_ok=True)
            tmp = d + '.part'
            shutil.copy2(src, tmp)
            os.replace(tmp, d)
            n += 1
        except OSError as exc:
            sys.stderr.write(f'[durable] restore: {exc}\n')

    # Point the fire at something real.
    cb = getattr(fire, 'crop_bin', '') or ''
    if not cb or not os.path.isfile(cb):
        cands = [c for c in sorted(glob.glob(os.path.join(
            ram, f'*_stack_{pfx}*.bin')), reverse=True)
            if '.kgc' not in os.path.basename(c)
            and os.path.isfile(os.path.splitext(c)[0] + '.hdr')]
        if cands:
            fire.crop_bin = cands[0]
            msg = (f'[durable] {fire.fire_numbe}: crop_bin pointed at a '
                   f'missing file; using '
                   f'{os.path.basename(cands[0])}')
            sys.stderr.write(msg + '\n')
            if log:
                log(msg)

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

        # The fire's OWN name decides which sidecars may be read.
        #
        # Never a wildcard. An earlier version derived this from
        # crop_bin and fell back to '*_overlays.json' when crop_bin was
        # empty -- which is precisely the state of a fire that needs
        # recovering. That matched every fire's sidecars and took the
        # newest, so several distinct incidents were all given one
        # another's bounding box. Recovering nothing is correct when
        # the fire's own files are absent; recovering somebody else's
        # AOI is not.
        # The FULL prefix: sanitized name AND identity hash.
        #
        # fire_prefix() gives '<safe>_<hash>'. Using the name alone
        # matched longer names that merely start the same way, so
        # K51490 could adopt K51490_ash's grid.
        safe = fire_prefix(fire)
        if not safe:
            sys.stderr.write(
                f'[recover] {getattr(fire, "fire_numbe", "?")}: cannot '
                f'determine its file prefix; not guessing a bbox\n')
            return False

        for root in (os.path.dirname(cb) or '/ram', store_dir()):
            if not root or not os.path.isdir(root):
                continue
            # Anchored on the full prefix, hash included.
            #
            # '*_stack_K51490_*' also matches K51490_ash -- a different
            # incident whose name merely starts the same way. Recovery
            # would then hand this fire the other one's bounding box,
            # and every product built afterwards would cover the wrong
            # ground under the right name.
            candidates.extend(sorted(glob.glob(os.path.join(
                root, f'*_stack_{safe}_*_overlays.json')), reverse=True))
            candidates.extend(sorted(glob.glob(os.path.join(
                root, f'*_stack_{safe}_overlays.json')), reverse=True))
        # The fire's own cache directory -- per fire by construction.
        cache = getattr(fire, 'cache_dir', '') or ''
        if cache and os.path.isdir(cache):
            candidates.extend(sorted(glob.glob(os.path.join(
                cache, 'previews*', 'geo.json')), reverse=True))
        for c in candidates:
            bbox = _bbox_from_sidecar(c)
            if bbox:
                # Sanity: a recovered box must be a real extent.
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w <= 0 or h <= 0 or w > 2.0e6 or h > 2.0e6:
                    sys.stderr.write(
                        f'[recover] {fire.fire_numbe}: ignoring '
                        f'implausible bbox {bbox} from '
                        f'{os.path.basename(c)}\n')
                    continue
                fire.bbox_native = bbox
                # WGS84 copy is derived from the native box elsewhere;
                # a stale one would disagree with what we just set.
                try:
                    fire.bbox_wgs84 = None
                except Exception:
                    pass
                changed.append(
                    f'bbox {tuple(round(v) for v in bbox)} from '
                    f'{os.path.basename(c)}')
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


def revive_fires(log=None) -> int:
    """Clear a stale error once a fire's imagery is back.

    A fire that failed to prepare keeps ERROR in the saved state. After
    its stacks have been restored the status is simply out of date, and
    leaving it red means an operator is told to re-create a fire whose
    data is sitting there. Only fires that now have a readable stack
    are revived, and only from ERROR -- nothing else is touched.
    """
    from .state import FireStatus
    n = 0
    try:
        with state.lock:
            fires = list(state.fires.values())
    except Exception:
        return 0
    for fire in fires:
        if getattr(fire, 'status', None) != FireStatus.ERROR:
            continue
        cb = getattr(fire, 'crop_bin', '') or ''
        if not cb or not os.path.isfile(cb):
            continue
        if not os.path.isfile(os.path.splitext(cb)[0] + '.hdr'):
            continue
        fire.status = FireStatus.READY
        try:
            fire.error_msg = ''
        except Exception:
            pass
        n += 1
        msg = (f'[recover] {fire.fire_numbe}: imagery is present; '
               f'clearing the error state')
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
    if n:
        try:
            from .persistence import _save_fire_state
            _save_fire_state()
        except Exception:
            pass
    return n


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


# ---------------------------------------------------------------
# Orphaned stack sets
# ---------------------------------------------------------------
def _identity_prefixes(state) -> dict:
    """``{<safe>_<hash>: fire name}`` for every fire that exists.

    Includes the empty-instance form as well as the real one, because
    two call sites built stacks without an instance key for months and
    those files belong to the fire even though nothing else knows it.
    """
    import re as _re
    from .aoi_stack import aoi_identity_hash, sanitize_identifier
    inst = getattr(state, 'shared_root', '') or ''
    out = {}
    for name in list(getattr(state, 'fires', {}) or {}):
        safe = sanitize_identifier(name)
        out[f'{safe}_{aoi_identity_hash(name, inst)}'] = name
        out[f'{safe}_{aoi_identity_hash(name, "")}'] = name
    return out


def orphan_stack_sets(state) -> list:
    """Stack sets on disk that belong to no fire that still exists.

    Returns one entry per ``<safe>_<hash>`` prefix, with its files and
    total size. Purely a report: nothing is removed here, and nothing
    calls this on a timer. Deleting a fire has never removed its
    imagery, so a long-lived server accumulates these -- 62 GB of them
    on the machine where this was written, including clustering graphs
    of several GB each.
    """
    import glob as _g
    import re as _re
    owned = _identity_prefixes(state)
    roots = []
    d = store_dir()
    if d:
        roots.append(d)
    for f in (getattr(state, 'fires', {}) or {}).values():
        r = os.path.dirname(getattr(f, 'crop_bin', '') or '')
        if r and r not in roots:
            roots.append(r)
    if not roots:
        return []
    groups = {}
    for root in roots:
        for path in _g.glob(os.path.join(root, '*_stack_*')):
            m = _re.match(r'^\d{8}_stack_(.+?_[0-9a-fA-F]{6,})',
                          os.path.basename(path))
            if not m:
                continue
            groups.setdefault(m.group(1), []).append(path)
    out = []
    for prefix, files in sorted(groups.items()):
        if prefix in owned:
            continue
        size = 0
        newest = 0.0
        for p in files:
            try:
                size += os.path.getsize(p)
                newest = max(newest, os.path.getmtime(p))
            except OSError:
                pass
        out.append({'prefix': prefix, 'files': len(files),
                    'bytes': size, 'newest': newest,
                    'paths': sorted(files)})
    return out


def purge_orphan_stack_sets(state, prefixes) -> dict:
    """Delete the named orphan sets. Explicit prefixes only.

    Refuses anything that currently belongs to a fire, re-deriving
    ownership at the moment of deletion rather than trusting the list
    the caller was shown -- a fire may have been created in between,
    and a stale list must never delete live imagery.
    """
    owned = _identity_prefixes(state)
    orphans = {o['prefix']: o for o in orphan_stack_sets(state)}
    removed, freed, refused = [], 0, []
    for prefix in list(prefixes or []):
        if prefix in owned:
            refused.append(f'{prefix}: belongs to {owned[prefix]}')
            continue
        o = orphans.get(prefix)
        if not o:
            refused.append(f'{prefix}: not an orphan set on disk')
            continue
        for path in o['paths']:
            try:
                freed += os.path.getsize(path)
            except OSError:
                pass
            try:
                os.remove(path)
                removed.append(path)
            except OSError as exc:
                sys.stderr.write(f'[orphans] {path}: {exc}\n')
        sys.stderr.write(
            '[orphans] purged %s: %d file(s)\n' % (prefix, len(o['paths'])))
    return {'removed': len(removed), 'freed_mb': round(freed / 1048576, 1),
            'refused': refused}
