"""An authoritative record of what belongs to each fire.

Everything this application creates for a fire -- stacks, previews,
hint masks, coverage sidecars, classification rasters, download
archives, clustering scratch -- is written here as it is made. The
manifest, not a filename pattern, is what deletion consults.

Why not globbing. Filenames carry a sanitized name and an identity
hash, and both are shared by more than one thing: ``K51490`` is a
prefix of ``K51490_ash``, and the hash covers the fire's NAME and the
server instance but not its bounding box, so a fire deleted and
recreated shares it with its predecessor. Every cross-fire fault in
this application has come from a pattern matching one file too many or
one too few. A recorded path cannot do that: it is either this fire's
or it is not, and the manifest says which.

The manifest lives beside the fire's cache on real disk, and a copy
goes to the durable store so it survives a lost ramdisk. It is rebuilt
from disk when absent -- see :func:`sync_from_disk` -- using the fire's
own identity prefix AND a grid check, so a migration adopts what is
genuinely this fire's and nothing else.
"""

import glob
import json
import os
import re
import sys
import threading
import time

_lock = threading.RLock()

# Artifact kinds. Recorded so deletion can report what it removed, and
# so a future retention policy can treat them differently.
KIND_STACK = 'stack'            # AOI raster + header + sidecars
KIND_PREVIEW = 'preview_dir'    # a previews_<product> directory
KIND_HINT = 'hint'              # derived hint mask
KIND_COVERAGE = 'coverage'      # per-product acquisition coverage
KIND_RESULT = 'result'          # classification raster / perimeter
KIND_DOWNLOAD = 'download'      # prepared archive
KIND_SCRATCH = 'scratch'        # clustering working directory
KIND_CACHE = 'cache_dir'        # the fire's cache directory itself
KIND_ACCEPTED = 'accepted_dir'  # the fire's deliverable directory

_ALL_KINDS = (KIND_STACK, KIND_PREVIEW, KIND_HINT, KIND_COVERAGE,
              KIND_RESULT, KIND_DOWNLOAD, KIND_SCRATCH, KIND_CACHE,
              KIND_ACCEPTED)


def manifest_path(fire) -> str:
    """Where a fire's manifest lives: beside its cache, on real disk."""
    cache = getattr(fire, 'cache_dir', '') or ''
    return os.path.join(cache, 'manifest.json') if cache else ''


def _empty(fire) -> dict:
    return {
        'fire_numbe': getattr(fire, 'fire_numbe', ''),
        'created_at': float(getattr(fire, 'created_at', 0) or 0),
        'bbox_native': list(getattr(fire, 'bbox_native', None) or []),
        'entries': [],
        'updated_at': time.time(),
        'version': 1,
    }


def load(fire) -> dict:
    p = manifest_path(fire)
    if not p or not os.path.isfile(p):
        return _empty(fire)
    try:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get('entries'), list):
            return d
    except (OSError, ValueError):
        pass
    return _empty(fire)


def save(fire, man: dict) -> bool:
    p = manifest_path(fire)
    if not p:
        return False
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        man['updated_at'] = time.time()
        man['fire_numbe'] = getattr(fire, 'fire_numbe', '')
        man['created_at'] = float(getattr(fire, 'created_at', 0) or 0)
        bb = getattr(fire, 'bbox_native', None)
        if bb:
            man['bbox_native'] = [float(v) for v in bb]
        tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(man, f, indent=1)
        os.replace(tmp, p)
        # Second copy in the durable store. Deliberately best-effort:
        # the SSD copy beside the cache is the authoritative one.
        try:
            mirror_to_store(fire)
        except Exception:
            pass
        return True
    except OSError as exc:
        sys.stderr.write(f'[manifest] could not save: {exc}\n')
        return False


def record(fire, kind: str, path: str, product: str = '') -> bool:
    """Note that *path* belongs to this fire. Idempotent."""
    if not path or kind not in _ALL_KINDS:
        return False
    with _lock:
        man = load(fire)
        rp = os.path.abspath(path)
        for e in man['entries']:
            if e.get('path') == rp:
                if product and not e.get('product'):
                    e['product'] = product
                    save(fire, man)
                return True
        man['entries'].append({
            'kind': kind, 'path': rp, 'product': product or '',
            'added_at': time.time(),
        })
        ok = save(fire, man)
    if ok:
        sys.stderr.write(
            '[manifest] %s: recorded %s %s\n'
            % (getattr(fire, 'fire_numbe', '?'), kind,
               os.path.basename(rp)))
    return ok


def record_many(fire, items) -> int:
    """items: iterable of (kind, path) or (kind, path, product)."""
    n = 0
    with _lock:
        man = load(fire)
        have = {e.get('path') for e in man['entries']}
        for it in items or []:
            kind, path = it[0], it[1]
            product = it[2] if len(it) > 2 else ''
            if not path or kind not in _ALL_KINDS:
                continue
            rp = os.path.abspath(path)
            if rp in have:
                continue
            have.add(rp)
            man['entries'].append({
                'kind': kind, 'path': rp, 'product': product,
                'added_at': time.time(),
            })
            n += 1
        if n:
            save(fire, man)
    return n


def paths(fire, kind: str = '') -> list:
    man = load(fire)
    return [e['path'] for e in man.get('entries', [])
            if e.get('path') and (not kind or e.get('kind') == kind)]


# --------------------------------------------------------- migration

def sync_from_disk(fire, state) -> int:
    """Adopt what is already on disk for a fire with no manifest.

    Deliberately conservative. A file is adopted only when it carries
    this fire's exact ``<safe>_<hash>`` prefix AND, for rasters, sits on
    a grid that covers this fire's recorded bounding box. Anything else
    is left alone: it is neither recorded, nor deleted, nor searched
    again. That is what keeps a predecessor of the same name out of
    this fire's record.
    """
    added = []
    try:
        from .durable import (fire_prefix, store_dir, _grid_of)
        from .bcws import bbox_covers_incident      # noqa: F401
    except Exception as exc:
        sys.stderr.write(f'[manifest] sync unavailable: {exc}\n')
        return 0

    pfx = fire_prefix(fire)
    if not pfx:
        return 0
    bb = getattr(fire, 'bbox_native', None)

    def _grid_ok(path):
        if not bb:
            return True
        try:
            from .durable import grid_matches_bbox
            g = _grid_of(path)
            return (not g) or grid_matches_bbox(fire, g)
        except Exception:
            return True

    # Stacks and their sidecars, on the ramdisk and in the store.
    roots = []
    cb = getattr(fire, 'crop_bin', '') or ''
    if cb:
        roots.append(os.path.dirname(cb))
    try:
        from .aoi_stack import RAM_DIR
        if RAM_DIR not in roots:
            roots.append(RAM_DIR)
    except Exception:
        pass
    sd = store_dir()
    if sd:
        roots.append(sd)

    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for cand in sorted(glob.glob(os.path.join(
                root, f'*_stack_{pfx}*'))):
            base = os.path.basename(cand)
            if '.kgc' in base:
                added.append((KIND_SCRATCH, cand))
                continue
            if base.endswith('.bin') and not _grid_ok(cand):
                sys.stderr.write(
                    '[manifest] %s: not adopting %s -- its grid does '
                    'not cover this AOI\n'
                    % (fire.fire_numbe, base))
                continue
            added.append((KIND_STACK, cand))

    # Everything under the fire's own cache directory.
    cache = getattr(fire, 'cache_dir', '') or ''
    if cache and os.path.isdir(cache):
        added.append((KIND_CACHE, cache))
        for name in sorted(os.listdir(cache)):
            full = os.path.join(cache, name)
            if name.startswith('previews'):
                added.append((KIND_PREVIEW, full,
                              name.replace('previews_', '', 1)))
            elif name == '_redwins':
                for h in sorted(glob.glob(os.path.join(full, '*'))):
                    added.append((KIND_HINT, h))
            elif name == 'coverage':
                for c in sorted(glob.glob(os.path.join(full, '*'))):
                    added.append((KIND_COVERAGE, c))
            elif name.endswith(('.bin', '.hdr', '.shp', '.dbf', '.shx',
                                '.prj', '.kml', '.csv', '.txt')):
                added.append((KIND_RESULT, full))

    # The deliverable directory.
    acc = getattr(fire, 'accepted_dir', '') or ''
    if acc and os.path.isdir(acc):
        added.append((KIND_ACCEPTED, acc))

    # Prepared download archives for this fire.
    try:
        root = getattr(state, 'output_root', '') or ''
        dc = os.path.join(root, '.download_cache')
        if os.path.isdir(dc):
            safe = pfx.rsplit('_', 1)[0]
            for z in sorted(glob.glob(os.path.join(dc, f'{safe}__*.zip'))):
                added.append((KIND_DOWNLOAD, z))
    except Exception:
        pass

    n = record_many(fire, added)
    if n:
        sys.stderr.write(
            '[manifest] %s: adopted %d existing item(s)\n'
            % (fire.fire_numbe, n))
    return n


def sync_all(state, log=None) -> int:
    total = 0
    try:
        with state.lock:
            fires = list(state.fires.values())
    except Exception:
        return 0
    for fire in fires:
        try:
            total += sync_from_disk(fire, state)
        except Exception as exc:
            sys.stderr.write(
                f'[manifest] {getattr(fire, "fire_numbe", "?")}: '
                f'sync failed: {exc}\n')
    if total and log:
        log(f'[manifest] adopted {total} existing item(s) across '
            f'{len(fires)} fire(s)')
    return total


# ---------------------------------------------------------- deletion

def purge(fire, log=None) -> dict:
    """Delete exactly what this fire's manifest lists.

    Returns counts by kind. Directories are removed whole; files
    individually. The manifest itself goes last, so an interrupted
    purge can be resumed rather than losing track of what remains.
    """
    import shutil
    stats = {'files': 0, 'dirs': 0, 'missing': 0, 'failed': 0}
    man = load(fire)
    entries = list(man.get('entries') or [])
    # Longest paths first: a file inside a directory we also remove
    # should not be reported as a failure afterwards.
    entries.sort(key=lambda e: len(e.get('path') or ''), reverse=True)
    for e in entries:
        p = e.get('path') or ''
        if not p:
            continue
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                stats['dirs'] += 1
            elif os.path.isfile(p):
                os.remove(p)
                stats['files'] += 1
            else:
                stats['missing'] += 1
        except OSError as exc:
            stats['failed'] += 1
            sys.stderr.write(f'[manifest] purge {p}: {exc}\n')
    msg = ('[manifest] %s: purged %d file(s), %d directory(ies); '
           '%d already gone, %d failed'
           % (getattr(fire, 'fire_numbe', '?'), stats['files'],
              stats['dirs'], stats['missing'], stats['failed']))
    sys.stderr.write(msg + '\n')
    if log:
        log(msg)
    return stats


# ------------------------------------------------- classification runs

def recover_serial_results(fire, state) -> int:
    """Rebuild the results list from this fire's own classification files.

    A classification can exist on disk while ``serial_results`` is
    empty -- the list lives in the fire record, and a record rebuilt
    from a partial save, or from a session where the run predated the
    list, has nothing in it. The pane still shows the raster, so the
    result is plainly there while the gallery says otherwise.

    Scoped to the fire's OWN cache directory, which is per fire by
    construction: no pattern is matched against a shared directory, so
    this cannot pick up a predecessor's or a neighbour's run.
    """
    cache = getattr(fire, 'cache_dir', '') or ''
    if not cache or not os.path.isdir(cache):
        return 0
    existing = list(getattr(fire, 'serial_results', None) or [])
    have_paths = {r.get('classified') for r in existing if r.get('classified')}
    name = getattr(fire, 'fire_numbe', '')
    added = 0

    # <fire>_serial_<n>_classified.bin, plus the canonical one.
    cands = []
    for f in sorted(os.listdir(cache)):
        if not f.endswith('_classified.bin'):
            continue
        if not f.startswith(f'{name}_'):
            continue                      # another record's file
        full = os.path.join(cache, f)
        m = re.match(rf'^{re.escape(name)}_serial_(\d+)_classified\.bin$',
                     f)
        run_id = int(m.group(1)) if m else 0
        cands.append((run_id, full))
    if not cands:
        return 0

    # Canonical (run_id 0) only counts when there is no serial run at
    # all: it is the same mask under its accepted name.
    if any(rid for rid, _ in cands):
        cands = [(rid, p) for rid, p in cands if rid]

    for run_id, path in sorted(cands):
        if path in have_paths:
            continue
        area = -1.0
        try:
            from .mapping import _compute_ml_area
            area = float(_compute_ml_area(fire, path))
        except Exception:
            try:
                from .prepare import _compute_ml_area as _cm
                area = float(_cm(fire, path))
            except Exception:
                area = -1.0
        entry = {
            'run_id': run_id or 1,
            'setting_idx': 0,
            'run_idx': 0,
            'setting_label': 'recovered',
            'agreement_pct': float(getattr(fire, 'agreement_pct', -1)
                                   or -1),
            'ml_area_ha': area,
            'error': '',
            'params': dict(getattr(fire, 'kgc_params', None) or {}),
            'is_previous': False,
            'classified': path,
            # Accepted if this is the mask the fire is delivering.
            'accepted': bool(
                getattr(fire, 'status', None) is not None
                and str(getattr(fire.status, 'value', '')) == 'accepted'),
        }
        existing.append(entry)
        added += 1
        sys.stderr.write(
            '[results] %s: recovered run %d from %s (%.2f ha)\n'
            % (name, entry['run_id'], os.path.basename(path), area))

    if added:
        fire.serial_results = existing
        record_many(fire, [(KIND_RESULT, r['classified'])
                           for r in existing if r.get('classified')])
        try:
            from .persistence import _save_fire_state
            _save_fire_state()
        except Exception:
            pass
    return added


def recover_all_results(state, log=None) -> int:
    total = 0
    try:
        with state.lock:
            fires = list(state.fires.values())
    except Exception:
        return 0
    for fire in fires:
        try:
            total += recover_serial_results(fire, state)
        except Exception as exc:
            sys.stderr.write(
                f'[results] {getattr(fire, "fire_numbe", "?")}: '
                f'recovery failed: {exc}\n')
    if total and log:
        log(f'[results] recovered {total} classification result(s)')
    return total


def stack_entries(fire) -> list:
    """Every stack path this fire has recorded, newest first.

    The manifest is written when a product is built, so it knows about
    products that a directory scan can miss -- a ramdisk cleared by a
    reboot, a durable copy not yet mirrored, a file moved by an
    operator. Enumeration consults it alongside the two directories,
    and the caller still applies the grid checks: being recorded proves
    the product was OURS, not that the file on disk today is usable.
    """
    out = []
    for e in load(fire).get('entries', []):
        if e.get('kind') != KIND_STACK:
            continue
        p = e.get('path') or ''
        if not p.endswith('.bin') or '.kgc' in os.path.basename(p):
            continue
        out.append(p)
    return sorted(set(out), reverse=True)


def mirror_to_store(fire) -> str:
    """Copy the manifest into the durable store.

    The manifest lives beside the fire's cache on the SSD. A second
    copy in the durable store means a fire can be reconstructed from
    the store alone, and makes it obvious where to look when the two
    disagree.
    """
    try:
        import shutil
        from .durable import store_dir, fire_prefix
        dest = store_dir()
        pfx = fire_prefix(fire)
        src = manifest_path(fire)
        if not dest or not pfx or not src or not os.path.isfile(src):
            return ''
        os.makedirs(dest, exist_ok=True)
        dst = os.path.join(dest, f'manifest_{pfx}.json')
        tmp = dst + '.part'
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst
    except Exception as exc:
        sys.stderr.write(f'[manifest] mirror failed: {exc}\n')
        return ''
