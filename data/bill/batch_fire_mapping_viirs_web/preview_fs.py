"""Race-free writes into preview directories.

A fire's preview images live in ``<cache>/previews`` (what the panes show
now) and ``<cache>/previews_<product>`` (one stash per product). Several
threads touch them at once: the creation worker, a source switch, the
preview warming queue and the on-demand renderer. Two things went wrong:

  * A writer created its temp file INSIDE the target directory and then
    renamed it into place. When another thread deleted or replaced that
    directory in between -- a switch clearing ``previews/``, a stash
    being refreshed, a stale stash being dropped -- the temp file went
    with it and the rename failed with FileNotFoundError. Unique temp
    names could not help: the directory itself was gone.
  * A render for product A could finish after a switch had moved
    ``previews/`` to product B, and land A's pixels in B's directory.

The fix has three parts, all here:

  * Renders write to ``<cache>/.render_tmp``, which no preview operation
    ever deletes, so an in-flight render cannot lose its temp file.
  * One short lock per fire cache. Every delete/replace of a preview
    directory and every final rename into one holds it, so a directory
    never disappears between the check and the rename. It is only ever
    held around filesystem operations, never while rendering or
    building, so nothing waits on it for long and it cannot deadlock
    with the renderer's own thread pool.
  * The live ``previews/`` directory has an owner token. A render into
    it takes ownership at the start; anything that clears or replaces
    the directory revokes it; a commit whose token has been revoked is
    discarded instead of written.
"""

import json
import os
import sys
import threading
import time
import uuid

_locks = {}
_locks_guard = threading.Lock()

# abs path of a live previews/ directory -> (product key, token)
_live_owner = {}

_SCRATCH = '.render_tmp'
_SCRATCH_MAX_AGE_S = 3600.0


def _abs(p: str) -> str:
    return os.path.abspath(p or '.')


def cache_dir_of(pdir: str) -> str:
    """The fire cache a preview directory belongs to."""
    pdir = _abs(pdir)
    if os.path.basename(pdir).startswith('previews'):
        return os.path.dirname(pdir)
    return pdir


def is_live_dir(pdir: str) -> bool:
    return os.path.basename(_abs(pdir)) == 'previews'


def previews_lock(cache_dir: str) -> threading.RLock:
    """The one lock for every preview directory of one fire."""
    key = _abs(cache_dir)
    with _locks_guard:
        lk = _locks.get(key)
        if lk is None:
            lk = threading.RLock()
            _locks[key] = lk
        return lk


def lock_for(pdir: str) -> threading.RLock:
    return previews_lock(cache_dir_of(pdir))


def read_stamp(pdir: str) -> str:
    """Which product a preview directory was rendered from ('' if unmarked)."""
    try:
        with open(os.path.join(pdir, '.product'), encoding='utf-8') as fh:
            return fh.read().strip()
    except OSError:
        return ''


def _write_stamp_locked(pdir: str, key: str) -> None:
    tmp = os.path.join(pdir, '.product.%d.%d' % (os.getpid(),
                                                  threading.get_ident()))
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(key)
    os.replace(tmp, os.path.join(pdir, '.product'))


def scratch_path(target: str, suffix: str = '') -> str:
    """A temp path for *target* that no preview operation can delete.

    Same filesystem as the target (inside the fire cache), so the final
    os.replace is still an atomic rename.
    """
    d = os.path.join(cache_dir_of(os.path.dirname(target)), _SCRATCH)
    os.makedirs(d, exist_ok=True)
    _sweep_scratch(d)
    return os.path.join(d, '%s.%d.%d.%s%s' % (
        os.path.basename(target), os.getpid(), threading.get_ident(),
        uuid.uuid4().hex[:8], suffix))


_last_sweep = {}


def _sweep_scratch(d: str) -> None:
    """Remove temp files a killed process left behind (at most hourly)."""
    now = time.time()
    if now - _last_sweep.get(d, 0.0) < 600.0:
        return
    _last_sweep[d] = now
    try:
        for name in os.listdir(d):
            p = os.path.join(d, name)
            try:
                if now - os.path.getmtime(p) > _SCRATCH_MAX_AGE_S:
                    os.remove(p)
            except OSError:
                pass
    except OSError:
        pass


def _discard(tmp: str) -> None:
    for p in (tmp, tmp + '.aux.xml'):
        try:
            os.remove(p)
        except OSError:
            pass


def commit(tmp: str, target: str, token=None, who: str = '') -> bool:
    """Rename *tmp* onto *target*, unless its directory moved on.

    Refused (and the temp file removed) when the target directory no
    longer exists -- it was deliberately cleared or replaced -- or, for
    the live directory, when *token* is given and no longer owns it.
    Returns True when written.
    """
    pdir = os.path.dirname(target)
    with lock_for(pdir):
        why = ''
        if not os.path.isdir(pdir):
            why = 'its directory was cleared meanwhile'
        elif token is not None and is_live_dir(pdir):
            owner = _live_owner.get(_abs(pdir))
            if not owner or owner[1] != token:
                why = ('the live previews moved to %s meanwhile'
                       % (owner[0] if owner else 'another product'))
        if why:
            _discard(tmp)
            sys.stderr.write('[preview_fs] discarded %s%s: %s\n' % (
                os.path.basename(target), (' (%s)' % who) if who else '',
                why))
            return False
        os.replace(tmp, target)
        return True


def begin_live_render(pdir: str, key: str):
    """Claim the live directory for a render of *key*.

    Returns a token, or None when the live directory is currently
    showing a DIFFERENT product -- the caller must then render into that
    product's stash instead of overwriting what is on screen.
    """
    pdir = _abs(pdir)
    with lock_for(pdir):
        os.makedirs(pdir, exist_ok=True)
        stamp = read_stamp(pdir)
        if stamp and key and stamp != key:
            return None
        token = uuid.uuid4().hex
        _live_owner[pdir] = (key or '', token)
        return token


def end_live_render(pdir: str, token, key: str, ok: bool) -> bool:
    """Release the claim; stamp the directory if the render still owns it."""
    pdir = _abs(pdir)
    with lock_for(pdir):
        owner = _live_owner.get(pdir)
        if not owner or owner[1] != token:
            return False
        _live_owner.pop(pdir, None)
        if ok and key and os.path.isdir(pdir):
            try:
                _write_stamp_locked(pdir, key)
            except OSError as exc:
                sys.stderr.write(f'[preview_fs] stamp failed: {exc}\n')
        return True


def revoke_live(pdir: str) -> None:
    """Anything that clears or replaces previews/ calls this (under lock)."""
    _live_owner.pop(_abs(pdir), None)


def rmtree(pdir: str, why: str = '') -> bool:
    """Delete a preview directory without pulling it from under a writer."""
    import shutil
    with lock_for(pdir):
        if is_live_dir(pdir):
            revoke_live(pdir)
        if not os.path.isdir(pdir):
            return False
        shutil.rmtree(pdir, ignore_errors=True)
        return not os.path.isdir(pdir)


def replace_tree(src: str, dst: str) -> None:
    """Make *dst* a copy of *src*, atomically with respect to writers."""
    import shutil
    with lock_for(dst):
        if is_live_dir(dst):
            revoke_live(dst)
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)


def merge_json(path: str, updates: dict = None, copy_from=None) -> bool:
    """Read-modify-write a small JSON map (geo.json) under the fire lock.

    Two renders recording different views used to read the same file,
    each add their own entry, and the second rename discarded the
    first's. Under the lock the merge is atomic. *copy_from* is an
    optional (src_key, dst_key) pair to duplicate an existing entry.
    Returns False when the directory is gone or the source entry for a
    copy is missing.
    """
    pdir = os.path.dirname(path)
    with lock_for(pdir):
        if not os.path.isdir(pdir):
            return False
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, encoding='utf-8') as fh:
                    data = json.load(fh) or {}
            except (OSError, ValueError):
                data = {}
        if copy_from:
            s_key, d_key = copy_from
            if s_key not in data:
                return False
            data[d_key] = dict(data[s_key])
        for k, v in (updates or {}).items():
            data[k] = v
        tmp = scratch_path(path, '.json')
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
        return True


def read_geo(pdir: str) -> dict:
    try:
        with open(os.path.join(pdir, 'geo.json'), encoding='utf-8') as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def geo_entry_matches(entry: dict, width: int, height: int, gt,
                      tol_px: float = 1e-6) -> bool:
    """Does a geo.json entry describe exactly this grid?"""
    try:
        px = abs(float(gt[1])) or 1.0
        py = abs(float(gt[5])) or 1.0
        g = [float(v) for v in entry.get('gt', [])]
        return (int(entry.get('rw', -1)) == int(width)
                and int(entry.get('rh', -1)) == int(height)
                and len(g) == 6
                and abs(g[0] - float(gt[0])) <= tol_px * px
                and abs(g[3] - float(gt[3])) <= tol_px * py
                and abs(abs(g[1]) - px) <= 1e-9
                and abs(abs(g[5]) - py) <= 1e-9)
    except Exception:
        return False
