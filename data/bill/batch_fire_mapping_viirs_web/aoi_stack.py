"""aoi_stack.py — build the 12-band fire-mapping stack on demand, per AOI.

Background
----------
The pipeline used to pre-build one province-wide stack per night
(``fire_mapping_build_and_serve_stack.py``) and each fire's crop was cut
out of it. That stack was ~307 GB on the ramdisk, dwarfing the ~103 GB
source mosaics it was derived from, because every one of its 12 bands
spans the whole province at 20 m even though only a handful of small
AOIs are ever mapped.

This module removes the pre-built stack entirely. It generates the
*equivalent* product for a single AOI, reading only the AOI window out
of the two source mosaics:

    pre   = /data/mrap_bc/composite/median.bin   (fixed path)
    post  = /data/mrap_bc/<yyyymmdd>_mrap.bin    (latest by date prefix)

and computing the anomaly bands in numpy with the same formula
``sentinel2_anomaly3`` uses. The result is byte-for-byte the same kind
of raster the old province-wide stack would have yielded for that
window: same band order, same band-name convention, same BSQ float32
ENVI layout, same CRS/geotransform (offset to the window's origin).

Output goes to ``/ram/<postdate>_stack_<identifier>.bin`` and is
deliberately not backed up anywhere — see :func:`ensure_aoi_stack` for
the regeneration path when the ramdisk is cleared by a reboot.

Why the window read matters
---------------------------
``ReadAsArray(xoff, yoff, xsize, ysize)`` pulls only the requested
window off disk. A typical AOI is a few thousand pixels square, so each
band read is a few tens of MB against a 103 GB file, and peak memory is
bounded by the AOI, not by the mosaic. Reading whole bands and slicing
afterwards would defeat the entire point of this change.
"""

import errno
import hashlib
import os
import re
import sys
import time

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


# Fixed source locations. These match the paths that
# fire_mapping_build_and_serve_stack.py used, deliberately unchanged.
MRAP_DIR = '/data/mrap_bc'
COMPOSITE_DIR = os.path.join(MRAP_DIR, 'composite')
PRE_BIN = os.path.join(COMPOSITE_DIR, 'median.bin')
RAM_DIR = '/ram'

_MRAP_NAME_RE = re.compile(r'^(\d{8})_mrap\.bin$')

# Band-name prefixes, matching the province-wide stack's header exactly:
#   pre 20260501 20m: B12 2190nm MRAP
#   pst 20260803 20m: B12 2190nm MRAP
#   anomaly: B12 2190nm MRAP (post-pre)/(post+pre)
_ANOMALY_FORMULA = '(post-pre)/(post+pre)'
_ANOMALY_FORMULA_DIVIDE = 'post/pre'


class AoiStackError(RuntimeError):
    """Raised when the AOI stack cannot be built."""


# ----------------------------------------------------------------------
# Source discovery
# ----------------------------------------------------------------------

def find_latest_mrap(mrap_dir: str = MRAP_DIR):
    """Return ``(yyyymmdd, path)`` for the newest ``<date>_mrap.bin``.

    Selected by the filename's date prefix rather than mtime, matching
    ``fire_mapping_build_and_serve_stack.find_latest_mrap`` — regenerated
    files can land with out-of-order mtimes, so mtime would pick the
    wrong mosaic.
    """
    candidates = []
    try:
        names = os.listdir(mrap_dir)
    except OSError as exc:
        raise AoiStackError(f'cannot list {mrap_dir}: {exc}')
    for name in names:
        m = _MRAP_NAME_RE.match(name)
        if m:
            candidates.append((m.group(1), os.path.join(mrap_dir, name)))
    if not candidates:
        raise AoiStackError(
            f'no <yyyymmdd>_mrap.bin files found in {mrap_dir}')
    date_str, path = max(candidates, key=lambda pair: pair[0])
    if not os.path.isfile(_hdr_for(path)):
        raise AoiStackError(f'missing header for {path}')
    return date_str, path


def _hdr_for(bin_path: str) -> str:
    return os.path.splitext(bin_path)[0] + '.hdr'


def _parse_band_names(hdr_path: str):
    """Read the ``band names = {...}`` block out of an ENVI header.

    Tolerant of the newline-per-band layout these headers use and of a
    trailing comma before the closing brace.
    """
    try:
        with open(hdr_path, 'r', errors='replace') as f:
            text = f.read()
    except OSError:
        return []
    m = re.search(r'band\s+names\s*=\s*\{(.*?)\}', text,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    inner = m.group(1)
    names = [p.strip() for p in inner.split(',')]
    return [n for n in names if n]


def _after_last_colon(s: str) -> str:
    """Mirror of ``after_last_colon`` in sentinel2_anomaly3.cpp."""
    idx = s.rfind(':')
    return s if idx < 0 else s[idx + 1:].strip()


def _date_from_band_names(names, fallback: str = '') -> str:
    """Pull the yyyymmdd token out of a band name like
    ``pre 20260501 20m: B12 2190nm MRAP``.

    The pre-image's date is a property of the median composite, not
    something this module should invent, so it is read back rather than
    hardcoded.
    """
    for n in names:
        m = re.search(r'\b(\d{8})\b', n)
        if m:
            return m.group(1)
    return fallback


# ----------------------------------------------------------------------
# Window geometry
# ----------------------------------------------------------------------

def _window_for_bbox(gt, raster_w, raster_h, xmin, ymin, xmax, ymax):
    """Map a native-CRS bbox to an integer pixel window.

    Returns ``(xoff, yoff, xsize, ysize, window_gt)`` where ``window_gt``
    is the geotransform of the window itself (origin shifted to the
    window's top-left), so the output raster georeferences correctly on
    its own.

    The window is clipped to the raster. A bbox entirely outside the
    raster raises rather than silently producing an empty file.
    """
    px_w = gt[1]
    px_h = gt[5]          # normally negative (north-up)
    if px_w == 0 or px_h == 0:
        raise AoiStackError('degenerate geotransform on source raster')

    # Rotated geotransforms would need the full affine inverse; these
    # mosaics are north-up and this keeps the mapping exact.
    if gt[2] != 0 or gt[4] != 0:
        raise AoiStackError(
            'rotated geotransform is not supported for AOI windowing')

    cols = [(xmin - gt[0]) / px_w, (xmax - gt[0]) / px_w]
    rows = [(ymin - gt[3]) / px_h, (ymax - gt[3]) / px_h]

    x0 = int(np.floor(min(cols)))
    x1 = int(np.ceil(max(cols)))
    y0 = int(np.floor(min(rows)))
    y1 = int(np.ceil(max(rows)))

    x0c = max(0, min(raster_w, x0))
    x1c = max(0, min(raster_w, x1))
    y0c = max(0, min(raster_h, y0))
    y1c = max(0, min(raster_h, y1))

    xsize = x1c - x0c
    ysize = y1c - y0c
    if xsize <= 0 or ysize <= 0:
        raise AoiStackError(
            'AOI does not overlap the source raster extent')

    window_gt = (
        gt[0] + x0c * px_w,
        px_w,
        0.0,
        gt[3] + y0c * px_h,
        0.0,
        px_h,
    )
    return x0c, y0c, xsize, ysize, window_gt


# ----------------------------------------------------------------------
# Naming
# ----------------------------------------------------------------------

_SAFE_ID_RE = re.compile(r'[^A-Za-z0-9_-]+')


def sanitize_identifier(identifier: str) -> str:
    """Make a fire name safe for use inside a filename.

    Fire names come from user input and routinely contain spaces (the
    ``new fire`` default) and occasionally slashes, either of which
    would break the path or silently write somewhere unintended.

    NOTE: this is deliberately lossy -- ``fire 1``, ``fire#1`` and
    ``fire_1`` all collapse to ``fire_1``. It is only ever used as a
    *human-readable* portion of the filename; uniqueness comes from the
    hash appended by :func:`aoi_stack_path`.
    """
    cleaned = _SAFE_ID_RE.sub('_', str(identifier or '').strip())
    cleaned = cleaned.strip('_')
    # Keep the readable part bounded so the final path stays well under
    # any filesystem name limit even for very long fire names.
    return (cleaned[:48] or 'aoi')


def aoi_identity_hash(identifier: str, instance_key: str = '') -> str:
    """Short, stable hash uniquely identifying an AOI stack.

    Guards against three distinct collisions that the sanitized name
    alone cannot:

    1. *Lossy sanitization.* ``fire 1`` / ``fire#1`` / ``fire_1`` all
       sanitize to ``fire_1``. Hashing the RAW identifier keeps them
       apart.
    2. *Multiple server instances sharing /ram.* Two servers running
       against different ``out_root``s can each have a fire called
       ``new fire``. Mixing ``instance_key`` (the server's out_root)
       into the hash separates them, so one instance can never read or
       overwrite another's stack.
    3. *Case-insensitive filesystems.* ``K52125`` and ``k52125`` are
       distinct fire names but the same filename on such a mount; the
       hash differs even when the sanitized names do not.
    """
    payload = f'{instance_key}\x00{identifier}'.encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:10]


def aoi_stack_path(identifier: str, post_date: str,
                   ram_dir: str = RAM_DIR,
                   instance_key: str = '',
                   post_source: str = 'mrap',
                   l2_date: str = '') -> str:
    """``/ram/<postdate>_stack_<identifier>_<hash>[_l2].bin``

    The readable identifier is kept so the files are diagnosable by eye;
    the hash is what actually guarantees uniqueness.

    *post_source* selects which post imagery the stack was built from
    ('mrap' or 'l2'). The two are different products over the same AOI,
    so they must not share a path -- otherwise switching sources in the
    UI would read whichever was written last.
    """
    safe = sanitize_identifier(identifier)
    h = aoi_identity_hash(identifier, instance_key)
    suffix = '' if post_source == 'mrap' else f'_{post_source}'
    # An L2 composite built from an earlier START DATE is a different
    # product over the same AOI, so it gets its own file and both
    # persist on the ramdisk. Switching back to a date already built is
    # then a path lookup, not a rebuild.
    #
    # The default (no start date) keeps the plain '_l2' name, so every
    # product built before this feature existed is still found.
    if post_source == 'l2' and l2_date:
        suffix += f'_d{l2_date}'
    return os.path.join(
        ram_dir, f'{post_date}_stack_{safe}_{h}{suffix}.bin')


# ----------------------------------------------------------------------
# Core build
# ----------------------------------------------------------------------

def build_aoi_stack(out_bin: str, xmin: float, ymin: float,
                    xmax: float, ymax: float,
                    pre_bin: str = PRE_BIN,
                    post_bin: str = None,
                    post_date: str = None,
                    divide_mode: bool = False,
                    progress_cb=None,
                    post_override: str = None,
                    post_tag: str = '') -> dict:
    """Generate the 12-band AOI stack at *out_bin*.

    Band order matches the province-wide stack exactly:
        1..N     pre  bands  (median composite)
        N+1..2N  post bands  (latest MRAP mosaic)
        2N+1..3N anomaly bands

    *progress_cb*, if given, is called as ``progress_cb(detail, fraction)``
    so callers can surface progress in the UI.

    Returns a dict describing what was written.
    """
    def _p(detail, frac):
        if progress_cb:
            try:
                progress_cb(detail, frac)
            except Exception:
                pass

    # post_override is an already-windowed 4-band raster on the AOI
    # grid (the L2-recent composite). When present its bands are used
    # verbatim as the post imagery instead of windowing the province
    # mosaic -- the pre bands and the anomaly formula are unchanged, so
    # the two sources produce structurally identical stacks that differ
    # only in where the post bands came from.
    if post_bin is None or post_date is None:
        _d, _p_ = find_latest_mrap()
        post_date = post_date or _d
        post_bin = post_bin or _p_

    if not os.path.isfile(pre_bin):
        raise AoiStackError(f'pre-image not found: {pre_bin}')
    if not os.path.isfile(post_bin):
        raise AoiStackError(f'post-image not found: {post_bin}')

    _p('opening source mosaics', 0.02)
    ds_pre = gdal.Open(pre_bin, gdal.GA_ReadOnly)
    ds_post = gdal.Open(post_bin, gdal.GA_ReadOnly)
    ds_override = (gdal.Open(post_override, gdal.GA_ReadOnly)
                   if post_override else None)
    if ds_pre is None:
        raise AoiStackError(f'cannot open {pre_bin}')
    if ds_post is None:
        raise AoiStackError(f'cannot open {post_bin}')

    try:
        if (ds_pre.RasterXSize != ds_post.RasterXSize
                or ds_pre.RasterYSize != ds_post.RasterYSize):
            raise AoiStackError(
                f'pre/post dimensions differ: '
                f'{ds_pre.RasterXSize}x{ds_pre.RasterYSize} vs '
                f'{ds_post.RasterXSize}x{ds_post.RasterYSize}')
        n_band = ds_pre.RasterCount
        if ds_post.RasterCount != n_band:
            raise AoiStackError(
                f'pre/post band counts differ: {n_band} vs '
                f'{ds_post.RasterCount}')

        gt = ds_pre.GetGeoTransform()
        proj = ds_pre.GetProjection() or ds_post.GetProjection()

        xoff, yoff, xsize, ysize, win_gt = _window_for_bbox(
            gt, ds_pre.RasterXSize, ds_pre.RasterYSize,
            xmin, ymin, xmax, ymax)

        if ds_override is not None:
            # The override was built on this same window, but guard
            # anyway: a mismatch here would misalign pre against post
            # and produce a meaningless anomaly.
            if (ds_override.RasterXSize != xsize
                    or ds_override.RasterYSize != ysize):
                raise AoiStackError(
                    f'post override is {ds_override.RasterXSize}x'
                    f'{ds_override.RasterYSize} but the AOI window is '
                    f'{xsize}x{ysize}')
            n_band = min(n_band, ds_override.RasterCount)

        pre_names = _parse_band_names(_hdr_for(pre_bin))
        post_names = _parse_band_names(_hdr_for(post_bin))
        pre_date = _date_from_band_names(pre_names)
        # Suffixes ("B12 2190nm MRAP") drive every generated band name,
        # exactly as sentinel2_anomaly3 does.
        suffixes = []
        for i in range(n_band):
            if i < len(post_names):
                suffixes.append(_after_last_colon(post_names[i]))
            elif i < len(pre_names):
                suffixes.append(_after_last_colon(pre_names[i]))
            else:
                suffixes.append(f'band {i + 1}')

        # sentinel2_anomaly3 refuses to run when the band-name suffixes
        # disagree, because that means the two mosaics are not the same
        # product and the anomaly would compare unrelated wavelengths.
        # Same check here.
        if pre_names and post_names:
            for i in range(min(n_band, len(pre_names), len(post_names))):
                s_pre = _after_last_colon(pre_names[i])
                s_post = _after_last_colon(post_names[i])
                if s_pre != s_post:
                    raise AoiStackError(
                        f'band {i} name suffix mismatch between pre and '
                        f'post: {s_pre!r} vs {s_post!r}')

        os.makedirs(os.path.dirname(out_bin) or '.', exist_ok=True)

        # Build under a process-private temporary name and rename into
        # place only once complete. Two clients confirming AOIs at the
        # same time (or one reading while another rebuilds after a
        # reboot) must never observe a half-written stack -- os.replace
        # is atomic within a filesystem, so a reader sees either the old
        # complete file or the new complete file, never a partial one.
        tmp_bin = f'{out_bin}.tmp{os.getpid()}'
        tmp_hdr = _hdr_for(tmp_bin)
        for path in (tmp_bin, tmp_hdr, tmp_bin + '.aux.xml'):
            try:
                os.remove(path)
            except OSError:
                pass

        driver = gdal.GetDriverByName('ENVI')
        out_ds = driver.Create(tmp_bin, xsize, ysize, n_band * 3,
                               gdal.GDT_Float32,
                               options=['INTERLEAVE=BSQ'])
        if out_ds is None:
            raise AoiStackError(f'could not create {tmp_bin}')
        out_ds.SetGeoTransform(win_gt)
        if proj:
            out_ds.SetProjection(proj)

        total = n_band * 3
        for i in range(n_band):
            _p(f'reading band {i + 1}/{n_band}',
               0.05 + 0.85 * (i / max(1, n_band)))
            pre_a = ds_pre.GetRasterBand(i + 1).ReadAsArray(
                xoff, yoff, xsize, ysize).astype(np.float32)
            if ds_override is not None:
                # Already cropped to the AOI window and on the same
                # grid, so it is read whole rather than windowed.
                post_a = ds_override.GetRasterBand(
                    i + 1).ReadAsArray().astype(np.float32)
            else:
                post_a = ds_post.GetRasterBand(i + 1).ReadAsArray(
                    xoff, yoff, xsize, ysize).astype(np.float32)

            # Anomaly, matching sentinel2_anomaly3.cpp exactly. That
            # code does the raw float division with no zero guard, so
            # post+pre == 0 yields inf/nan there and the same here --
            # downstream consumers (and every previously built stack)
            # already expect that, so "fixing" it would change results.
            with np.errstate(divide='ignore', invalid='ignore'):
                if divide_mode:
                    anom = post_a / pre_a
                else:
                    anom = (post_a - pre_a) / (post_a + pre_a)

            out_ds.GetRasterBand(i + 1).WriteArray(pre_a)
            out_ds.GetRasterBand(n_band + i + 1).WriteArray(post_a)
            out_ds.GetRasterBand(2 * n_band + i + 1).WriteArray(
                anom.astype(np.float32))
            del pre_a, post_a, anom

        _p('flushing stack to ramdisk', 0.92)
        out_ds.FlushCache()
        out_ds = None
    finally:
        ds_pre = None
        ds_post = None
        ds_override = None

    formula = (_ANOMALY_FORMULA_DIVIDE if divide_mode
               else _ANOMALY_FORMULA)
    band_names = (
        [f'pre {pre_date} 20m: {s}' if pre_date else f'pre 20m: {s}'
         for s in suffixes]
        # The 'pst' prefix is load-bearing: preview.detect_band_groups
        # identifies the post group by it, and the mapping CLI's RGB
        # scan depends on that grouping. Provenance therefore goes in
        # the trailing tag, never the prefix -- labelling these 'l2r'
        # made detect_band_groups return zero post bands, which broke
        # red-wins ("need 3, found 0"), the previews and the CLI.
        + [f'pst {post_date} 20m: {s}{post_tag}' for s in suffixes]
        + [f'anomaly: {s} {formula}' for s in suffixes]
    )

    _p('writing header', 0.96)
    _write_envi_header(tmp_hdr, xsize, ysize, total,
                       band_names, win_gt, proj)

    # Publish atomically: header first, then the data file. A reader
    # checks for BOTH (see stack_is_valid), and only the .bin rename
    # makes the pair visible, so ordering here cannot expose a stack
    # whose header is missing.
    os.replace(tmp_hdr, _hdr_for(out_bin))
    os.replace(tmp_bin, out_bin)
    for junk in (tmp_bin + '.aux.xml',):
        try:
            os.remove(junk)
        except OSError:
            pass
    _p('AOI stack ready', 1.0)

    return {
        'path': out_bin,
        'hdr': _hdr_for(out_bin),
        'width': xsize,
        'height': ysize,
        'bands': total,
        'pre_bin': pre_bin,
        'post_bin': post_bin,
        'post_date': post_date,
        'pre_date': pre_date,
    }


def _write_envi_header(hdr_path, samples, lines, bands, band_names,
                       gt, proj):
    """Rewrite the ENVI header with the full band-name block.

    GDAL's ENVI driver writes a serviceable header but names the bands
    "Band 1..N". The mapping CLI keys off the band names to find its
    RGB groups, so they have to carry the real
    ``pre/pst/anomaly ... B12 2190nm MRAP`` text. The map info /
    projection info records GDAL emitted are preserved as-is.
    """
    existing = ''
    try:
        with open(hdr_path, 'r', errors='replace') as f:
            existing = f.read()
    except OSError:
        pass

    geo_records = []
    for key in ('map info', 'projection info',
                'coordinate system string'):
        m = re.search(
            r'^(' + key.replace(' ', r'\s+') + r')\s*=\s*\{.*?\}',
            existing, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if m:
            geo_records.append(m.group(0).strip())

    lines_out = [
        'ENVI',
        f'samples = {samples}',
        f'lines = {lines}',
        f'bands = {bands}',
        'header offset = 0',
        'file type = ENVI Standard',
        'data type = 4',
        'interleave = bsq',
        'byte order = 0',
        'band names = {' + ',\n'.join(band_names) + '}',
    ]
    lines_out.extend(geo_records)

    tmp = hdr_path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(lines_out))
    os.replace(tmp, hdr_path)


# ----------------------------------------------------------------------
# Public entry point used by the web app
# ----------------------------------------------------------------------

class _BuildLock:
    """Best-effort cross-process lock for one AOI stack path.

    Two clients confirming the same fire at once -- or a rebuild racing
    a serial sweep in another process -- would otherwise both do the
    full read/compute/write. The loser now waits and reuses the
    winner's output instead.

    Uses O_EXCL lock-file creation rather than fcntl so it behaves the
    same across processes and threads without holding an fd open, and
    carries a staleness timeout so a killed builder cannot deadlock the
    next one.
    """

    def __init__(self, target: str, timeout_s: float = 900.0,
                 poll_s: float = 0.5):
        self.path = f'{target}.lock'
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.acquired = False

    def _stale(self) -> bool:
        try:
            age = time.time() - os.path.getmtime(self.path)
        except OSError:
            return False
        return age > self.timeout_s

    def acquire(self, wait_s: float = 900.0) -> bool:
        """True if we hold the lock, False if we timed out waiting."""
        deadline = time.time() + wait_s
        while True:
            try:
                fd = os.open(self.path,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, f'{os.getpid()}\n'.encode())
                os.close(fd)
                self.acquired = True
                return True
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    # Cannot lock (read-only dir, etc.) -- proceed
                    # unlocked rather than failing the build outright.
                    return True
            if self._stale():
                sys.stderr.write(
                    f'[aoi_stack] removing stale lock {self.path}\n')
                try:
                    os.remove(self.path)
                except OSError:
                    pass
                continue
            if time.time() >= deadline:
                return False
            time.sleep(self.poll_s)

    def release(self):
        if not self.acquired:
            return
        try:
            os.remove(self.path)
        except OSError:
            pass
        self.acquired = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def stack_is_valid(path: str, expect_w: int = 0, expect_h: int = 0) -> bool:
    """True if *path* looks like a usable AOI stack.

    Checks the header exists too: a .bin with no .hdr is unreadable as
    ENVI, which is exactly the state a partially-cleared ramdisk can
    leave behind.
    """
    if not path or not os.path.isfile(path):
        return False
    if not os.path.isfile(_hdr_for(path)):
        return False
    try:
        if os.path.getsize(path) == 0:
            return False
    except OSError:
        return False
    if expect_w and expect_h:
        try:
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                return False
            ok = (ds.RasterXSize == expect_w and ds.RasterYSize == expect_h)
            ds = None
            return ok
        except Exception:
            return False
    return True


def ensure_aoi_stack(identifier: str, bbox_native, progress_cb=None,
                     ram_dir: str = RAM_DIR, force: bool = False,
                     instance_key: str = '',
                     post_source: str = 'mrap',
                     ref_raster: str = None,
                     log_cb=None,
                     l2_start_date: str = '') -> dict:
    """Return the AOI stack for *identifier*, building it if needed.

    This is the function that makes the ramdisk safe to lose. ``/ram``
    is tmpfs, so a reboot empties it while the server's fire state (on
    real disk) still references the stack. Every consumer calls through
    here, so a missing stack is rebuilt from the source mosaics on
    first use instead of surfacing as a file-not-found.

    *instance_key* separates servers that share the same ramdisk (pass
    the server's out_root); see :func:`aoi_identity_hash`.

    *progress_cb* is forwarded to :func:`build_aoi_stack`, and is how
    the "regenerating" message reaches the UI.
    """
    xmin, ymin, xmax, ymax = (float(v) for v in bbox_native)
    post_date, post_bin = find_latest_mrap()
    out_bin = aoi_stack_path(identifier, post_date, ram_dir=ram_dir,
                             l2_date=(l2_start_date
                                      if post_source == 'l2' else ''),
                             instance_key=instance_key,
                             post_source=post_source)

    def _describe(rebuilt: bool) -> dict:
        ds = gdal.Open(out_bin, gdal.GA_ReadOnly)
        info = {
            'path': out_bin,
            'hdr': _hdr_for(out_bin),
            'width': ds.RasterXSize if ds else 0,
            'height': ds.RasterYSize if ds else 0,
            'bands': ds.RasterCount if ds else 0,
            'post_bin': post_bin,
            'post_date': post_date,
            'rebuilt': rebuilt,
            'post_source': post_source,
        }
        ds = None
        return info

    if not force and stack_is_valid(out_bin):
        return _describe(False)

    # Serialize builders of this exact stack. Whoever gets the lock
    # builds; anyone waiting re-checks afterwards and normally finds
    # the finished file rather than repeating the work.
    with _BuildLock(out_bin) as lock:
        got = lock.acquire()
        if not got:
            sys.stderr.write(
                f'[aoi_stack] timed out waiting for another builder of '
                f'{out_bin}; building anyway\n')
        elif not force and stack_is_valid(out_bin):
            # Another process finished it while we waited.
            return _describe(False)

        sys.stderr.write(
            f'[aoi_stack] building {out_bin} for bbox '
            f'({xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}) ...\n')
        sys.stderr.flush()

        override = None
        post_tag = ''
        if post_source == 'l2':
            # Build the most-recent-L2 mosaic first; it becomes the
            # post imagery for this stack. Its grid comes from the same
            # reference raster, so it lands on the AOI window exactly.
            from .l2_recent import build_l2_recent_post, L2RecentError
            ref = ref_raster or post_bin
            l2_tmp = f'{out_bin}.post.bin'
            try:
                l2_info = build_l2_recent_post(
                    (xmin, ymin, xmax, ymax), ref, l2_tmp,
                    progress_cb=(
                        (lambda d, f: progress_cb(d, 0.6 * f))
                        if progress_cb else None),
                    log_cb=log_cb,
                    start_date=l2_start_date or '')
            except L2RecentError as exc:
                raise AoiStackError(f'L2-recent composite failed: {exc}')
            override = l2_tmp

            # Every L2 product for this AOI must sit on the SAME grid.
            #
            # The window is derived from the reference raster and the
            # bbox, so it should already match -- but if a dated build
            # ever lands on a different geotransform or size, the
            # imagery shifts under the overlays and the BCWS perimeter
            # appears in the wrong place, which is exactly the symptom
            # that prompted this check. Compare against the default
            # product when one exists and refuse to publish a
            # mismatch rather than silently misregister it.
            try:
                import glob as _g
                _safe = sanitize_identifier(identifier)
                _h = aoi_identity_hash(identifier, instance_key)
                _refs = [f for f in _g.glob(os.path.join(
                    ram_dir, f'*_stack_{_safe}_{_h}_l2.bin'))
                    if not re.search(r'_l2_d\d{8}\.bin$', f)]
                if _refs and l2_start_date:
                    a = gdal.Open(_refs[0], gdal.GA_ReadOnly)
                    b = gdal.Open(l2_tmp, gdal.GA_ReadOnly)
                    if a is not None and b is not None:
                        ga, gb = a.GetGeoTransform(), b.GetGeoTransform()
                        same = (a.RasterXSize == b.RasterXSize
                                and a.RasterYSize == b.RasterYSize
                                and all(abs(x - y) < 1e-6
                                        for x, y in zip(ga, gb)))
                        if not same:
                            msg = (f'[aoi_stack] GRID MISMATCH for '
                                   f'{identifier} date {l2_start_date}: '
                                   f'{b.RasterXSize}x{b.RasterYSize} '
                                   f'{gb} vs default '
                                   f'{a.RasterXSize}x{a.RasterYSize} '
                                   f'{ga}')
                            sys.stderr.write(msg + '\n')
                            if log_cb:
                                log_cb('  ' + msg)
                            raise AoiStackError(
                                'the composite for this date landed on a '
                                'different grid than the existing L2 '
                                'product; refusing to publish it')
                        sys.stderr.write(
                            f'[aoi_stack] grid verified identical to the '
                            f'default L2 product\n')
                    a = None
                    b = None
            except AoiStackError:
                raise
            except Exception as exc:
                sys.stderr.write(f'[aoi_stack] grid check skipped: '
                                 f'{exc}\n')

            post_tag = ' L2'
            post_date = l2_info.get('post_date') or post_date

            # build_l2_recent_post writes its per-acquisition coverage
            # sidecar next to the file it was told to create -- which
            # is the TEMPORARY post buffer, not the stack. Move it
            # beside the stack, where every reader (the date_plot
            # endpoint) expects it. Without this the sidecar was
            # generated correctly on every build and then silently
            # orphaned at <stack>.bin.post_dates.json, so the UI
            # reported "no coverage recorded" even for a fire created
            # seconds earlier.
            try:
                from .l2_recent import date_polygons_path
                src_json = l2_info.get('dates_json')
                dst_json = date_polygons_path(out_bin)
                if src_json and os.path.isfile(src_json):
                    os.replace(src_json, dst_json)
                    l2_info['dates_json'] = dst_json
                    sys.stderr.write(
                        f'[aoi_stack] date coverage -> '
                        f'{os.path.basename(dst_json)}\n')
            except OSError as exc:
                sys.stderr.write(
                    f'[aoi_stack] could not relocate date sidecar: '
                    f'{exc}\n')

        info = build_aoi_stack(out_bin, xmin, ymin, xmax, ymax,
                               post_bin=post_bin, post_date=post_date,
                               progress_cb=progress_cb,
                               post_override=override,
                               post_tag=post_tag)
        if post_source == 'l2':
            # The temporary post buffer has been consumed into the
            # stack; on a tmpfs it is worth reclaiming immediately
            # rather than leaving a second full copy of the AOI in RAM.
            for junk in (override, _hdr_for(override or ''),
                         (override or '') + '.aux.xml'):
                if junk:
                    try:
                        os.remove(junk)
                    except OSError:
                        pass
            info['tiles'] = l2_info.get('tiles', [])
            info['tile_dates'] = l2_info.get('tile_dates', {})
            info['filled_fraction'] = l2_info.get('filled_fraction')
            info['filled_px'] = l2_info.get('filled_px')
            info['total_px'] = l2_info.get('total_px')
    info['rebuilt'] = True
    info['post_source'] = post_source
    return info


def purge_other_aoi_stacks(keep_paths, ram_dir: str = RAM_DIR) -> int:
    """Delete ``*_stack_*.bin`` in *ram_dir* not in *keep_paths*.

    Bounded cleanup for tmpfs. Deliberately only matches the
    ``_stack_<id>`` shape this module produces, so a hand-placed file in
    /ram is never touched.
    """
    keep = {os.path.abspath(p) for p in keep_paths if p}
    removed = 0
    try:
        names = os.listdir(ram_dir)
    except OSError:
        return 0
    for name in names:
        # Only the finished-stack shape. Deliberately excludes
        # .lock / .tmp<pid> files, which belong to a build that may
        # still be in flight in another process.
        if not re.match(r'^\d{8}_stack_.+\.bin$', name):
            continue
        if name.endswith('.lock') or '.tmp' in name:
            continue
        path = os.path.abspath(os.path.join(ram_dir, name))
        if path in keep:
            continue
        for p in (path, _hdr_for(path), path + '.aux.xml'):
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
    return removed
