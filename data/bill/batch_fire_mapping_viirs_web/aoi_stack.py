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
import json
import math
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

def list_mrap_dates(mrap_dir: str = MRAP_DIR) -> list:
    """Every province-wide MRAP mosaic on disk, newest first.

    The counterpart of the L2 date list: it is what lets an operator
    reach back to an earlier day's composite instead of only the newest
    one the builder would pick on its own.
    """
    out = []
    try:
        for name in os.listdir(mrap_dir):
            m = _MRAP_NAME_RE.match(name)
            if not m:
                continue
            path = os.path.join(mrap_dir, name)
            # A mosaic that cannot be opened -- no header, or still
            # being written -- would only produce a failed build later.
            # Either header convention counts: see existing_hdr().
            if not existing_hdr(path):
                continue
            if not mosaic_is_readable(path):
                sys.stderr.write(
                    f'[mrap] {name} is not readable yet; leaving it '
                    f'out of the date list\n')
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            out.append({'date': m.group(1), 'path': path,
                        'bytes': size})
    except OSError as exc:
        sys.stderr.write(f'[mrap] cannot list {mrap_dir}: {exc}\n')
    out.sort(key=lambda d: d['date'], reverse=True)
    return out


def find_mrap_for_date(date: str, mrap_dir: str = MRAP_DIR):
    """(date, path) for one specific mosaic, or (None, None)."""
    if not re.fullmatch(r'\d{8}', date or ''):
        return None, None
    cand = os.path.join(mrap_dir, f'{date}_mrap.bin')
    # Either header convention counts: see existing_hdr().
    if os.path.isfile(cand) and existing_hdr(cand):
        return date, cand
    if os.path.isfile(cand):
        sys.stderr.write(
            '[mrap] %s exists but has no .hdr in either form; it '
            'cannot be opened\n' % os.path.basename(cand))
    return None, None


def mosaic_is_readable(path: str) -> bool:
    """Can GDAL actually open this mosaic?

    Existence and a header are not enough. A mosaic still being written
    by the nightly pipeline, or truncated by a failed copy, satisfies
    both and then fails at the first read -- and because EVERY product
    is built against it, one bad file turns every fire on the list into
    "not recognized as being in a supported file format". Checking here
    is the difference between one clear message and a server that looks
    entirely broken.
    """
    if not path or not os.path.isfile(path):
        return False
    if not existing_hdr(path):
        return False
    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            return False
        try:
            ok = (ds.RasterCount > 0 and ds.RasterXSize > 0
                  and ds.RasterYSize > 0)
            if ok:
                # Touch one pixel: a truncated file opens happily and
                # only fails when something reads it.
                b = ds.GetRasterBand(1)
                ok = b is not None and b.ReadAsArray(0, 0, 1, 1) is not None
            return bool(ok)
        finally:
            ds = None
    except Exception:
        return False


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
    # Newest READABLE, not merely newest.
    #
    # The nightly mosaic appears on disk before it is complete, so the
    # newest name is routinely the one that cannot be opened yet.
    # Falling back one night keeps every fire working instead of
    # failing them all until the copy finishes.
    skipped = []
    for date_str, path in sorted(candidates, key=lambda p: p[0],
                                 reverse=True):
        if mosaic_is_readable(path):
            if skipped:
                sys.stderr.write(
                    f'[mrap] using {os.path.basename(path)}; skipped '
                    f'{len(skipped)} unreadable newer mosaic(s): '
                    f'{", ".join(skipped)}\n')
            return date_str, path
        skipped.append(os.path.basename(path))
    raise AoiStackError(
        f'no readable <yyyymmdd>_mrap.bin in {mrap_dir}; tried '
        f'{len(skipped)}: {", ".join(skipped[:6])}')


def _hdr_for(bin_path: str) -> str:
    """The header path to WRITE for a raster: <stem>.hdr."""
    return os.path.splitext(bin_path)[0] + '.hdr'


def existing_hdr(bin_path: str) -> str:
    """The header that actually exists for a raster, or ''.

    ENVI has two conventions: <stem>.hdr, and <name>.bin.hdr with the
    extension appended rather than replaced. GDAL reads either. The
    mosaics in /data/mrap_bc all use the first form; some older STACKS
    in the durable store carry the second. Accepting both costs
    nothing and removes a whole class of "the file is right there"
    failure -- but it was NOT the cause of the missing dated MRAP
    composites, and this docstring said so before the mosaic listing
    proved otherwise.
    """
    if not bin_path:
        return ''
    for cand in (os.path.splitext(bin_path)[0] + '.hdr',
                 bin_path + '.hdr'):
        if os.path.isfile(cand):
            return cand
    return ''


_ENVI_GEO_KEYS = ('map info', 'projection info',
                  'coordinate system string')


def _envi_records(text: str) -> dict:
    """The ``key = {...}`` records of an ENVI header, by lower-case key."""
    out = {}
    for key in _ENVI_GEO_KEYS + ('band names', 'default bands'):
        m = re.search(r'^(' + key.replace(' ', r'\s+') + r')\s*=\s*\{.*?\}',
                      text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if m:
            out[key] = m.group(0).strip()
    return out


def normalize_envi_header(bin_path: str) -> str:
    """Leave exactly ONE complete header, at <stem>.hdr. Returns it.

    The convention in this system is <stem>.hdr -- X.bin is described
    by X.hdr, never by X.bin.hdr. The C++ tools write the appended
    form, so both can end up on disk, and then:

      * GDAL reads the appended one. Observed, not theorised: a stack
        whose X.hdr carried full map info was reported by gdal.Open at
        origin (0, 0), because a 136-byte X.bin.hdr written by the tool
        sat beside it with no map info at all. The product was then
        rejected from its own AOI as "grid ... at (0.0, 0.0)".
      * The normalisers that were supposed to clean this up read
        "if not os.path.isfile(<stem>.hdr)" first, so they did nothing
        precisely when both files existed -- the only case that matters.

    Merging rather than deleting: the appended header is the tool's
    statement about the file it just wrote (dimensions, band count),
    while the stem header may carry geolocation and band names the tool
    does not know. Taking dimensions from the newer file and the
    geolocation records from whichever header has them keeps both.
    """
    if not bin_path:
        return ''
    stem = os.path.splitext(bin_path)[0] + '.hdr'
    appended = bin_path + '.hdr'
    have_stem = os.path.isfile(stem)
    have_app = os.path.isfile(appended)

    if not have_app:
        return stem if have_stem else ''

    def _read(p):
        try:
            with open(p, 'r', errors='replace') as fh:
                return fh.read()
        except OSError:
            return ''

    app_txt = _read(appended)
    if not have_stem:
        # Nothing to merge: adopt the tool's header under the right name.
        try:
            os.replace(appended, stem)
        except OSError as exc:
            sys.stderr.write(f'[aoi_stack] could not rename {appended}: '
                             f'{exc}\n')
            return appended
        return stem

    stem_txt = _read(stem)
    # Whichever was written last describes the current raster geometry.
    try:
        newer_txt, older_txt = ((app_txt, stem_txt)
                                if os.path.getmtime(appended)
                                >= os.path.getmtime(stem)
                                else (stem_txt, app_txt))
    except OSError:
        newer_txt, older_txt = app_txt, stem_txt

    have = _envi_records(newer_txt)
    add = [rec for key, rec in _envi_records(older_txt).items()
           if key not in have]
    merged = newer_txt.rstrip('\n')
    if add:
        merged += '\n' + '\n'.join(add)
    merged += '\n'

    try:
        tmp = stem + '.tmp%d' % os.getpid()
        with open(tmp, 'w') as fh:
            fh.write(merged)
        os.replace(tmp, stem)
        os.remove(appended)
        if add:
            sys.stderr.write(
                '[aoi_stack] merged %d record(s) into %s and removed the '
                'duplicate %s\n'
                % (len(add), os.path.basename(stem),
                   os.path.basename(appended)))
        else:
            sys.stderr.write(
                '[aoi_stack] removed the duplicate header %s\n'
                % os.path.basename(appended))
    except OSError as exc:
        sys.stderr.write(f'[aoi_stack] could not normalise headers for '
                         f'{bin_path}: {exc}\n')
    return stem


def envi_grid_signature(bin_path: str):
    """(samples, lines, map info, projection info, CRS) from the HEADER.

    Read from the ENVI header rather than through GDAL because the
    header IS the definition at the image level, and because reading it
    directly cannot be fooled by a second header file.
    """
    hdr = existing_hdr(bin_path)
    if not hdr:
        return None
    try:
        with open(hdr, 'r', errors='replace') as fh:
            txt = fh.read()
    except OSError:
        return None

    def _scalar(key):
        m = re.search(r'^\s*' + key + r'\s*=\s*(\S+)', txt,
                      re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ''

    def _record(key):
        m = re.search(r'^(' + key.replace(' ', r'\s+') + r')\s*=\s*\{(.*?)\}',
                      txt, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return ' '.join(m.group(2).split()) if m else ''

    return {
        'samples': _scalar('samples'),
        'lines': _scalar('lines'),
        'map info': _record('map info'),
        'projection info': _record('projection info'),
        'coordinate system string': _record('coordinate system string'),
        'hdr': hdr,
    }


def check_grid_conformance(paths, label: str = '', log=None) -> dict:
    """Do all of *paths* share one grid? Reports; changes nothing.

    The AOI's grid -- rows, columns, map info, projection info and
    coordinate system string -- is the definition of the fire at the
    image level, and ENVI is where that definition lives. Every input
    and every derived output for one fire must carry the same five
    values. This does not force them: forcing would hide the fault. It
    names the file that deviates, which is what lets the cause be found.
    """
    sigs = {}
    for p in paths:
        sig = envi_grid_signature(p)
        if sig:
            sigs[p] = sig
    result = {'checked': len(sigs), 'conforming': True, 'deviations': []}
    if len(sigs) < 2:
        return result

    keys = ('samples', 'lines', 'map info', 'projection info',
            'coordinate system string')
    # The most common signature is taken as the AOI's; anything else
    # is the deviation, whichever way round the counts fall.
    from collections import Counter
    counts = Counter(tuple(s[k] for k in keys) for s in sigs.values())
    ref = counts.most_common(1)[0][0]
    for p, s in sorted(sigs.items()):
        got = tuple(s[k] for k in keys)
        if got == ref:
            continue
        differing = [k for k, a, b in zip(keys, got, ref) if a != b]
        result['conforming'] = False
        result['deviations'].append({'path': p, 'fields': differing,
                                     'got': dict(zip(keys, got))})
        msg = ('[grid] %s: %s DEVIATES from this AOI in %s '
               '(%sx%s vs %sx%s)'
               % (label or 'fire', os.path.basename(p),
                  ', '.join(differing), got[0], got[1], ref[0], ref[1]))
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass
    if result['conforming']:
        msg = ('[grid] %s: %d product(s) all on one grid %sx%s'
               % (label or 'fire', len(sigs), ref[0], ref[1]))
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass
    return result


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

def _read_window_padded(band, xoff, yoff, xsize, ysize,
                        raster_w, raster_h):
    """Read a window that may extend beyond the raster.

    Returns a full xsize-by-ysize float32 array; anything outside the
    raster is NaN. GDAL refuses an out-of-range window, so the caller
    used to avoid that by CLIPPING the window to the source -- which is
    what made a fire's products come out on different grids: a source
    covering less than the AOI produced a smaller, shifted stack, and
    the overlays, the zoom and every grid comparison then disagreed
    between products of the same fire.
    """
    out = np.full((ysize, xsize), np.nan, dtype=np.float32)
    sx0 = max(0, xoff)
    sy0 = max(0, yoff)
    sx1 = min(raster_w, xoff + xsize)
    sy1 = min(raster_h, yoff + ysize)
    if sx1 <= sx0 or sy1 <= sy0:
        return out                       # no overlap at all
    data = band.ReadAsArray(sx0, sy0, sx1 - sx0, sy1 - sy0)
    if data is None:
        return out
    out[sy0 - yoff:sy1 - yoff, sx0 - xoff:sx1 - xoff] = \
        np.asarray(data, dtype=np.float32)
    return out


def _aoi_grid_sidecar(out_bin: str, where: str = '') -> str:
    """Path of the AOI's pinned-grid record, beside the stacks."""
    m = re.match(r'^\d{8}_stack_(.+?_[0-9a-fA-F]{6,})',
                 os.path.basename(out_bin or ''))
    if not m:
        return ''
    d = where or os.path.dirname(out_bin)
    return os.path.join(d, f'aoi_grid_{m.group(1)}.json')


def load_pinned_grid(out_bin: str):
    """The AOI's authoritative grid, or None.

    A fire's footprint is decided ONCE, when its first stack is built
    from the rectangle drawn on the province-wide mosaic, and every
    product afterwards is cut to that exact grid. Re-deriving the
    window from the bounding box on each build made the footprint a
    computation rather than a fact: floating-point dust, a different
    source raster, a recovered bbox -- any of them could shift it by a
    column, and products of one fire then disagreed with each other.

    Kept as a sidecar rather than a field on the fire so it is shared
    by every process that builds a stack and survives a lost ramdisk.
    """
    for cand in (_aoi_grid_sidecar(out_bin),
                 _aoi_grid_sidecar(out_bin, _durable_dir())):
        if not cand or not os.path.isfile(cand):
            continue
        try:
            with open(cand, 'r', encoding='utf-8') as fh:
                g = json.load(fh)
            w, h = int(g['width']), int(g['height'])
            gt = tuple(float(v) for v in g['gt'])
            if w > 0 and h > 0 and len(gt) == 6 and gt[1] and gt[5]:
                return {'width': w, 'height': h, 'gt': gt,
                        'proj': g.get('proj', ''), 'path': cand}
        except Exception as exc:
            sys.stderr.write(
                f'[aoi_stack] unreadable AOI grid {cand}: {exc}\n')
    return None


def save_pinned_grid(out_bin: str, width: int, height: int, gt,
                     proj: str = '') -> None:
    """Record the AOI's grid, in both the ramdisk and the store."""
    payload = {'width': int(width), 'height': int(height),
               'gt': [float(v) for v in gt], 'proj': proj or '',
               'written_at': time.time()}
    for d in ('', _durable_dir()):
        cand = _aoi_grid_sidecar(out_bin, d)
        if not cand:
            continue
        try:
            os.makedirs(os.path.dirname(cand), exist_ok=True)
            tmp = f'{cand}.tmp{os.getpid()}'
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)
            os.replace(tmp, cand)
        except OSError as exc:
            sys.stderr.write(
                f'[aoi_stack] could not record the AOI grid at '
                f'{cand}: {exc}\n')
    sys.stderr.write(
        '[aoi_stack] AOI GRID PINNED %dx%d at (%.3f, %.3f) px %.9f -- '
        'every product of this fire will be cut to exactly this\n'
        % (width, height, gt[0], gt[3], gt[1]))


def _durable_dir() -> str:
    try:
        from .durable import store_dir
        return store_dir() or ''
    except Exception:
        return ''


def grid_contains_bbox(grid, bbox_native, slack_px: float = 0.5) -> bool:
    """Does a pinned grid still cover the AOI it was pinned for?"""
    try:
        gt = grid['gt']
        w, h = grid['width'], grid['height']
        xmin, ymin, xmax, ymax = (float(v) for v in bbox_native)
        px, py = abs(gt[1]), abs(gt[5])
        return (gt[0] <= xmin + slack_px * px
                and gt[3] >= ymax - slack_px * py
                and gt[0] + w * px >= xmax - slack_px * px
                and gt[3] - h * py <= ymin + slack_px * py)
    except Exception:
        return False


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

    # Snap to whole pixels before rounding outward.
    #
    # A bbox that IS a whole number of pixels comes out of the division
    # as, say, 50609.00000000019 -- nineteen zeros of agreement and
    # then floating-point dust. ceil() sees a fraction and adds a
    # column that contains nothing, so the same AOI yields 57 px on one
    # run and 58 on the next. Every product built at the old width is
    # then judged "on an OLD grid" and rebuilt -- on every restart,
    # forever. Rounding first, and only then flooring and ceiling,
    # makes an exact boundary behave like the exact boundary it is.
    # 1e-6 px is 20 microns of ground: far below anything real, far
    # above the dust.
    _EPS = 1e-6

    def _snap(v):
        r = round(v)
        return float(r) if abs(v - r) < _EPS else v

    cols = [_snap(v) for v in cols]
    rows = [_snap(v) for v in rows]

    x0 = int(np.floor(min(cols)))
    x1 = int(np.ceil(max(cols)))
    y0 = int(np.floor(min(rows)))
    y1 = int(np.ceil(max(rows)))

    # The window is the AOI, NOT the part of it this source happens to
    # cover.
    #
    # Clipping to the raster made the output grid depend on the source:
    # a mosaic or tile set covering less than the AOI produced a
    # smaller stack at a shifted origin, so two products of the same
    # fire ended up on different grids -- 1445x1737 for one date,
    # 1495x1739 for another. Everything downstream then disagreed: the
    # overlays refused to draw, the view jumped when stepping between
    # products, and each new build retired the others as "old grid".
    # The AOI is a property of the FIRE, so it is the same for every
    # product; pixels the source does not reach are nodata.
    x0c, x1c, y0c, y1c = x0, x1, y0, y1

    xsize = x1c - x0c
    ysize = y1c - y0c
    if xsize <= 0 or ysize <= 0:
        raise AoiStackError('degenerate AOI window')
    # Still require SOME overlap: a stack with no data at all is not
    # worth building, and silently producing one would hide a genuine
    # mismatch between the AOI and the source.
    if (min(raster_w, x1) - max(0, x0) <= 0
            or min(raster_h, y1) - max(0, y0) <= 0):
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
                    post_tag: str = '',
                    aoi_grid: dict = None,
                    identifier: str = '') -> dict:
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
        # Dimensions are NOT the thing that has to match.
        #
        # Two rasters can be the same size and start in different
        # places. The pixel window below was computed from the PRE
        # raster and then read verbatim from the POST raster, so
        # whenever their origins differed the post-fire bands came from
        # a different patch of ground than the pre-fire bands -- the
        # product looked fine and showed the wrong area. What must
        # match is the pixel SIZE (otherwise the two grids cannot be
        # aligned without resampling); the offset is handled per raster
        # a few lines down.
        _gt_pre = ds_pre.GetGeoTransform()
        _gt_post = ds_post.GetGeoTransform()
        if (abs(abs(_gt_pre[1]) - abs(_gt_post[1])) > 1e-6
                or abs(abs(_gt_pre[5]) - abs(_gt_post[5])) > 1e-6):
            raise AoiStackError(
                f'pre/post pixel sizes differ: '
                f'{_gt_pre[1]:.9f}/{_gt_pre[5]:.9f} vs '
                f'{_gt_post[1]:.9f}/{_gt_post[5]:.9f}')
        if (abs(_gt_pre[0] - _gt_post[0]) > 1e-6
                or abs(_gt_pre[3] - _gt_post[3]) > 1e-6):
            sys.stderr.write(
                '[aoi_stack] pre and post start in different places '
                '(pre %.3f,%.3f vs post %.3f,%.3f); each is windowed '
                'on its own grid so both cover the SAME ground\n'
                % (_gt_pre[0], _gt_pre[3], _gt_post[0], _gt_post[3]))
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

        # The AOI's footprint is a FACT, not a calculation.
        #
        # Once a fire's first stack has been cut from the rectangle
        # drawn on the province-wide mosaic, that grid is what the fire
        # IS. Every later product is cut to it exactly -- same columns,
        # same rows, same geotransform -- instead of re-deriving a
        # window that can differ by a column for reasons that have
        # nothing to do with the fire. The window computed above is
        # used only to ESTABLISH the grid the first time.
        _pin = load_pinned_grid(out_bin)
        if _pin and not grid_contains_bbox(
                _pin, (xmin, ymin, xmax, ymax)):
            sys.stderr.write(
                '[aoi_stack] the pinned AOI grid %dx%d no longer covers '
                'this bounding box; re-pinning from the current one\n'
                % (_pin['width'], _pin['height']))
            _pin = None
        if _pin:
            win_gt = tuple(_pin['gt'])
            xsize, ysize = _pin['width'], _pin['height']
            # Offsets into the PRE raster for that same ground.
            xoff = int(round((win_gt[0] - gt[0]) / gt[1]))
            yoff = int(round((win_gt[3] - gt[3]) / gt[5]))
            sys.stderr.write(
                '[aoi_stack] using the pinned AOI grid %dx%d at '
                '(%.3f, %.3f) for %s\n'
                % (xsize, ysize, win_gt[0], win_gt[3],
                   os.path.basename(out_bin)))
        else:
            save_pinned_grid(out_bin, xsize, ysize, win_gt,
                             ds_pre.GetProjection())

        # The grid this product will have. Identical for every product
        # of a fire now, so any difference in this line between two
        # builds of the same fire is a bug worth reporting.
        # ENFORCE the fire's authoritative footprint.
        #
        # aoi_grid is written once, from the first build, and every
        # later product must match it exactly. Without this the window
        # is only as stable as the arithmetic that produced it, and
        # that has already yielded 1440, 1442, 1445, 57 and 58 columns
        # for AOIs nobody edited. Here the answer is compared with the
        # recorded one and the recorded one wins.
        _grid = None
        if aoi_grid:
            try:
                _gw = int(aoi_grid.get('w') or 0)
                _gh = int(aoi_grid.get('h') or 0)
                _ggt = [float(v) for v in (aoi_grid.get('gt') or [])]
            except (TypeError, ValueError):
                _gw = _gh = 0
                _ggt = []
            if _gw > 0 and _gh > 0 and len(_ggt) == 6:
                _same = (xsize == _gw and ysize == _gh
                         and abs(win_gt[0] - _ggt[0]) < 1e-6
                         and abs(win_gt[3] - _ggt[3]) < 1e-6
                         and abs(win_gt[1] - _ggt[1]) < 1e-9)
                if not _same:
                    sys.stderr.write(
                        '[aoi_stack] FOOTPRINT: computed %dx%d at '
                        '(%.3f, %.3f) but this AOI is fixed at %dx%d at '
                        '(%.3f, %.3f); using the fixed footprint so '
                        'every product of this fire matches\n'
                        % (xsize, ysize, win_gt[0], win_gt[3],
                           _gw, _gh, _ggt[0], _ggt[3]))
                    # Re-derive the pixel offsets for the RECORDED
                    # origin, so the data still lands where it belongs.
                    xoff += int(round((_ggt[0] - win_gt[0]) / win_gt[1]))
                    yoff += int(round((_ggt[3] - win_gt[3]) / win_gt[5]))
                    xsize, ysize = _gw, _gh
                    win_gt = (_ggt[0], _ggt[1], _ggt[2],
                              _ggt[3], _ggt[4], _ggt[5])
                else:
                    sys.stderr.write(
                        '[aoi_stack] FOOTPRINT ok: %dx%d at (%.3f, '
                        '%.3f) matches this AOI\n'
                        % (xsize, ysize, win_gt[0], win_gt[3]))
        _grid = {'w': int(xsize), 'h': int(ysize),
                 'gt': [float(v) for v in win_gt]}

        # Where that same ground rectangle sits in the POST raster.
        # Whole pixels, because the pixel sizes are equal and both
        # grids are north-up in the same CRS.
        post_xoff = int(round((win_gt[0] - _gt_post[0]) / _gt_post[1]))
        post_yoff = int(round((win_gt[3] - _gt_post[3]) / _gt_post[5]))

        sys.stderr.write(
            '[aoi_stack] window %dx%d at (%.3f, %.3f) px %.6f  '
            'pre@(%d,%d) post@(%d,%d)  from %s\n'
            % (xsize, ysize, win_gt[0], win_gt[3], win_gt[1],
               xoff, yoff, post_xoff, post_yoff,
               os.path.basename(out_bin)))

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

        pre_names = _parse_band_names(existing_hdr(pre_bin)
                                      or _hdr_for(pre_bin))
        post_names = _parse_band_names(existing_hdr(post_bin)
                                       or _hdr_for(post_bin))
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
            pre_a = _read_window_padded(
                ds_pre.GetRasterBand(i + 1), xoff, yoff, xsize, ysize,
                ds_pre.RasterXSize, ds_pre.RasterYSize)
            if ds_override is not None:
                # Already cropped to the AOI window and on the same
                # grid, so it is read whole rather than windowed.
                post_a = ds_override.GetRasterBand(
                    i + 1).ReadAsArray().astype(np.float32)
            else:
                # The POST raster's own offsets for the SAME ground
                # rectangle. Reusing the pre raster's offsets is what
                # made a product show a different area.
                post_a = _read_window_padded(
                    ds_post.GetRasterBand(i + 1), post_xoff, post_yoff,
                    xsize, ysize,
                    ds_post.RasterXSize, ds_post.RasterYSize)

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
    # One header per raster, at <stem>.hdr, complete.
    normalize_envi_header(out_bin)
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
        # The footprint this product came out on, so the
        # caller can record it as the fire's authoritative grid.
        'grid': _grid,
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

    # Open on the POST bands, not on band 1.
    #
    # The stack is pre | post | anomaly, so a viewer that defaults to
    # bands 1-3 shows the PRE composite -- which is identical across
    # every product for this AOI. Downloading MRAP, L2 and a dated L2
    # then looks like three copies of the same image, when in fact only
    # the half nobody was looking at differs. Naming the post bands as
    # the default makes each file open on what distinguishes it.
    try:
        _post_start = next(
            (i for i, nm in enumerate(band_names)
             if nm.lower().startswith('pst')), None)
        if _post_start is not None and bands >= _post_start + 3:
            lines_out.append(
                'default bands = {'
                + ','.join(str(_post_start + 1 + k) for k in range(3))
                + '}')
    except Exception:
        pass
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
    leave behind. Either header convention counts -- see
    existing_hdr() -- because both are present in this data.
    """
    if not path or not os.path.isfile(path):
        return False
    if not existing_hdr(path):
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


def stack_grid_is_canonical(path: str, bbox_native):
    """Is this stack on the grid the CURRENT bbox would produce?

    True, False, or None when it cannot be told.

    Coverage is not enough. A stack built before the pixel-snapping fix
    is one column too wide, and a wider stack still covers the AOI, so
    the reuse check kept it forever: the fire went on reporting the old
    size and every product built on the correct grid was rejected as
    "a different grid". The AOI is defined by the bounding box, so the
    test is whether the file is on the grid that box implies -- same
    origin pixel, same width, same height -- not merely whether it
    contains it.
    """
    if not path or not os.path.isfile(path) or not bbox_native:
        return None
    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            return None
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None
        if not gt or gt[1] == 0 or gt[5] == 0:
            return None

        # The PINNED grid is the authority when there is one.
        #
        # Deriving the answer from the bounding box here while builds
        # were cut to the pinned grid meant two authorities: a stack
        # from before the pin measured 57 columns and the bbox said 57,
        # so it passed as canonical and was offered -- next to products
        # built to the pinned 58. One layer then sat a column narrower
        # than all the others. The fire's footprint is the pin; every
        # file is judged against it.
        _pin = load_pinned_grid(path)
        if _pin:
            pgt = _pin['gt']
            same_origin = (abs(pgt[0] - gt[0]) < abs(gt[1]) * 1e-6
                           and abs(pgt[3] - gt[3]) < abs(gt[5]) * 1e-6)
            return bool(w == _pin['width'] and h == _pin['height']
                        and same_origin)

        xmin, ymin, xmax, ymax = (float(v) for v in bbox_native)
        px, py = abs(gt[1]), abs(gt[5])

        def _snap(v, eps=1e-6):
            r = round(v)
            return float(r) if abs(v - r) < eps else v

        # The origin must be the pixel that CONTAINS the bbox corner:
        # zero or a fraction of a pixel to its west and north.
        off_x = _snap((xmin - gt[0]) / px)
        off_y = _snap((gt[3] - ymax) / py)
        if not (-1e-6 <= off_x < 1.0) or not (-1e-6 <= off_y < 1.0):
            return False
        want_w = int(math.ceil(_snap((xmax - gt[0]) / px)))
        want_h = int(math.ceil(_snap((gt[3] - ymin) / py)))
        return bool(w == want_w and h == want_h)
    except Exception:
        return None


def stack_covers_bbox(path: str, bbox_native, slack_px: float = 1.5):
    """Does the stack at *path* actually cover this bounding box?

    A file being readable says nothing about WHERE it is. A fire that
    was deleted and recreated, or whose bounding box was repaired,
    leaves a stack from its previous extent at exactly the path a new
    build would write -- same fire name, same identity hash, same
    product key. Reusing it reports "already built" for a product
    covering somebody else's ground, and the selector then withholds
    that product because its grid does not match the AOI. From the
    operator's side the date was requested, the build reported
    success, and nothing appeared.

    True when it covers the box, False when it does not, None when
    that cannot be determined -- the caller treats None as "no
    opinion" and leaves the file alone.
    """
    if not bbox_native or not path or not os.path.isfile(path):
        return None
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in bbox_native)
    except (TypeError, ValueError):
        return None
    ds = None
    try:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            return None
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        if not gt or not w or not h:
            return None
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
    except Exception:
        return None
    finally:
        ds = None


def ensure_aoi_stack(identifier: str, bbox_native, progress_cb=None,
                     ram_dir: str = RAM_DIR, force: bool = False,
                     instance_key: str = '',
                     post_source: str = 'mrap',
                     ref_raster: str = None,
                     log_cb=None,
                     l2_start_date: str = '',
                     mrap_date: str = '',
                     aoi_grid: dict = None) -> dict:
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
    # A specific mosaic, when asked for.
    #
    # Without this the builder always took the newest one, so an
    # earlier day's MRAP composite could not be produced at all -- the
    # imagery was on disk, but nothing could clip it to an AOI.
    post_date, post_bin = (None, None)
    if mrap_date:
        post_date, post_bin = find_mrap_for_date(mrap_date)
        if not post_bin:
            raise AoiStackError(
                f'No province-wide MRAP mosaic for {mrap_date} in '
                f'{MRAP_DIR}.')
        sys.stderr.write(
            '[aoi_stack] %s: MRAP %s -> %s\n'
            % (identifier, mrap_date, os.path.basename(post_bin)))
    if not post_bin:
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

    # A stack from a DIFFERENT extent is not an input, and deleting it
    # is not enough.
    #
    # A fire deleted and recreated leaves a stack at exactly the path a
    # new build writes: same name, same identity hash, same product
    # key. Removing the file and carrying on does not work, because the
    # durable-restore step below then copies the very same stale file
    # back from .stacks and the build is skipped again. Forcing the
    # rebuild skips both the reuse and the restore, which is the only
    # way the requested date actually gets made.
    if not force and stack_is_valid(out_bin):
        _cov = stack_covers_bbox(out_bin, bbox_native)
        _canon = stack_grid_is_canonical(out_bin, bbox_native)
        if _cov is False:
            sys.stderr.write(
                '[aoi_stack] REBUILD %s: covers a different extent than '
                'this AOI (its durable copy is stale too, so a plain '
                'delete would just be restored)\n'
                % os.path.basename(out_bin))
            force = True
        elif _canon is False:
            # Covers the AOI but is not ON the AOI's grid -- one column
            # or row too many from the pre-snapping era. Left alone it
            # would be reused forever, and every product built on the
            # correct grid would be rejected against it.
            sys.stderr.write(
                '[aoi_stack] REBUILD %s: covers the AOI but is not on '
                'the grid the bbox implies; rebuilding so every product '
                'of this fire lands on one grid\n'
                % os.path.basename(out_bin))
            force = True
        else:
            sys.stderr.write(
                '[aoi_stack] reuse %s (covers=%s canonical=%s)\n'
                % (os.path.basename(out_bin), _cov, _canon))
        # Drop the sidecars as well. They describe the OLD grid, and
        # anything that reads them afterwards -- the product
        # enumeration, the overlay builder -- would judge the new stack
        # by the old one's dimensions.
        _stem = os.path.splitext(out_bin)[0]
        for _side in ('_overlays.json', '_dates.json'):
            try:
                os.remove(_stem + _side)
            except OSError:
                pass

    if not force and stack_is_valid(out_bin):
        return _describe(False)

    # A present-but-unreadable stack is rubbish, not a build input.
    #
    # Left in place, everything downstream reports "not recognized as
    # being in a supported file format" and the fire lands in the error
    # state -- when the right answer is simply to make it again. This
    # is how a torn ramdisk file, or a bad restore, heals itself.
    if os.path.isfile(out_bin) and not stack_is_valid(out_bin):
        sys.stderr.write(
            f'[aoi_stack] {os.path.basename(out_bin)} is unreadable; '
            f'discarding it and rebuilding\n')
        for _sfx in ('.bin', '.hdr'):
            try:
                os.remove(os.path.splitext(out_bin)[0] + _sfx)
            except OSError:
                pass

    # Missing from the ramdisk but kept on real disk? Copy it back.
    #
    # /ram is tmpfs: a reboot empties it and every stack has to be
    # rebuilt from the source mosaics -- which costs minutes, and for a
    # dated composite whose day's mosaic has rolled off the source
    # directory is not possible at all. Restoring the durable copy
    # first makes the ramdisk expendable rather than authoritative.
    if not force:
        try:
            from .durable import restore_stack
            if restore_stack(out_bin, log=log_cb) \
                    and stack_is_valid(out_bin):
                _cov = stack_covers_bbox(out_bin, bbox_native)
                if _cov is False:
                    sys.stderr.write(
                        '[aoi_stack] the durable copy of %s covers a '
                        'different extent; discarding it and building '
                        'fresh\n' % os.path.basename(out_bin))
                    for _sfx in ('.bin', '.hdr'):
                        try:
                            os.remove(
                                os.path.splitext(out_bin)[0] + _sfx)
                        except OSError:
                            pass
                else:
                    return _describe(False)
        except Exception as _dexc:
            sys.stderr.write(f'[aoi_stack] durable restore: {_dexc}\n')

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
                # Every product over this AOI must share one grid,
                # whatever the source or date: the MRAP stack, the
                # default L2 stack, and every dated L2 stack. Comparing
                # only against the default L2 product left MRAP vs L2
                # unchecked, and a difference there moves every vector
                # overlay when the source is switched.
                # Real product stacks ONLY.
                #
                # A '*_stack_<safe>_<hash>*.bin' glob also matches the
                # clustering's neighbour tables (…​.bin.kgc_knn_32.bin),
                # whose dimensions have nothing to do with the AOI. The
                # check then compared this composite against a KNN
                # table, found the shapes different, and refused to
                # publish a perfectly good build.
                _prod = re.compile(
                    r'^\d{8}_stack_' + re.escape(_safe) + '_'
                    + re.escape(_h) + r'(_l2(_d\d{8})?)?\.bin$')
                _refs = [
                    f for f in _g.glob(os.path.join(
                        ram_dir, f'*_stack_{_safe}_{_h}*.bin'))
                    if _prod.match(os.path.basename(f))
                    and os.path.abspath(f) != os.path.abspath(l2_tmp)
                    and os.path.isfile(os.path.splitext(f)[0] + '.hdr')]

                b = gdal.Open(l2_tmp, gdal.GA_ReadOnly)
                gb = b.GetGeoTransform() if b is not None else None
                _stale = []
                for _ref in _refs:
                    a = gdal.Open(_ref, gdal.GA_ReadOnly)
                    if a is None or b is None:
                        a = None
                        continue
                    ga = a.GetGeoTransform()
                    same = (a.RasterXSize == b.RasterXSize
                            and a.RasterYSize == b.RasterYSize
                            and all(abs(x - y) < 1e-6
                                    for x, y in zip(ga, gb)))
                    if not same:
                        _stale.append(
                            (_ref, f'{a.RasterXSize}x{a.RasterYSize}',
                             tuple(round(v, 6) for v in ga)))
                    a = None

                if _stale:
                    # Publish anyway, and retire the odd ones out.
                    #
                    # This build came from the CURRENT AOI window --
                    # reference raster plus bounding box -- so when a
                    # sibling disagrees it is the sibling that is out of
                    # date, typically from before the grids were
                    # unified. Refusing the new build left the operator
                    # with a dead end and no way to fix it; deleting the
                    # stale sibling is self-healing, because it is a
                    # ramdisk cache that rebuilds on next use.
                    for _ref, _dims, _gt in _stale:
                        msg = (f'[aoi_stack] {identifier}: '
                               f'{os.path.basename(_ref)} is on an OLD '
                               f'grid ({_dims} {_gt}) -- this build is '
                               f'{b.RasterXSize}x{b.RasterYSize} '
                               f'{tuple(round(v, 6) for v in gb)}. '
                               f'Retiring the old one so it rebuilds.')
                        sys.stderr.write(msg + '\n')
                        if log_cb:
                            log_cb('  ' + msg)
                        for _ext in ('.bin', '.hdr', '_dates.json'):
                            _victim = os.path.splitext(_ref)[0] + _ext
                            try:
                                if os.path.isfile(_victim):
                                    os.remove(_victim)
                            except OSError as _rexc:
                                sys.stderr.write(
                                    f'[aoi_stack] could not remove '
                                    f'{_victim}: {_rexc}\n')
                elif _refs:
                    sys.stderr.write(
                        f'[aoi_stack] grid verified identical across '
                        f'{len(_refs)} existing product(s)\n')
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
