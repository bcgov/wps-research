"""Scaling applied to the AOI stack before clustering and export.

KGC and the t-SNE pipeline both measure Euclidean distance, so how the
bands are scaled decides what "similar" means. Plain 0-1 per band is a
reasonable default but it is only one choice, and a poor one when a
single outlier pixel compresses everything else into a narrow range.

Applied AFTER band selection, so a method only ever sees the bands the
user chose, and to a derived copy of the stack -- the displayed
previews are untouched, because a scaled image is not a picture anyone
should be asked to interpret.

Methods, in the notation used in the UI. x is a band, p a pixel:

  none        x' = x                            (values passed through)
  band01      x' = (x - min_b) / (max_b - min_b)        per band
  trim_band   as band01 but min/max are the P-th and (100-P)-th
              percentiles of that band, optionally clipped to [0, 1]
  trim_int    as trim_band but the percentiles come from the per-pixel
              intensity  I(p) = sqrt(sum_b x_b(p)^2), and the same
              two numbers scale every band
  global      x' = (x - min) / (max - min)   over ALL bands together
  pixel_mm    x' = (x - min_p) / (max_p - min_p)   per pixel, across
              bands (removes brightness, keeps spectral shape)
  pixel_l2    x' = x / sqrt(sum_b x_b^2)      per pixel (unit sphere;
              Euclidean distance becomes spectral angle)
  zscore      x' = (x - mean_b) / std_b                 per band
  robust_z    x' = (x - median_b) / (1.4826 * MAD_b)    per band
  logdb       x' = 10 * log10(max(x, eps))    then band01
"""

import os
import sys

METHODS = ('none', 'band01', 'trim_band', 'trim_int', 'global',
           'pixel_mm', 'pixel_l2', 'zscore', 'robust_z', 'logdb')

DEFAULTS = {
    'method': 'band01',
    'trim_percent': 1.5,        # per side, per band
    'trim_no_clip': True,       # scale by the percentiles, do not clamp
    'trim_right_only': False,   # upper limit from percentile, lower = min
    'trim_left_only': False,    # lower limit from percentile, upper = max
    'int_percent': 1.5,         # per side, intensity-based
    'int_no_clip': True,
    'int_right_only': False,
    'int_left_only': False,
}


def _limits(vals, pct, right_only, left_only):
    """Lower and upper limits for a 1-D array.

    *right_only* takes the upper limit from the percentile and the
    lower from the true minimum; *left_only* is the mirror image. They
    are mutually exclusive, and with neither set both ends come from
    the percentiles.
    """
    import numpy as np
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return 0.0, 1.0
    pct = max(0.0, min(49.0, float(pct)))
    lo_p = float(np.percentile(v, pct))
    hi_p = float(np.percentile(v, 100.0 - pct))
    lo_t, hi_t = float(np.min(v)), float(np.max(v))
    if right_only and not left_only:
        lo, hi = lo_t, hi_p
    elif left_only and not right_only:
        lo, hi = lo_p, hi_t
    else:
        lo, hi = lo_p, hi_p
    if not (hi > lo):
        lo, hi = lo_t, hi_t
    if not (hi > lo):
        hi = lo + 1.0
    return lo, hi


def scale_array(cube, params: dict, log=None):
    """Scale *cube*, shaped (bands, rows, cols). Returns a new array.

    Never raises on degenerate input: a constant band, an all-NaN band
    or a zero-magnitude pixel produce zeros rather than infinities,
    because a NaN reaching the clustering fails the whole run for what
    is usually one bad pixel.
    """
    import numpy as np

    p = dict(DEFAULTS)
    p.update(params or {})
    method = str(p.get('method') or 'band01')
    x = np.array(cube, dtype='float64', copy=True)
    nb = x.shape[0]

    def emit(msg):
        sys.stderr.write(f'[scale] {msg}\n')
        if log:
            try:
                log('  ' + msg)
            except Exception:
                pass

    def _safe(num, den):
        den = np.where(np.abs(den) < 1e-12, 1.0, den)
        return num / den

    if method == 'none':
        emit('no scaling applied')
        return x.astype('float32')

    if method == 'band01':
        for b in range(nb):
            v = x[b]
            mn = np.nanmin(v)
            mx = np.nanmax(v)
            x[b] = _safe(v - mn, mx - mn)
        emit(f'per-band min-max over {nb} band(s)')

    elif method == 'global':
        mn = np.nanmin(x)
        mx = np.nanmax(x)
        x = _safe(x - mn, mx - mn)
        emit(f'global min-max: [{mn:.4g}, {mx:.4g}]')

    elif method == 'trim_band':
        pct = float(p.get('trim_percent', 1.5))
        for b in range(nb):
            lo, hi = _limits(x[b].ravel(), pct,
                             bool(p.get('trim_right_only')),
                             bool(p.get('trim_left_only')))
            x[b] = _safe(x[b] - lo, hi - lo)
        if not p.get('trim_no_clip', True):
            x = np.clip(x, 0.0, 1.0)
        emit(f'per-band {pct}% trim'
             + (' (no clip)' if p.get('trim_no_clip', True)
                else ' (clipped to [0,1])')
             + (' right-only' if p.get('trim_right_only') else '')
             + (' left-only' if p.get('trim_left_only') else ''))

    elif method == 'trim_int':
        pct = float(p.get('int_percent', 1.5))
        inten = np.sqrt(np.nansum(x * x, axis=0))
        lo, hi = _limits(inten.ravel(), pct,
                         bool(p.get('int_right_only')),
                         bool(p.get('int_left_only')))
        # One pair of limits for every band, so inter-band ratios --
        # which is where the burn signal lives -- are preserved.
        x = _safe(x - lo, hi - lo)
        if not p.get('int_no_clip', True):
            x = np.clip(x, 0.0, 1.0)
        emit(f'intensity {pct}% trim: limits [{lo:.4g}, {hi:.4g}]'
             + (' (no clip)' if p.get('int_no_clip', True)
                else ' (clipped)'))

    elif method == 'pixel_mm':
        mn = np.nanmin(x, axis=0)
        mx = np.nanmax(x, axis=0)
        x = _safe(x - mn[None, :, :], (mx - mn)[None, :, :])
        emit('per-pixel min-max across bands (shape only)')

    elif method == 'pixel_l2':
        mag = np.sqrt(np.nansum(x * x, axis=0))
        x = _safe(x, mag[None, :, :])
        emit('per-pixel L2: every spectrum on the unit sphere')

    elif method == 'zscore':
        for b in range(nb):
            v = x[b]
            x[b] = _safe(v - np.nanmean(v), np.nanstd(v))
        emit(f'per-band z-score over {nb} band(s)')

    elif method == 'robust_z':
        for b in range(nb):
            v = x[b]
            med = np.nanmedian(v)
            mad = np.nanmedian(np.abs(v - med))
            x[b] = _safe(v - med, 1.4826 * mad)
        emit('per-band robust z-score (median / MAD)')

    elif method == 'logdb':
        eps = 1e-6
        x = 10.0 * np.log10(np.maximum(x, eps))
        for b in range(nb):
            v = x[b]
            mn = np.nanmin(v)
            mx = np.nanmax(v)
            x[b] = _safe(v - mn, mx - mn)
        emit('10*log10 then per-band min-max')

    else:
        emit(f'unknown method {method!r}; passing values through')
        return np.array(cube, dtype='float32')

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype('float32')


def scaling_tag(params: dict) -> str:
    """Short, stable tag for cache filenames.

    Includes every parameter that changes the OUTPUT, so a stack scaled
    one way is never served from a cache built another way.
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    m = str(p.get('method') or 'band01')
    if m in ('none', 'band01'):
        return m
    if m == 'trim_band':
        bits = [f"{float(p['trim_percent']):g}"]
        if not p.get('trim_no_clip', True):
            bits.append('clip')
        if p.get('trim_right_only'):
            bits.append('R')
        if p.get('trim_left_only'):
            bits.append('L')
        return 'trimb' + '-'.join(bits)
    if m == 'trim_int':
        bits = [f"{float(p['int_percent']):g}"]
        if not p.get('int_no_clip', True):
            bits.append('clip')
        if p.get('int_right_only'):
            bits.append('R')
        if p.get('int_left_only'):
            bits.append('L')
        return 'trimi' + '-'.join(bits)
    return m


def scale_raster(src_path: str, out_path: str, params: dict,
                 log=None) -> str:
    """Write a scaled copy of an ENVI stack, preserving georeferencing.

    Returns *out_path*, or *src_path* unchanged when the method is a
    no-op or anything fails -- an unscaled run still produces a result.
    """
    try:
        from osgeo import gdal

        method = str((params or {}).get('method') or 'band01')
        if method in ('none',):
            return src_path

        ds = gdal.Open(src_path, gdal.GA_ReadOnly)
        if ds is None:
            return src_path
        nb = ds.RasterCount
        cube = [ds.GetRasterBand(i + 1).ReadAsArray()
                for i in range(nb)]
        names = [ds.GetRasterBand(i + 1).GetDescription() or ''
                 for i in range(nb)]
        gt, proj = ds.GetGeoTransform(), ds.GetProjection()
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None

        import numpy as np
        out = scale_array(np.stack(cube, axis=0), params, log=log)

        drv = gdal.GetDriverByName('ENVI')
        o = drv.Create(out_path, w, h, nb, gdal.GDT_Float32,
                       options=['INTERLEAVE=BSQ'])
        o.SetGeoTransform(gt)
        if proj:
            o.SetProjection(proj)
        for i in range(nb):
            b = o.GetRasterBand(i + 1)
            b.WriteArray(out[i])
            if names[i]:
                b.SetDescription(names[i])
            b = None
        o = None
        hdr = os.path.splitext(out_path)[0] + '.hdr'
        if not os.path.isfile(hdr) and os.path.isfile(out_path + '.hdr'):
            os.replace(out_path + '.hdr', hdr)
        return out_path
    except Exception as exc:
        sys.stderr.write(
            f'[scale] failed ({type(exc).__name__}: {exc}); using the '
            f'unscaled stack\n')
        return src_path
