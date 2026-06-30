"""Per-year raster overview PNG + sidecar JSON.

Generates one downsampled preview image per year-raster, cached on disk.
The bbox-drawing UI uses this PNG as its background; pixel <-> map <->
WGS84 math is done client-side off the sidecar JSON.

Memory-bounded: uses ``GDAL.Band.ReadAsArray(buf_xsize=, buf_ysize=)`` so
a 100 GB raster reads at the same memory cost as a 100 MB one.
"""

import datetime
import json
import os
import re
import sys
import threading

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

from .preview import detect_band_groups, parse_envi_band_names

DEFAULT_MAX_DIM = 2000


def _year_from_filename(path: str) -> int | None:
    """Extract a 4-digit year from a raster filename, or None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    now_year = datetime.datetime.now().year
    lo, hi = 1970, now_year + 1
    found = set()
    for m in re.finditer(r'(?=(\d{4}))', stem):
        try:
            y = int(m.group(1))
        except ValueError:
            continue
        if lo <= y <= hi:
            found.add(y)
    if len(found) == 1:
        return next(iter(found))
    return None


def _date_from_filename(path: str) -> datetime.date | None:
    """Extract a full yyyymmdd date from a raster filename if present
    (e.g. "20260622_stack.bin" -> 2026-06-22), else None. Used to seed
    sensible default dates in the UI from the actual data on disk,
    rather than hardcoding "today" -- the stack file's own date is the
    authoritative answer to "what date is this post-imagery from."""
    stem = os.path.splitext(os.path.basename(path))[0]
    for m in re.finditer(r'(\d{8})', stem):
        token = m.group(1)
        try:
            d = datetime.datetime.strptime(token, '%Y%m%d').date()
        except ValueError:
            continue
        now_year = datetime.datetime.now().year
        if 1970 <= d.year <= now_year + 1:
            return d
    return None


def _bbox_to_wgs84(crs_wkt: str, xmin, ymin, xmax, ymax):
    """Reproject native bbox corners to WGS84. Returns (W, S, E, N)."""
    src = osr.SpatialReference()
    src.ImportFromWkt(crs_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    lons, lats = [], []
    for x, y in corners:
        lon, lat, _ = ct.TransformPoint(x, y)
        lons.append(lon)
        lats.append(lat)
    return min(lons), min(lats), max(lons), max(lats)


def _cache_key_for(raster_path: str) -> dict:
    st = os.stat(raster_path)
    return {'st_mtime_ns': int(st.st_mtime_ns), 'st_size': int(st.st_size)}


def overview_is_fresh(raster_path: str, json_path: str) -> bool:
    """Return True iff json_path exists and its cache_key matches the
    current raster_path stat."""
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    saved = meta.get('cache_key') or {}
    try:
        cur = _cache_key_for(raster_path)
    except OSError:
        return False
    return (int(saved.get('st_mtime_ns', -1)) == cur['st_mtime_ns']
            and int(saved.get('st_size', -1)) == cur['st_size'])


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes atomically: tmp + fsync + rename + parent dir fsync."""
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(os.path.dirname(path) or '.', os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: str, text: str) -> None:
    _atomic_write_bytes(path, text.encode('utf-8'))


def _pick_band_indices(raster_path: str, n_bands: int) -> tuple[list[int], str]:
    """Decide which bands to render. Prefers post group, else pre, else
    first three bands. Returns (band_indices, group_key) where
    group_key is 'post', 'pre', or 'fallback'."""
    band_names = parse_envi_band_names(raster_path)
    if not band_names:
        band_names = [f'band {i + 1}' for i in range(n_bands)]
    groups = detect_band_groups(band_names)
    for key in ('post', 'pre'):
        idxs = groups.get(key) or []
        if idxs:
            return list(idxs[:3]), key
    return [b for b in (1, 2, 3) if b <= n_bands] or [1], 'fallback'


def _stretched_uint8(channels: list[np.ndarray]) -> np.ndarray:
    """Apply per-channel 2-98% stretch + clip + cast to uint8 RGB."""
    while len(channels) < 3:
        channels.append(channels[-1].copy())
    rgb = np.stack(channels[:3], axis=2).astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        valid = ch[np.isfinite(ch)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        rgb[:, :, c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def generate_overview(
    raster_path: str,
    png_path: str,
    json_path: str,
    max_dim: int = DEFAULT_MAX_DIM,
) -> None:
    """Generate overview PNG + sidecar JSON. Raises on failure.

    Reads the raster with GDAL ``ReadAsArray(buf_xsize=, buf_ysize=)`` so
    only ``~max_dim*max_dim*4*3`` bytes are allocated regardless of source
    size. Writes both files atomically (tmp + rename + parent fsync)."""
    if not os.path.isfile(raster_path):
        raise FileNotFoundError(raster_path)

    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Cannot open raster: {raster_path}')
    try:
        W, H = ds.RasterXSize, ds.RasterYSize
        n_bands = ds.RasterCount
        gt = ds.GetGeoTransform()
        crs_wkt = ds.GetProjection() or ''

        # Compute overview dims preserving aspect, longest edge ≤ max_dim
        if max(W, H) > max_dim:
            scale = max_dim / max(W, H)
            ovr_W = max(1, int(round(W * scale)))
            ovr_H = max(1, int(round(H * scale)))
        else:
            ovr_W, ovr_H = W, H

        band_indices, band_group = _pick_band_indices(raster_path, n_bands)
        all_band_names = parse_envi_band_names(raster_path)
        channels: list[np.ndarray] = []
        for b_idx in band_indices[:3]:
            if b_idx < 1 or b_idx > n_bands:
                channels.append(np.zeros((ovr_H, ovr_W), dtype=np.float32))
                continue
            arr = ds.GetRasterBand(b_idx).ReadAsArray(
                buf_xsize=ovr_W, buf_ysize=ovr_H).astype(np.float32)
            channels.append(arr)
    finally:
        ds = None

    # Build PNG into memory, then atomically write
    rgb_uint8 = _stretched_uint8(channels)
    import io
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.image import imsave
    buf = io.BytesIO()
    imsave(buf, rgb_uint8, format='png')
    os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)
    _atomic_write_bytes(png_path, buf.getvalue())

    # Native + WGS84 extents
    xmin = gt[0]
    xmax = gt[0] + W * gt[1]
    ymin = gt[3] + H * gt[5]   # gt[5] is negative
    ymax = gt[3]
    extent_native = [float(xmin), float(ymin), float(xmax), float(ymax)]
    try:
        w_, s_, e_, n_ = _bbox_to_wgs84(crs_wkt, xmin, ymin, xmax, ymax)
        extent_wgs84 = [float(w_), float(s_), float(e_), float(n_)]
    except Exception:
        extent_wgs84 = None

    year = _year_from_filename(raster_path)
    file_date = _date_from_filename(raster_path)

    # Start: January 1 of the raster's year (the post-imagery's year,
    # or today's year if the filename has no parseable year at all).
    start_year = file_date.year if file_date else (
        year if year is not None else datetime.date.today().year)
    default_start = datetime.date(start_year, 1, 1).isoformat()

    # End / fire date: the date actually encoded in the stack
    # filename (e.g. "20260622_stack.bin" -> 2026-06-22) -- this is
    # the post-imagery's real acquisition date, which is the
    # authoritative answer, not "today" (today's run could be using
    # yesterday's, or an older, stack file). Falls back to Dec 31 of
    # the year if the filename has no full date, matching the
    # previous whole-year-window behaviour.
    if file_date:
        default_end = file_date.isoformat()
    elif year is not None:
        default_end = f'{year}-12-31'
    else:
        default_end = ''

    # Native pixel size (e.g. 20m), read from the geotransform, never
    # hardcoded -- and the actual resolution this overview is sampled
    # at, given how much it was downscaled to fit max_dim.
    native_resolution_m = abs(gt[1])
    if ovr_W > 0:
        overview_resolution_m = native_resolution_m * (W / ovr_W)
    else:
        overview_resolution_m = native_resolution_m

    # Human-readable "R: <band name>" / "G: ..." / "B: ..." lines for
    # whichever bands actually went into the overview's RGB channels,
    # so the UI can show exactly what's being displayed rather than
    # leaving the user to guess from the raw header.
    rgb_letters = ('R', 'G', 'B')
    rgb_band_names = []
    raster_basename = os.path.basename(raster_path)
    for letter, b_idx in zip(rgb_letters, band_indices[:3]):
        if 1 <= b_idx <= len(all_band_names):
            name = all_band_names[b_idx - 1]
        else:
            name = f'band {b_idx}'
        rgb_band_names.append(f'{letter}: {raster_basename} {name}')

    meta = {
        'raster_path': raster_path,
        'raster_stem': os.path.splitext(os.path.basename(raster_path))[0],
        'raster_W': int(W),
        'raster_H': int(H),
        'geotransform': [float(g) for g in gt],
        'crs_wkt': crs_wkt,
        'overview_W': int(ovr_W),
        'overview_H': int(ovr_H),
        'native_resolution_m': float(native_resolution_m),
        'overview_resolution_m': float(overview_resolution_m),
        'year': int(year) if year is not None else None,
        'default_start': default_start,
        'default_end': default_end,
        'extent_native': extent_native,
        'extent_wgs84': extent_wgs84,
        'band_group': band_group,
        'band_indices': [int(b) for b in band_indices[:3]],
        'rgb_band_names': rgb_band_names,
        'cache_key': _cache_key_for(raster_path),
    }
    _atomic_write_text(
        json_path,
        json.dumps(meta, indent=2, sort_keys=False) + '\n')


def ensure_overview(
    raster_path: str,
    png_path: str,
    json_path: str,
    max_dim: int = DEFAULT_MAX_DIM,
) -> bool:
    """Generate overview only if cache is stale. Returns True if (re)generated."""
    if overview_is_fresh(raster_path, json_path) and os.path.isfile(png_path):
        return False
    sys.stderr.write(
        f'[overview] Generating {os.path.basename(png_path)} from '
        f'{os.path.basename(raster_path)} ...\n')
    sys.stderr.flush()
    generate_overview(raster_path, png_path, json_path, max_dim=max_dim)
    return True
