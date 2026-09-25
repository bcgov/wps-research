"""Band detection and preview image generation from ENVI rasters.

No external dependencies beyond numpy, GDAL, scipy, and matplotlib
(all already required by the fire mapping pipeline).
"""

import os
import threading
import sys
import re

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

MAX_PREVIEW_DIM = 2000  # max pixels on longest side for web display

# How many post-hoc diff/anomaly groups the UI knows about. Increase if
# you start shipping stacks with more than 3 derived groups after pre+post.
MAX_DIFF_GROUPS = 3
DIFF_KEYS = tuple(f'diff{k}' for k in range(1, MAX_DIFF_GROUPS + 1))


# ---------------------------------------------------------------------------
# ENVI header parsing
# ---------------------------------------------------------------------------

def parse_envi_band_names(raster_path: str) -> list[str]:
    """Parse band names from the ENVI .hdr companion file."""
    base = os.path.splitext(raster_path)[0]
    for hdr in (base + '.hdr', raster_path + '.hdr'):
        if not os.path.exists(hdr):
            continue
        with open(hdr, encoding='utf-8', errors='replace') as f:
            content = f.read()
        m = re.search(
            r'band names\s*=\s*\{(.+?)\}', content,
            re.DOTALL | re.IGNORECASE)
        if m:
            return [n.strip().strip("'\"") for n in m.group(1).split(',')]
    return []


# ---------------------------------------------------------------------------
# Band group detection — positional, formula-agnostic.
# ---------------------------------------------------------------------------

def detect_band_groups(band_names: list[str]) -> dict[str, list[int]]:
    """Detect pre/post/diffK groups from ENVI band names — positional.

    Strategy, keyword-agnostic beyond the pre/post prefix:
      * ``pre``  = bands whose name starts with ``pre``.
      * ``post`` = bands whose name starts with ``pst`` or ``post``.
      * ``N``    = band-count of ``pre`` (or ``post`` if no pre was
        found). This is the group size.
      * ``diff1``, ``diff2``, … ``diffMAX_DIFF_GROUPS`` = successive
        chunks of ``N`` bands taken from everything *not* claimed by
        pre/post, in band-index order. Anomaly-labelling keywords in
        the header are ignored — position decides the group.
      * If neither pre nor post can be identified by prefix, fall back
        to the legacy B12/B11/B9 positional scan (same behaviour as
        before).

    Returns a dict mapping group key to a list of 1-based band indices.
    Every diffK slot up to ``MAX_DIFF_GROUPS`` is always present; empty
    lists mean that chunk wasn't available.
    """
    groups: dict[str, list[int]] = {'pre': [], 'post': []}
    for k in DIFF_KEYS:
        groups[k] = []

    pre_idxs: list[int] = []
    post_idxs: list[int] = []
    for i, name in enumerate(band_names):
        low = name.lower().lstrip()
        if low.startswith('pre'):
            pre_idxs.append(i + 1)
        elif low.startswith('pst') or low.startswith('post'):
            post_idxs.append(i + 1)

    if pre_idxs or post_idxs:
        n_per_group = len(pre_idxs) or len(post_idxs)
        groups['pre'] = pre_idxs[:n_per_group]
        groups['post'] = post_idxs[:n_per_group]

        claimed = set(groups['pre']) | set(groups['post'])
        remaining = [i + 1 for i in range(len(band_names))
                     if (i + 1) not in claimed]
        for k, key in enumerate(DIFF_KEYS):
            chunk = remaining[k * n_per_group: (k + 1) * n_per_group]
            if len(chunk) == n_per_group:
                groups[key] = chunk
        return groups

    # Fallback: positional B12/B11/B9 groups (legacy behaviour).
    positional: list[list[int]] = []
    i = 0
    while i < len(band_names):
        if 'B12' in band_names[i]:
            for j in range(i + 1, min(i + 3, len(band_names))):
                if 'B11' in band_names[j]:
                    for k in range(j + 1, min(j + 3, len(band_names))):
                        if 'B9' in band_names[k]:
                            positional.append([i + 1, j + 1, k + 1])
                            break
                    break
        i += 1

    if len(positional) >= 2:
        groups['pre'] = positional[0]
        groups['post'] = positional[1]
    elif len(positional) == 1:
        groups['post'] = positional[0]
    else:
        n = len(band_names)
        groups['post'] = list(range(1, min(4, n + 1)))

    return groups


# ---------------------------------------------------------------------------
# Preview PNG generation  (uses scipy + matplotlib — no Pillow)
# ---------------------------------------------------------------------------

def generate_preview_png(raster_path: str, band_indices: list[int],
                         output_path: str,
                         max_dim: int = MAX_PREVIEW_DIM,
                         token=None) -> bool:
    """Generate a web-ready preview PNG from specific bands.

    Applies 2nd-98th percentile stretch per channel.
    Resamples down if the image exceeds *max_dim* on either axis.
    Returns True on success.
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        return False

    try:
        w, h = ds.RasterXSize, ds.RasterYSize
        n_bands = ds.RasterCount

        channels = []
        for b_idx in band_indices[:3]:
            if b_idx < 1 or b_idx > n_bands:
                channels.append(np.zeros((h, w), dtype=np.float32))
                continue
            arr = ds.GetRasterBand(b_idx).ReadAsArray().astype(np.float32)
            channels.append(arr)
    finally:
        ds = None

    while len(channels) < 3:
        channels.append(channels[-1].copy())

    rgb = np.stack(channels, axis=2)

    # Percentile stretch per channel
    for c in range(3):
        ch = rgb[:, :, c]
        valid = ch[np.isfinite(ch)]
        if len(valid) == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        rgb[:, :, c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)

    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # Resample if needed
    if max(h, w) > max_dim:
        from scipy.ndimage import zoom as _zoom
        scale = max_dim / max(h, w)
        rgb_uint8 = _zoom(
            rgb_uint8, (scale, scale, 1), order=1,
        ).clip(0, 255).astype(np.uint8)

    # Save using matplotlib (no Pillow dependency)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.image import imsave
    # Write atomically. A preview served while it is being rewritten
    # is decoded as a partial image -- the picture appears with the
    # correct width but only a fraction of its rows. The background
    # prebuild rewrites these files, so a page open can land exactly
    # in that window. tmp+rename makes the swap indivisible: a reader
    # sees either the whole old file or the whole new one.
    # Unique per process AND thread. A fixed temp name is the
    # SAME path for two concurrent renders of this artifact: the
    # first rename wins and the second fails with FileNotFound,
    # which is why previews for a freshly created fire silently
    # failed and the layers only appeared after re-entering it.
    # Rendered OUTSIDE the target directory, committed under the fire's
    # preview lock: another thread may clear or replace this directory
    # while the render runs (see preview_fs). A commit refused because
    # the directory moved on is not an error -- the render is simply no
    # longer wanted -- so it returns False without the twins.
    from .preview_fs import scratch_path, commit
    _tmp = scratch_path(output_path, '.tmp.png')
    imsave(_tmp, rgb_uint8)
    if not commit(_tmp, output_path, token=token, who='render'):
        return False

    # Continuous-tone imagery also gets a JPEG twin.
    #
    # These previews are the single biggest cost in the UI: a 2000 px
    # PNG of satellite imagery runs ~6.7 MB, which is seconds of
    # transfer on this link. The same picture as JPEG is roughly an
    # order of magnitude smaller with no visible difference on
    # continuous-tone data.
    #
    # Masks are deliberately excluded: hint/result overlays have hard
    # edges and exact overlay colours that JPEG would ring and shift.
    #
    # The PNG is still written, so anything that looks for <view>.png
    # -- available_views, geo.json keys, the aspect checks -- is
    # unaffected, and the JPEG is a pure serving optimisation that can
    # be ignored or deleted at any time.
    base = os.path.splitext(os.path.basename(output_path))[0]
    if base in JPEG_VIEWS:
        try:
            _write_jpeg_twin(rgb_uint8, output_path, token=token)
        except Exception as exc:
            sys.stderr.write(
                f'[preview] JPEG twin for {base} failed ({exc}); '
                f'PNG will be served instead\n')

    # Low-resolution proxy for progressive loading.
    #
    # The full preview is megabytes; on a slow link that is seconds of
    # blank pane. A ~400 px proxy is tens of kilobytes and arrives
    # almost immediately, so the pane shows the right scene straight
    # away and sharpens when the full image lands. Cheap to make (a
    # decimation of an array already in memory) and written for every
    # view, masks included, since the wait applies to all of them.
    try:
        _write_low_proxy(rgb_uint8, output_path, token=token)
    except Exception as exc:
        sys.stderr.write(
            f'[preview] low proxy for {base} failed ({exc}); '
            f'progressive loading disabled for this view\n')
    return True


LOW_PROXY_DIM = 400


def _write_low_proxy(rgb_uint8, png_path: str, token=None) -> str:
    """Write ``<view>.low.jpg`` — a small proxy of the same scene."""
    h, w = rgb_uint8.shape[0], rgb_uint8.shape[1]
    step = max(1, int(round(max(h, w) / float(LOW_PROXY_DIM))))
    small = rgb_uint8[::step, ::step, :]
    from osgeo import gdal
    sh, sw = small.shape[0], small.shape[1]
    mem = gdal.GetDriverByName('MEM').Create('', sw, sh, 3, gdal.GDT_Byte)
    for b in range(3):
        mem.GetRasterBand(b + 1).WriteArray(small[:, :, b])
    out_path = os.path.splitext(png_path)[0] + '.low.jpg'
    from .preview_fs import scratch_path, commit
    tmp = scratch_path(out_path, '.tmp.jpg')
    drv = gdal.GetDriverByName('JPEG')
    if drv is None:
        raise RuntimeError('GDAL has no JPEG driver')
    ds = drv.CreateCopy(tmp, mem, options=['QUALITY=70'])
    ds = None
    mem = None
    commit(tmp, out_path, token=token, who='low proxy')
    for junk in (out_path + '.aux.xml', tmp + '.aux.xml'):
        try:
            os.remove(junk)
        except OSError:
            pass
    return out_path


# Views whose previews are continuous-tone imagery, safe for JPEG.
JPEG_VIEWS = ('pre', 'post', 'diff1', 'diff2', 'diff3')
JPEG_QUALITY = 85


def _write_jpeg_twin(rgb_uint8, png_path: str, token=None) -> str:
    """Write a JPEG alongside *png_path*, atomically.

    Uses GDAL rather than matplotlib/Pillow: GDAL is a hard dependency
    here already, whereas Pillow may not be installed, and matplotlib's
    JPEG support requires it.
    """
    from osgeo import gdal
    h, w = rgb_uint8.shape[0], rgb_uint8.shape[1]
    mem = gdal.GetDriverByName('MEM').Create('', w, h, 3, gdal.GDT_Byte)
    for b in range(3):
        mem.GetRasterBand(b + 1).WriteArray(rgb_uint8[:, :, b])
    jpg = os.path.splitext(png_path)[0] + '.jpg'
    from .preview_fs import scratch_path, commit
    tmp = scratch_path(jpg, '.tmp.jpg')
    drv = gdal.GetDriverByName('JPEG')
    if drv is None:
        raise RuntimeError('GDAL has no JPEG driver')
    out = drv.CreateCopy(tmp, mem,
                         options=[f'QUALITY={JPEG_QUALITY}'])
    out = None
    mem = None
    if not commit(tmp, jpg, token=token, who='jpeg twin'):
        return jpg
    # Remove GDAL's sidecar; it serves no purpose for a web preview.
    for junk in (jpg + '.aux.xml', tmp + '.aux.xml'):
        try:
            os.remove(junk)
        except OSError:
            pass
    try:
        p_sz = os.path.getsize(png_path)
        j_sz = os.path.getsize(jpg)
        sys.stderr.write(
            f'[preview] {os.path.basename(jpg)}: '
            f'{j_sz / 1e6:.2f} MB vs PNG {p_sz / 1e6:.2f} MB '
            f'({p_sz / max(1, j_sz):.1f}x smaller)\n')
    except OSError:
        pass
    return jpg


def generate_all_previews(crop_path: str, cache_dir: str,
                          fire_numbe: str,
                          preview_dir: str = None) -> list[str]:
    """Generate all preview PNGs for a cropped raster.

    Returns list of available view keys (e.g. ['post', 'pre', 'diff1']).
    """
    band_names = parse_envi_band_names(crop_path)
    if not band_names:
        ds = gdal.Open(crop_path, gdal.GA_ReadOnly)
        if ds:
            try:
                n = ds.RasterCount
            finally:
                ds = None
            band_names = [f'band {i + 1}' for i in range(n)]

    groups = detect_band_groups(band_names)

    # An explicit directory lets a caller render a product's previews
    # beside the others without the fire being switched to it. Without
    # this the path was always <cache>/previews, so rendering for a
    # different product either overwrote the live set or, if the
    # caller passed the stash directory as cache_dir, landed in
    # <stash>/previews and was never found.
    preview_dir = preview_dir or os.path.join(cache_dir, 'previews')

    # Which product these pixels are, and where they may go.
    #
    # previews/ shows ONE product at a time. A render for a product that
    # is not the one on screen -- the creation worker finishing after a
    # switch, for instance -- used to overwrite it anyway, putting one
    # product's imagery under another's name. It now claims the live
    # directory only when that directory is unmarked or already this
    # product's, and otherwise renders into this product's own stash.
    from .preview_fs import (begin_live_render, end_live_render,
                             is_live_dir)
    try:
        from .prepare import product_key_for_path as _pkfp
        _key = _pkfp(crop_path) or ''
    except Exception:
        _key = ''
    _token = None
    if is_live_dir(preview_dir):
        _token = begin_live_render(preview_dir, _key)
        if _token is None:
            _stash = os.path.join(os.path.dirname(preview_dir),
                                  f'previews_{_key}')
            sys.stderr.write(
                f'[preview] {fire_numbe}: previews/ is showing another '
                f'product; rendering {_key} into '
                f'{os.path.basename(_stash)} instead\n')
            preview_dir = _stash
    os.makedirs(preview_dir, exist_ok=True)
    try:
        return _generate_all_previews_into(
            crop_path, fire_numbe, groups, preview_dir, _token)
    finally:
        if _token is not None:
            end_live_render(preview_dir, _token, _key, ok=True)


def _record_render_geo(preview_dir: str, crop_path: str,
                       views: list) -> None:
    """Record the grid of each rendered view beside the PNGs.

    Every preview directory now describes its own grid, so one rendered
    on a different grid can be recognised and never shown -- and a stash
    is not mistaken for stale merely because nothing recorded its grid,
    which is what had the serve path deleting stashes the warming queue
    was still writing into.
    """
    if not views:
        return
    try:
        ds = gdal.Open(crop_path, gdal.GA_ReadOnly)
        if ds is None:
            return
        gt = [float(v) for v in ds.GetGeoTransform()]
        rw, rh = ds.RasterXSize, ds.RasterYSize
        ds = None
        entries = {}
        for v in views:
            pw = ph = 0
            try:
                from matplotlib.image import imread
                a = imread(os.path.join(preview_dir, f'{v}.png'))
                ph, pw = a.shape[0], a.shape[1]
            except Exception:
                pass
            entries[v] = {'gt': gt, 'rw': rw, 'rh': rh,
                          'w': pw or rw, 'h': ph or rh}
        from .preview_fs import merge_json
        merge_json(os.path.join(preview_dir, 'geo.json'), entries)
    except Exception as exc:
        sys.stderr.write(f'[preview] geo record for '
                         f'{os.path.basename(preview_dir)} failed: {exc}\n')


def _generate_all_previews_into(crop_path, fire_numbe, groups,
                                preview_dir, token) -> list:
    """The render itself, into a directory already chosen and claimed."""

    jobs = []
    for key in ('post', 'pre', *DIFF_KEYS):
        indices = groups.get(key, [])
        if not indices:
            continue
        jobs.append((key, indices,
                     os.path.join(preview_dir, f'{key}.png')))
    if not jobs:
        return []

    # Each view is an independent read-and-render over the same file,
    # so they go together. Serially this was the slowest part of a
    # switch for no reason.
    try:
        from .state import PREVIEW_WORKERS as _pw
    except Exception:
        _pw = 4
    workers = max(1, min(len(jobs), _pw))

    available: list[str] = []
    if workers == 1:
        for vkey, indices, output in jobs:
            if generate_preview_png(crop_path, indices, output,
                                    token=token):
                available.append(vkey)
        _record_render_geo(preview_dir, crop_path, available)
        return available

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(generate_preview_png, crop_path,
                            indices, output, token=token): vkey
                for vkey, indices, output in jobs}
        for fut, vkey in futs.items():
            try:
                if fut.result():
                    available.append(vkey)
            except Exception as exc:
                sys.stderr.write(
                    f'[preview] {fire_numbe}: {vkey} failed: {exc}\n')
    # Stable order regardless of completion order.
    order = ['post', 'pre', *DIFF_KEYS]
    available.sort(key=lambda k: order.index(k) if k in order else 99)
    _record_render_geo(preview_dir, crop_path, available)
    return available
