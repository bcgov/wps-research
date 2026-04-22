"""Band detection and preview image generation from ENVI rasters.

No external dependencies beyond numpy, GDAL, scipy, and matplotlib
(all already required by the fire mapping pipeline).
"""

import os
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
        with open(hdr) as f:
            content = f.read()
        m = re.search(
            r'band names\s*=\s*\{(.+?)\}', content,
            re.DOTALL | re.IGNORECASE)
        if m:
            return [n.strip().strip("'\"") for n in m.group(1).split(',')]
    return []


# ---------------------------------------------------------------------------
# Band group detection (mirrors fire_mapping_cli._find_band_groups)
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
                         max_dim: int = MAX_PREVIEW_DIM) -> bool:
    """Generate a web-ready preview PNG from specific bands.

    Applies 2nd-98th percentile stretch per channel.
    Resamples down if the image exceeds *max_dim* on either axis.
    Returns True on success.
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        return False

    w, h = ds.RasterXSize, ds.RasterYSize
    n_bands = ds.RasterCount

    channels = []
    for b_idx in band_indices[:3]:
        if b_idx < 1 or b_idx > n_bands:
            channels.append(np.zeros((h, w), dtype=np.float32))
            continue
        arr = ds.GetRasterBand(b_idx).ReadAsArray().astype(np.float32)
        channels.append(arr)
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
    imsave(output_path, rgb_uint8)
    return True


def generate_all_previews(crop_path: str, cache_dir: str,
                          fire_numbe: str) -> list[str]:
    """Generate all preview PNGs for a cropped raster.

    Returns list of available view keys (e.g. ['post', 'pre', 'diff1']).
    """
    band_names = parse_envi_band_names(crop_path)
    if not band_names:
        ds = gdal.Open(crop_path, gdal.GA_ReadOnly)
        if ds:
            n = ds.RasterCount
            ds = None
            band_names = [f'band {i + 1}' for i in range(n)]

    groups = detect_band_groups(band_names)

    preview_dir = os.path.join(cache_dir, 'previews')
    os.makedirs(preview_dir, exist_ok=True)

    available: list[str] = []
    for key in ('post', 'pre', *DIFF_KEYS):
        indices = groups.get(key, [])
        if not indices:
            continue
        output = os.path.join(preview_dir, f'{key}.png')
        if generate_preview_png(crop_path, indices, output):
            available.append(key)

    return available
