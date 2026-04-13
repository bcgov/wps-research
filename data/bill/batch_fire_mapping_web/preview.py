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
    """Detect pre/post/diff1/diff2 groups from ENVI band names.

    Returns dict mapping group key to list of 1-based band indices (max 3).
    """
    groups: dict[str, list[int]] = {
        'pre': [], 'post': [], 'diff1': [], 'diff2': [],
    }

    for i, name in enumerate(band_names):
        low = name.lower()
        idx = i + 1  # 1-based
        if 'anomaly2' in low or ('post/pre' in low and 'anomaly' not in low):
            groups['diff2'].append(idx)
        elif 'anomaly1' in low or '(post-pre)/(post+pre)' in low:
            groups['diff1'].append(idx)
        elif low.startswith('pst') or low.startswith('post'):
            groups['post'].append(idx)
        elif low.startswith('pre'):
            groups['pre'].append(idx)

    has_keywords = any(len(v) > 0 for v in groups.values())
    if has_keywords:
        for k in groups:
            groups[k] = groups[k][:3]
        return groups

    # Fallback: positional B12/B11/B9 groups
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
    for key in ('post', 'pre', 'diff1', 'diff2'):
        indices = groups.get(key, [])
        if not indices:
            continue
        output = os.path.join(preview_dir, f'{key}.png')
        if generate_preview_png(crop_path, indices, output):
            available.append(key)

    return available
