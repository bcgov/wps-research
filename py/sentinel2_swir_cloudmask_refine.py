#!/usr/bin/env python3
"""
sentinel2_swir_cloudmask.py
===========================
Extract cloud-masked SWIR (B12, B11, B9) from Sentinel-2 zip files.

For each L2A zip file the script:
  1. Extracts CLD (cloud probability) and SCL (scene classification) to RAM.
  2. Refines the cloud probability mask using the ABCD Random-Forest method
     (image bands are features A=C, CLD probability is target B → predicts D).
  3. Combines the refined cloud mask with SCL-based nodata pixels
     (SCL classes 0=NO_DATA, 1=SATURATED_OR_DEFECTIVE, 2=DARK_AREA_PIXELS,
      3=CLOUD_SHADOWS are all treated as invalid / written as NaN).
  4. Extracts B12, B11, B9 either from the same L2A zip (default) or from a
     matching L1C zip when --l1_dir is supplied.  Matching is done by tile-ID
     and acquisition date parsed from the Sentinel-2 filename convention.
  5. Writes an ENVI float32 output (.bin + .hdr) next to the source zip, with
     the same filename stem plus .bin extension, and georeference copied from
     the B12 subdataset.

SCL nodata classes (marked * = treated as invalid):
  0  *NO_DATA
  1  *SATURATED_OR_DEFECTIVE
  2  *DARK_AREA_PIXELS
  3  *CLOUD_SHADOWS
  4   VEGETATION
  5   NOT_VEGETATED
  6   WATER
  7   UNCLASSIFIED
  8   CLOUD_MEDIUM_PROBABILITY
  9   CLOUD_HIGH_PROBABILITY
  10  THIN_CIRRUS
  11  SNOW

Usage examples
--------------
# Process all L2A zips in current directory (B12/B11/B9 from L2A):
    python3 sentinel2_swir_cloudmask.py

# Specify L2A directory explicitly:
    python3 sentinel2_swir_cloudmask.py --l2_dir /data/L2A

# Use L1C bands instead (still needs L2A for cloud masks):
    python3 sentinel2_swir_cloudmask.py --l2_dir /data/L2A --l1_dir /data/L1C

# Tune RF sampling stride and cloud threshold:
    python3 sentinel2_swir_cloudmask.py --skip_f 3000 --cloud_threshold 0.4

# Save trained RF models for reuse:
    python3 sentinel2_swir_cloudmask.py --save_model --model_dir ./models

# Skip RF refinement, use raw CLD probability only:
    python3 sentinel2_swir_cloudmask.py --no_rf

# Save a side-by-side diagnostic PNG next to each output:
    python3 sentinel2_swir_cloudmask.py --png

Flags
-----
--l2_dir          DIR   Directory containing L2A zip/SAFE files  [default: cwd]
--l1_dir          DIR   Directory containing L1C zip/SAFE files  [optional]
--skip_f          INT   RF training pixel stride                  [default: 5000]
--offset          INT   RF training pixel offset                  [default: 0]
--cloud_threshold FLOAT Probability threshold for cloud masking   [default: 0.5]
--save_model            Save trained RF model (.pkl) per scene
--model_dir       DIR   Directory for saved models                [default: ./models]
--no_rf                 Skip RF refinement; use raw CLD directly
--png                   Write diagnostic PNG alongside each output
--overwrite             Re-process scenes whose output .bin already exists
--workers         INT   Number of parallel worker processes       [default: 1]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from osgeo import gdal
except ImportError:
    import gdal

gdal.AllRegister()
gdal.UseExceptions()

# ---------------------------------------------------------------------------
# SCL classes treated as nodata / always-masked
# ---------------------------------------------------------------------------
SCL_NODATA_CLASSES = {0, 1, 2, 3}   # NO_DATA, SATURATED, DARK_AREA, CLOUD_SHADOW


# ===========================================================================
# In-RAM raster extraction helpers
# ===========================================================================

def _suppress_gdal_warnings(x, y, z):
    pass


def _open_zip_or_safe(path: str) -> Optional[gdal.Dataset]:
    """Open a zip or SAFE file with GDAL, suppressing non-fatal warnings."""
    gdal.PushErrorHandler(_suppress_gdal_warnings)
    if path.endswith('.zip'):
        vsi = f'/vsizip/{path}'
        ds = None
        # Try opening vsi path directly — may raise RuntimeError if not a
        # self-describing dataset (most Sentinel-2 zips are not), so catch it.
        try:
            ds = gdal.Open(vsi)
        except RuntimeError:
            ds = None
        if ds is None:
            # Walk the zip contents and open the MTD XML inside the .SAFE folder
            files = gdal.ReadDir(vsi) or []
            for f in files:
                if f.endswith('.SAFE'):
                    for xml_name in ('MTD_MSIL2A.xml', 'MTD_MSIL1C.xml'):
                        xml = f'{vsi}/{f}/{xml_name}'
                        try:
                            ds = gdal.Open(xml)
                        except RuntimeError:
                            ds = None
                        if ds is not None:
                            break
                if ds is not None:
                    break
        gdal.PopErrorHandler()
        return ds
    elif path.endswith('.SAFE'):
        level    = 'L2A' if 'MSIL2A' in path else 'L1C'
        xml_name = 'MTD_MSIL2A.xml' if level == 'L2A' else 'MTD_MSIL1C.xml'
        try:
            ds = gdal.Open(os.path.join(path, xml_name))
        except RuntimeError:
            ds = None
        gdal.PopErrorHandler()
        return ds
    gdal.PopErrorHandler()
    return None


def _find_bands_in_subdatasets(
    ds: gdal.Dataset,
    wanted: list[str]
) -> Dict[str, Tuple[np.ndarray, gdal.Dataset]]:
    """
    Iterate all subdatasets of *ds* and return a dict
    {band_name: (array_float32, subdataset_ds)} for each wanted band name.
    Stops early once all wanted bands are found.
    """
    found: Dict[str, Tuple[np.ndarray, gdal.Dataset]] = {}
    wanted_set = set(wanted)

    for sub_path, _desc in ds.GetSubDatasets():
        if len(found) == len(wanted_set):
            break
        sub_ds = gdal.Open(sub_path)
        if sub_ds is None:
            continue
        for b_idx in range(1, sub_ds.RasterCount + 1):
            band = sub_ds.GetRasterBand(b_idx)
            meta = band.GetMetadata()
            bn = meta.get('BANDNAME', '')
            if bn in wanted_set and bn not in found:
                arr = band.ReadAsArray().astype(np.float32)
                found[bn] = (arr, sub_ds)
        # Don't close sub_ds here — the array is already in memory but we
        # keep sub_ds alive for geotransform queries below; caller is
        # responsible for cleanup (they will fall out of scope naturally).

    return found


def _resample_to_target(
    arr: np.ndarray,
    src_ds: gdal.Dataset,
    target_xsize: int,
    target_ysize: int,
    target_geo: tuple,
    target_proj: str,
    resample_alg: str = 'bilinear',
) -> np.ndarray:
    """Resample *arr* from *src_ds* geometry to the target grid using GDAL MEM."""
    mem = gdal.GetDriverByName('MEM')
    in_ds = mem.Create('', arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    in_ds.SetGeoTransform(src_ds.GetGeoTransform())
    in_ds.SetProjection(src_ds.GetProjection())
    in_ds.GetRasterBand(1).WriteArray(arr)

    out_ds = mem.Create('', target_xsize, target_ysize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(target_geo)
    out_ds.SetProjection(target_proj)

    gdal.Warp(out_ds, in_ds, resampleAlg=resample_alg)
    result = out_ds.GetRasterBand(1).ReadAsArray()
    in_ds = out_ds = None
    return result


# ===========================================================================
# L2A extraction: CLD, SCL, and optionally SWIR bands
# ===========================================================================

def extract_l2a_masks_and_swir(
    l2a_path: str,
    extract_swir: bool = True,
) -> Optional[Dict]:
    """
    Extract from a L2A zip/SAFE file into RAM:
      - CLD array  (cloud probability, 0-100 raw)
      - SCL array  (scene classification)
      - Optionally B12, B11, B9 arrays
      - Geotransform and projection for the 20m grid (masks)
      - Geotransform and projection for the SWIR grid (B12 native res)

    Returns a dict with keys:
        cld, scl, cld_geo, cld_proj, cld_xsize, cld_ysize,
        b12, b11, b9, swir_geo, swir_proj, swir_xsize, swir_ysize  (if extract_swir)
    Returns None on failure.
    """
    ds = _open_zip_or_safe(l2a_path)
    if ds is None:
        print(f'[ERROR] Cannot open {l2a_path}')
        return None

    # --- cloud / SCL ---
    wanted_masks = ['CLD', 'SCL']
    if extract_swir:
        wanted_swir = ['B12', 'B11', 'B9']
    else:
        wanted_swir = []

    mask_found = _find_bands_in_subdatasets(ds, wanted_masks)
    if 'CLD' not in mask_found or 'SCL' not in mask_found:
        print(f'[ERROR] CLD or SCL not found in {l2a_path}')
        return None

    cld_arr, cld_sub = mask_found['CLD']
    scl_arr, scl_sub = mask_found['SCL']

    # Reference grid from CLD (20 m)
    cld_geo   = list(cld_sub.GetGeoTransform())
    cld_proj  = cld_sub.GetProjection()
    cld_xsize = cld_sub.RasterXSize
    cld_ysize = cld_sub.RasterYSize
    target_res = 20.0

    # Resample CLD to 20 m if needed
    src_res = abs(cld_geo[1])
    if abs(src_res - target_res) > 0.5:
        print(f'  Resampling CLD from {src_res:.1f}m → {target_res:.1f}m')
        geo20 = list(cld_geo)
        geo20[1] = target_res
        geo20[5] = -target_res
        extent_x = cld_xsize * src_res
        extent_y = cld_ysize * src_res
        nx = int(round(extent_x / target_res))
        ny = int(round(extent_y / target_res))
        cld_arr = _resample_to_target(
            cld_arr, cld_sub, nx, ny, tuple(geo20), cld_proj, 'bilinear')
        cld_geo = geo20
        cld_xsize, cld_ysize = nx, ny

    # Resample SCL to 20 m if needed, and match CLD grid
    scl_geo  = list(scl_sub.GetGeoTransform())
    scl_res  = abs(scl_geo[1])
    if abs(scl_res - target_res) > 0.5 or scl_arr.shape != (cld_ysize, cld_xsize):
        print(f'  Resampling SCL from {scl_res:.1f}m → {target_res:.1f}m')
        scl_arr = _resample_to_target(
            scl_arr, scl_sub, cld_xsize, cld_ysize, tuple(cld_geo), cld_proj, 'near')

    result = dict(
        cld=cld_arr,
        scl=scl_arr,
        cld_geo=tuple(cld_geo),
        cld_proj=cld_proj,
        cld_xsize=cld_xsize,
        cld_ysize=cld_ysize,
    )

    if not extract_swir:
        return result

    # --- SWIR bands ---
    swir_found = _find_bands_in_subdatasets(ds, wanted_swir)
    missing = [b for b in wanted_swir if b not in swir_found]
    if missing:
        print(f'[ERROR] SWIR band(s) {missing} not found in {l2a_path}')
        return None

    b12_arr, b12_sub = swir_found['B12']
    b11_arr, b11_sub = swir_found['B11']
    b9_arr,  b9_sub  = swir_found['B9']

    swir_geo   = list(b12_sub.GetGeoTransform())
    swir_proj  = b12_sub.GetProjection()
    swir_xsize = b12_sub.RasterXSize
    swir_ysize = b12_sub.RasterYSize
    swir_res   = abs(swir_geo[1])

    # Resample B11 and B9 to match B12 grid
    if b11_arr.shape != (swir_ysize, swir_xsize):
        b11_arr = _resample_to_target(
            b11_arr, b11_sub, swir_xsize, swir_ysize,
            tuple(swir_geo), swir_proj, 'bilinear')
    if b9_arr.shape != (swir_ysize, swir_xsize):
        b9_arr = _resample_to_target(
            b9_arr, b9_sub, swir_xsize, swir_ysize,
            tuple(swir_geo), swir_proj, 'bilinear')

    result.update(dict(
        b12=b12_arr,
        b11=b11_arr,
        b9=b9_arr,
        swir_geo=tuple(swir_geo),
        swir_proj=swir_proj,
        swir_xsize=swir_xsize,
        swir_ysize=swir_ysize,
        swir_res=swir_res,
    ))
    return result


# ===========================================================================
# L1C extraction: SWIR bands only (B12, B11, B9)
# ===========================================================================

def extract_l1c_swir(l1c_path: str) -> Optional[Dict]:
    """
    Extract B12, B11, B9 from a L1C zip/SAFE file into RAM.
    B9 is resampled to match B12 native resolution.

    Returns dict with keys: b12, b11, b9, swir_geo, swir_proj,
                             swir_xsize, swir_ysize, swir_res
    Returns None on failure.
    """
    gdal.PushErrorHandler(_suppress_gdal_warnings)
    ds = None
    if l1c_path.endswith('.zip'):
        vsi = f'/vsizip/{l1c_path}'
        files = gdal.ReadDir(vsi) or []
        for f in files:
            if f.endswith('.SAFE'):
                xml = f'{vsi}/{f}/MTD_MSIL1C.xml'
                try:
                    ds = gdal.Open(xml)
                except RuntimeError:
                    ds = None
                if ds is not None:
                    break
    else:
        try:
            ds = gdal.Open(os.path.join(l1c_path, 'MTD_MSIL1C.xml'))
        except RuntimeError:
            ds = None
    gdal.PopErrorHandler()

    if ds is None:
        print(f'[ERROR] Cannot open L1C file: {l1c_path}')
        return None

    wanted = ['B12', 'B11', 'B9']
    found  = _find_bands_in_subdatasets(ds, wanted)
    missing = [b for b in wanted if b not in found]
    if missing:
        print(f'[ERROR] L1C band(s) {missing} not found in {l1c_path}')
        return None

    b12_arr, b12_sub = found['B12']
    b11_arr, b11_sub = found['B11']
    b9_arr,  b9_sub  = found['B9']

    swir_geo   = list(b12_sub.GetGeoTransform())
    swir_proj  = b12_sub.GetProjection()
    swir_xsize = b12_sub.RasterXSize
    swir_ysize = b12_sub.RasterYSize
    swir_res   = abs(swir_geo[1])

    if b11_arr.shape != (swir_ysize, swir_xsize):
        b11_arr = _resample_to_target(
            b11_arr, b11_sub, swir_xsize, swir_ysize,
            tuple(swir_geo), swir_proj, 'bilinear')
    if b9_arr.shape != (swir_ysize, swir_xsize):
        b9_arr = _resample_to_target(
            b9_arr, b9_sub, swir_xsize, swir_ysize,
            tuple(swir_geo), swir_proj, 'bilinear')

    return dict(
        b12=b12_arr,
        b11=b11_arr,
        b9=b9_arr,
        swir_geo=tuple(swir_geo),
        swir_proj=swir_proj,
        swir_xsize=swir_xsize,
        swir_ysize=swir_ysize,
        swir_res=swir_res,
    )


# ===========================================================================
# ABCD Random Forest helpers (self-contained, no external abcd_rf import)
# ===========================================================================

def _bad_pixel_mask(data: np.ndarray) -> np.ndarray:
    """Boolean (n_pixels,) — True where pixel should be excluded."""
    bp = np.any(~np.isfinite(data), axis=0)
    if data.shape[0] > 1:
        bp |= np.all(data == 0, axis=0)
    return bp


def _fit_rf(X_train: np.ndarray, Y_train: np.ndarray):
    """Fit a RandomForestRegressor, preferring cuML GPU if available."""
    print('  Fitting RandomForestRegressor ...')
    try:
        from cuml.ensemble import RandomForestRegressor
        print('  (using cuML GPU)')
        rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        print('  (sklearn CPU)')
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)
    return rf


def abcd_rf_arrays(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    skip_f: int,
    offset: int = 0,
    pkl_path: Optional[str] = None,
) -> Tuple[np.ndarray, object]:
    """
    A:B::C:D RF regression entirely in RAM.

    Parameters
    ----------
    A        : (n_bands, n_pixels) float32 — training features
    B        : (1, n_pixels)       float32 — training targets (cloud prob)
    C        : (n_bands, n_pixels) float32 — inference features (== A here)
    skip_f   : sampling stride
    offset   : sampling offset
    pkl_path : path to cache/load trained model

    Returns (D, rf) where D is (1, n_pixels).
    """
    if pkl_path and os.path.isfile(pkl_path):
        print(f'  Loading cached model: {pkl_path}')
        with open(pkl_path, 'rb') as fh:
            rf = pickle.load(fh)
    else:
        n_px = A.shape[1]
        if A.shape[1] != B.shape[1]:
            raise ValueError('A and B must have the same number of pixels.')
        if skip_f < 1 or skip_f >= n_px:
            raise ValueError(f'Illegal skip_f={skip_f} for n_px={n_px}.')

        bp = _bad_pixel_mask(A)
        print(f'  Bad pixels: {bp.sum():,}  Good: {(~bp).sum():,}')

        idx       = np.arange(offset, n_px, skip_f)
        good_idx  = idx[~bp[idx]]
        n_train   = len(good_idx)
        print(f'  Training samples: {n_train:,} (skip_f={skip_f}, offset={offset})')
        if n_train == 0:
            raise ValueError('No good training pixels after sampling.')

        X_train = A[:, good_idx].T
        Y_train = B[:, good_idx].T
        np.nan_to_num(Y_train, copy=False, nan=0.0)

        rf = _fit_rf(X_train, Y_train.ravel() if Y_train.shape[1] == 1 else Y_train)

        if pkl_path:
            os.makedirs(os.path.dirname(pkl_path) or '.', exist_ok=True)
            with open(pkl_path, 'wb') as fh:
                pickle.dump(rf, fh)
            print(f'  Model saved → {pkl_path}')

    # Inference
    bp_c     = _bad_pixel_mask(C)
    good_c   = np.where(~bp_c)[0]
    D        = np.full((1, C.shape[1]), np.nan, dtype=np.float32)
    if len(good_c):
        preds = rf.predict(C[:, good_c].T)
        if preds.ndim > 1:
            preds = preds[:, 0]
        D[0, good_c] = preds.astype(np.float32)
        print(f'  Predicted {len(good_c):,} pixels')
    return D, rf


# ===========================================================================
# Cloud mask refinement via ABCD RF
# ===========================================================================

def refine_cloud_mask(
    cld_arr: np.ndarray,
    swir_stack: np.ndarray,
    skip_f: int,
    offset: int,
    pkl_path: Optional[str] = None,
) -> np.ndarray:
    """
    Use the ABCD RF method to predict cloud probability from SWIR bands.

    cld_arr    : (H, W) raw CLD probability array (0-100 raw or 0-1 scaled)
    swir_stack : (H, W, 3) float32 B12/B11/B9 stack at 20m (same grid as CLD)
    skip_f     : RF stride
    offset     : RF offset

    Returns refined probability array (H, W) in [0, 1].
    """
    H, W = cld_arr.shape
    n_px = H * W

    # Normalise CLD to [0,1]
    cld_prob = cld_arr.astype(np.float32)
    if cld_prob.max() > 1.0:
        cld_prob = cld_prob / 100.0

    # Build A (features) and B (targets) as (n_bands, n_pixels)
    A = (swir_stack.astype(np.float32) / 10_000.0).reshape(n_px, 3).T   # (3, n_px)
    B = cld_prob.ravel()[np.newaxis, :]                                   # (1, n_px)
    np.nan_to_num(B, copy=False, nan=0.0)

    try:
        D, _ = abcd_rf_arrays(A, B, A, skip_f=skip_f, offset=offset, pkl_path=pkl_path)
    except ValueError as exc:
        print(f'  [WARN] RF failed ({exc}), falling back to raw CLD.')
        return cld_prob

    refined = np.clip(D[0].reshape(H, W), 0.0, 1.0)
    return refined


# ===========================================================================
# ENVI output writer (uses GDAL directly — no file-based reference needed)
# ===========================================================================

def write_envi(
    out_path: str,
    data: np.ndarray,
    geo: tuple,
    proj: str,
    band_names: Optional[list[str]] = None,
) -> None:
    """
    Write a float32 ENVI (.bin + .hdr) file.

    data : (H, W, K) or (H, W) — NaN where invalid
    geo  : GDAL geotransform 6-tuple
    proj : WKT projection string
    """
    if data.ndim == 2:
        data = data[..., np.newaxis]
    H, W, K = data.shape

    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(str(out_path), W, H, K, gdal.GDT_Float32)
    if out_ds is None:
        raise RuntimeError(f'GDAL could not create {out_path}')

    out_ds.SetGeoTransform(geo)
    out_ds.SetProjection(proj)

    for i in range(K):
        band = out_ds.GetRasterBand(i + 1)
        band.WriteArray(data[..., i])
        band.SetNoDataValue(float('nan'))
        if band_names and i < len(band_names):
            band.SetDescription(band_names[i])

    out_ds.FlushCache()
    out_ds = None
    print(f'  Written → {out_path}')


# ===========================================================================
# Optional diagnostic PNG
# ===========================================================================

def _save_png(
    png_path: str,
    raw_cld: np.ndarray,
    refined_cld: np.ndarray,
    final_mask: np.ndarray,
    swir_rgb: np.ndarray,
    raw_cloud_pct: float = 0.0,
    refined_cloud_pct: float = 0.0,
) -> None:
    """Save a 4-panel diagnostic PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not available — skipping PNG')
        return

    H, W = raw_cld.shape
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    lo = np.nanpercentile(swir_rgb, 1, axis=(0, 1))
    hi = np.nanpercentile(swir_rgb, 99, axis=(0, 1))
    rgb_disp = np.clip((swir_rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)

    axes[0].imshow(rgb_disp)
    axes[0].set_title('SWIR (B12/B11/B9)')
    axes[0].axis('off')

    axes[1].imshow(raw_cld / (raw_cld.max() if raw_cld.max() > 0 else 1),
                   cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Sen2Cor CLD (raw) — {raw_cloud_pct:.1f}% cloud')
    axes[1].axis('off')

    axes[2].imshow(refined_cld, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'ABCD-RF (refined) — {refined_cloud_pct:.1f}% cloud')
    axes[2].axis('off')

    axes[3].imshow(final_mask, cmap='Reds', vmin=0, vmax=1)
    axes[3].set_title('Final mask (cloud + SCL)')
    axes[3].axis('off')

    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  PNG → {png_path}')


# ===========================================================================
# Filename matching helpers
# ===========================================================================

def _parse_sentinel2_key(filename: str) -> Optional[Tuple[str, str]]:
    """
    Return (tile_id, date_str) from a Sentinel-2 filename, e.g.
    S2A_MSIL2A_20210715T102021_N0301_R065_T32TPT_20210715T140000.zip
    -> ('T32TPT', '20210715')
    Returns None if parsing fails.
    """
    parts = Path(filename).stem.split('_')
    try:
        date_str = parts[2][:8]          # YYYYMMDD
        tile_id  = next(p for p in parts if p.startswith('T') and len(p) == 6)
        return tile_id, date_str
    except (IndexError, StopIteration):
        return None


def _build_l1c_lookup(l1c_dir: str) -> Dict[Tuple[str, str], str]:
    """
    Scan l1c_dir for L1C zip/SAFE files and build a dict
    {(tile_id, date_str): full_path}.
    """
    lookup: Dict[Tuple[str, str], str] = {}
    for entry in sorted(Path(l1c_dir).iterdir()):
        name = entry.name
        if ('MSIL1C' in name or 'MSI1C' in name) and (
                name.endswith('.zip') or name.endswith('.SAFE')):
            key = _parse_sentinel2_key(name)
            if key:
                lookup[key] = str(entry)
    return lookup


# ===========================================================================
# Per-scene processing
# ===========================================================================

def process_scene(
    l2a_path: str,
    l1c_path: Optional[str],
    skip_f: int,
    offset: int,
    cloud_threshold: float,
    use_rf: bool,
    save_model: bool,
    model_dir: str,
    write_png: bool,
    overwrite: bool,
) -> bool:
    """
    Process one L2A scene (with optional L1C SWIR source).
    Returns True on success, False on skip/failure.
    """
    stem     = Path(l2a_path).stem
    out_bin  = Path(l2a_path).parent / (stem + '.bin')
    out_png  = Path(l2a_path).parent / (stem + '.png')

    if out_bin.exists() and not overwrite:
        print(f'  [SKIP] Output exists: {out_bin}')
        return False

    source_label = 'L1C' if l1c_path else 'L2A'
    print(f'\n{"="*60}')
    print(f'  Scene : {stem}')
    print(f'  SWIR  : {source_label}')
    print(f'  Cloud : L2A (ABCD-RF{"" if use_rf else " disabled"})')

    # ------------------------------------------------------------------ #
    # 1. Extract cloud masks from L2A (always)
    # ------------------------------------------------------------------ #
    print('  Extracting L2A cloud masks ...')
    l2a_data = extract_l2a_masks_and_swir(l2a_path, extract_swir=(l1c_path is None))
    if l2a_data is None:
        print(f'  [FAIL] Could not extract from {l2a_path}')
        return False

    cld_raw  = l2a_data['cld']    # (H20, W20), values 0-100
    scl_arr  = l2a_data['scl']    # (H20, W20)
    cld_geo  = l2a_data['cld_geo']
    cld_proj = l2a_data['cld_proj']
    H20, W20 = cld_raw.shape

    # ------------------------------------------------------------------ #
    # 2. Extract SWIR bands (L1C or L2A)
    # ------------------------------------------------------------------ #
    if l1c_path:
        print(f'  Extracting L1C SWIR from {Path(l1c_path).name} ...')
        swir_data = extract_l1c_swir(l1c_path)
    else:
        print('  Using L2A SWIR bands ...')
        swir_data = {
            'b12': l2a_data['b12'],
            'b11': l2a_data['b11'],
            'b9':  l2a_data['b9'],
            'swir_geo':   l2a_data['swir_geo'],
            'swir_proj':  l2a_data['swir_proj'],
            'swir_xsize': l2a_data['swir_xsize'],
            'swir_ysize': l2a_data['swir_ysize'],
            'swir_res':   l2a_data['swir_res'],
        }

    if swir_data is None:
        print('  [FAIL] SWIR extraction failed.')
        return False

    b12       = swir_data['b12']
    b11       = swir_data['b11']
    b9        = swir_data['b9']
    swir_geo  = swir_data['swir_geo']
    swir_proj = swir_data['swir_proj']
    Hs, Ws    = swir_data['swir_ysize'], swir_data['swir_xsize']
    swir_res  = swir_data['swir_res']

    # Stack for RF: (H, W, 3)
    swir_stack_20m = np.dstack([b12, b11, b9])  # native SWIR grid initially
    # We need SWIR at 20m for RF (same grid as CLD).
    # Resample if SWIR resolution != 20m:
    if abs(swir_res - 20.0) > 0.5:
        print(f'  Resampling SWIR stack from {swir_res:.1f}m → 20m for RF ...')
        mem = gdal.GetDriverByName('MEM')
        tmp_b12 = _resample_to_target(b12, _make_mem_ds(b12, swir_geo, swir_proj),
                                      W20, H20, cld_geo, cld_proj, 'bilinear')
        tmp_b11 = _resample_to_target(b11, _make_mem_ds(b11, swir_geo, swir_proj),
                                      W20, H20, cld_geo, cld_proj, 'bilinear')
        tmp_b9  = _resample_to_target(b9,  _make_mem_ds(b9,  swir_geo, swir_proj),
                                      W20, H20, cld_geo, cld_proj, 'bilinear')
        swir_stack_20m = np.dstack([tmp_b12, tmp_b11, tmp_b9])
    elif swir_stack_20m.shape[:2] != (H20, W20):
        # Shape mismatch even if resolution matches — force resample
        tmp_b12 = _resample_to_target(b12, _make_mem_ds(b12, swir_geo, swir_proj),
                                      W20, H20, cld_geo, cld_proj, 'bilinear')
        tmp_b11 = _resample_to_target(b11, _make_mem_ds(b11, swir_geo, swir_proj),
                                      W20, H20, cld_geo, cld_proj, 'bilinear')
        tmp_b9  = _resample_to_target(b9,  _make_mem_ds(b9,  swir_geo, swir_proj),
                                      W20, H20, cld_geo, cld_proj, 'bilinear')
        swir_stack_20m = np.dstack([tmp_b12, tmp_b11, tmp_b9])

    # ------------------------------------------------------------------ #
    # 3. Build / refine cloud probability at 20 m
    # ------------------------------------------------------------------ #
    if use_rf:
        print('  Running ABCD-RF cloud refinement ...')
        pkl_path: Optional[str] = None
        if save_model:
            safe_stem = stem.replace(' ', '_')
            pkl_path  = str(Path(model_dir) / f'{safe_stem}_{skip_f}_{offset}.pkl')
        refined_cld = refine_cloud_mask(
            cld_raw, swir_stack_20m, skip_f, offset, pkl_path)
    else:
        print('  Using raw CLD (RF disabled) ...')
        refined_cld = cld_raw.astype(np.float32)
        if refined_cld.max() > 1.0:
            refined_cld /= 100.0

    # ------------------------------------------------------------------ #
    # 4. Build final invalid-pixel mask at 20m
    #    (cloud by threshold OR SCL nodata classes)
    # ------------------------------------------------------------------ #
    cloud_mask_20m = refined_cld >= cloud_threshold
    scl_nodata_20m = np.zeros((H20, W20), dtype=bool)
    for cls in SCL_NODATA_CLASSES:
        scl_nodata_20m |= (scl_arr == cls)
    invalid_20m = cloud_mask_20m | scl_nodata_20m

    # Compute raw Sen2Cor cloud % for comparison
    cld_raw_norm = cld_raw.astype(np.float32)
    if cld_raw_norm.max() > 1.0:
        cld_raw_norm = cld_raw_norm / 100.0
    raw_cloud_pct     = (cld_raw_norm >= cloud_threshold).mean() * 100
    refined_cloud_pct = cloud_mask_20m.mean() * 100

    print(f'  Cloud coverage (Sen2Cor raw)  : {raw_cloud_pct:.1f}%')
    print(f'  Cloud coverage (ABCD refined) : {refined_cloud_pct:.1f}%')
    print(f'  SCL nodata                    : {scl_nodata_20m.mean()*100:.1f}%')
    print(f'  Total invalid                 : {invalid_20m.mean()*100:.1f}%')

    # ------------------------------------------------------------------ #
    # 5. Upsample the invalid mask to the native SWIR grid
    # ------------------------------------------------------------------ #
    if invalid_20m.shape != (Hs, Ws):
        print(f'  Resampling mask from 20m → {swir_res:.1f}m SWIR grid ...')
        invalid_float = invalid_20m.astype(np.float32)
        mem_inv = _make_mem_ds(invalid_float, cld_geo, cld_proj)
        invalid_swir = _resample_to_target(
            invalid_float, mem_inv, Ws, Hs, swir_geo, swir_proj, 'near')
        invalid_swir_bool = invalid_swir >= 0.5
    else:
        invalid_swir_bool = invalid_20m

    # ------------------------------------------------------------------ #
    # 6. Apply mask to SWIR stack and write output
    # ------------------------------------------------------------------ #
    out_stack = np.dstack([b12, b11, b9]).astype(np.float32)
    out_stack[invalid_swir_bool] = np.nan

    # Also NaN-out pixels that are zero across all bands (nodata sentinel)
    all_zero = np.all(out_stack == 0.0, axis=-1)
    out_stack[all_zero] = np.nan

    date_str = stem.split('_')[2][:8] if len(stem.split('_')) > 2 else stem
    band_names = [
        f'{date_str} {int(swir_res)}m: B12 2190nm',
        f'{date_str} {int(swir_res)}m: B11 1610nm',
        f'{date_str} {int(swir_res)}m: B9  945nm',
    ]

    write_envi(str(out_bin), out_stack, swir_geo, swir_proj, band_names)

    # ------------------------------------------------------------------ #
    # 7. Optional PNG
    # ------------------------------------------------------------------ #
    if write_png:
        _save_png(
            str(out_png),
            cld_raw,
            refined_cld,
            invalid_20m.astype(np.float32),
            swir_stack_20m,
            raw_cloud_pct=raw_cloud_pct,
            refined_cloud_pct=refined_cloud_pct,
        )

    return True


def _make_mem_ds(arr: np.ndarray, geo: tuple, proj: str) -> gdal.Dataset:
    """Create an in-memory GDAL dataset from a 2D array (helper for resampling)."""
    mem = gdal.GetDriverByName('MEM')
    ds  = mem.Create('', arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geo)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(arr)
    return ds


# ===========================================================================
# Scene discovery
# ===========================================================================

def find_l2a_files(l2a_dir: str) -> list[str]:
    """Return sorted list of L2A zip/SAFE files in l2a_dir."""
    files = []
    for entry in sorted(Path(l2a_dir).iterdir()):
        name = entry.name
        if ('MSIL2A' in name or 'MSI2A' in name) and (
                name.endswith('.zip') or name.endswith('.SAFE')):
            files.append(str(entry))
    return files


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract cloud-masked SWIR (B12, B11, B9) from Sentinel-2 zips.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--l2_dir', default=None,
        help='Directory of L2A zip/SAFE files. Default: current working directory.')
    parser.add_argument(
        '--l1_dir', default=None,
        help='Optional directory of L1C zip/SAFE files. '
             'When supplied, SWIR bands are taken from L1C; '
             'L2A files are still used for cloud masks.')
    parser.add_argument(
        '--skip_f', type=int, default=5000,
        help='RF training pixel stride (default: 5000).')
    parser.add_argument(
        '--offset', type=int, default=0,
        help='RF training pixel offset (default: 0).')
    parser.add_argument(
        '--cloud_threshold', type=float, default=0.5,
        help='Probability threshold for cloud masking (default: 0.5).')
    parser.add_argument(
        '--no_rf', action='store_true',
        help='Skip RF refinement and use raw CLD probability directly.')
    parser.add_argument(
        '--save_model', action='store_true',
        help='Save trained RF model (.pkl) for each scene.')
    parser.add_argument(
        '--model_dir', default='./models',
        help='Directory for saved RF models (default: ./models).')
    parser.add_argument(
        '--png', action='store_true',
        help='Write a 4-panel diagnostic PNG alongside each output .bin.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Re-process scenes whose output .bin already exists.')
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of parallel worker processes (default: 1).')

    args = parser.parse_args()

    l2a_dir  = args.l2_dir or os.getcwd()
    l2a_dir  = os.path.abspath(l2a_dir)
    use_rf   = not args.no_rf

    if not os.path.isdir(l2a_dir):
        sys.exit(f'[ERROR] l2_dir not found: {l2a_dir}')

    l2a_files = find_l2a_files(l2a_dir)
    if not l2a_files:
        sys.exit(f'[ERROR] No L2A zip/SAFE files found in {l2a_dir}')
    print(f'Found {len(l2a_files)} L2A scene(s) in {l2a_dir}')

    # Build L1C lookup if requested
    l1c_lookup: Dict[Tuple[str, str], str] = {}
    if args.l1_dir:
        l1c_dir = os.path.abspath(args.l1_dir)
        if not os.path.isdir(l1c_dir):
            sys.exit(f'[ERROR] l1_dir not found: {l1c_dir}')
        l1c_lookup = _build_l1c_lookup(l1c_dir)
        print(f'Found {len(l1c_lookup)} L1C scene(s) in {l1c_dir}')

    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)

    # Build per-scene task list
    tasks = []
    for l2a_path in l2a_files:
        l1c_path: Optional[str] = None
        if args.l1_dir:
            key = _parse_sentinel2_key(Path(l2a_path).name)
            if key and key in l1c_lookup:
                l1c_path = l1c_lookup[key]
            else:
                print(f'[WARN] No matching L1C for {Path(l2a_path).name} — '
                      f'falling back to L2A SWIR.')
        tasks.append((
            l2a_path, l1c_path,
            args.skip_f, args.offset, args.cloud_threshold,
            use_rf, args.save_model, args.model_dir,
            args.png, args.overwrite,
        ))

    # Execute
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            results = pool.starmap(process_scene, tasks)
    else:
        results = [process_scene(*t) for t in tasks]

    n_ok   = sum(bool(r) for r in results)
    n_skip = len(results) - n_ok
    print(f'\n[DONE] {n_ok} scene(s) processed, {n_skip} skipped/failed.')


if __name__ == '__main__':
    main()
