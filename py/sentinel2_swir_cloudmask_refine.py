#!/usr/bin/env python3
"""
sentinel2_swir_cloudmask_refine.py
===========================
Extract cloud-masked SWIR (B12, B11, B9) from Sentinel-2 zip files.

For each L2A zip file the script:
  1. Extracts CLD (cloud probability) and SCL (scene classification) to RAM.
  2. Refines the cloud probability mask using the ABCD Random-Forest method
     (image bands are features A=C, CLD probability is target B → predicts D).
  3. Combines the refined cloud mask with SCL-based nodata pixels
     (SCL classes 0=NO_DATA, 1=SATURATED_OR_DEFECTIVE, 2=DARK_AREA_PIXELS,
      3=CLOUD_SHADOWS are all treated as invalid / written as NaN).
  4. Extracts B12, B11, B9, B8 (default) or B12, B11, B9 (with --three_band)
     either from the same L2A zip (default) or from a matching L1C zip when
     --l1_dir is supplied.  Matching is done by tile-ID and acquisition date
     parsed from the Sentinel-2 filename convention.
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
# Process all L2A zips in current directory (B12/B11/B9/B8 from L2A, default):
    python3 sentinel2_swir_cloudmask_refine.py

# Three-band output (B12/B11/B9 only, no B8):
    python3 sentinel2_swir_cloudmask_refine.py --three_band

# Specify L2A directory explicitly:
    python3 sentinel2_swir_cloudmask_refine.py --l2_dir /data/L2A

# Use L1C bands instead (still needs L2A for cloud masks):
    python3 sentinel2_swir_cloudmask_refine.py --l2_dir /data/L2A --l1_dir /data/L1C

# Tune RF sampling stride and cloud threshold:
    python3 sentinel2_swir_cloudmask_refine.py --skip_f 3000 --cloud_threshold 0.4

# Save trained RF models for reuse:
    python3 sentinel2_swir_cloudmask_refine.py --save_model --model_dir ./models

# Skip RF refinement, use raw CLD probability only:
    python3 sentinel2_swir_cloudmask_refine.py --no_rf

# Save a side-by-side diagnostic PNG next to each output:
    python3 sentinel2_swir_cloudmask_refine.py --png

Flags
-----
--l2_dir          DIR   Directory containing L2A zip/SAFE files  [default: cwd]
--l1_dir          DIR   Directory containing L1C zip/SAFE files  [optional]
--three_band            Output only B12, B11, B9 (omit B8)       [default: 4 bands]
--skip_f          INT   RF training pixel stride                  [default: 5000]
--offset          INT   RF training pixel offset                  [default: 0]
--cloud_threshold FLOAT Probability threshold for cloud masking   [default: 0.0001]
--save_model            Save trained RF model (.pkl) per scene
--model_dir       DIR   Directory for saved models                [default: ./models]
--no_rf                 Skip RF refinement; use raw CLD directly
--png                   Write diagnostic PNG alongside each output
--overwrite             Re-process scenes whose output .bin already exists
--mrap                  Write MRAP composites instead of per-scene outputs
--N_workers       INT   Number of parallel worker processes       [default: 32]
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

# SCL cloud classes used by the deprecated (SCL-only) masking product.
# These are combined with SCL_NODATA_CLASSES; no RF-refined probability is used.
SCL_CLOUD_CLASSES  = {8, 9, 10}    # CLOUD_MEDIUM_PROB, CLOUD_HIGH_PROB, THIN_CIRRUS


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
        try:
            ds = gdal.Open(vsi)
        except RuntimeError:
            ds = None
        if ds is None:
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


def _make_mem_ds(arr: np.ndarray, geo: tuple, proj: str) -> gdal.Dataset:
    """Create an in-memory GDAL dataset from a 2D array (helper for resampling)."""
    mem = gdal.GetDriverByName('MEM')
    ds  = mem.Create('', arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geo)
    ds.SetProjection(proj)
    ds.GetRasterBand(1).WriteArray(arr)
    return ds


# ===========================================================================
# Band list helpers (centralised three_band logic)
# ===========================================================================

def _output_band_names_list(three_band: bool) -> list[str]:
    """Return the list of output band name keys in order."""
    if three_band:
        return ['B12', 'B11', 'B9']
    return ['B12', 'B11', 'B9', 'B8']


def _wanted_swir_bands(three_band: bool) -> list[str]:
    """Band names to request from subdatasets."""
    return _output_band_names_list(three_band)


def _build_output_band_names(date_str: str, swir_res: float, three_band: bool) -> list[str]:
    """Return band description strings for the output ENVI file."""
    info = {
        'B12': '2190nm',
        'B11': '1610nm',
        'B9':  '945nm',
        'B8':  '842nm',
    }
    return [f'{date_str} {int(swir_res)}m: {b} {info[b]}'
            for b in _output_band_names_list(three_band)]


def _build_output_stack(bands_dict: dict, three_band: bool) -> np.ndarray:
    """Stack output bands into (H, W, K) float32 array.

    bands_dict must contain keys 'b12', 'b11', 'b9' [, 'b8'].
    """
    layers = [bands_dict['b12'], bands_dict['b11'], bands_dict['b9']]
    if not three_band:
        layers.append(bands_dict['b8'])
    return np.dstack(layers).astype(np.float32)


# ===========================================================================
# L2A extraction: CLD, SCL, and optionally SWIR bands
# ===========================================================================

def extract_l2a_masks_and_swir(
    l2a_path: str,
    extract_swir: bool = True,
    three_band: bool = False,
) -> Optional[Dict]:
    """
    Extract from a L2A zip/SAFE file into RAM.

    Returns a dict with keys:
        cld, scl, cld_geo, cld_proj, cld_xsize, cld_ysize,
        b12, b11, b9, [b8,] swir_geo, swir_proj, swir_xsize, swir_ysize  (if extract_swir)
    Returns None on failure.
    """
    ds = _open_zip_or_safe(l2a_path)
    if ds is None:
        print(f'[ERROR] Cannot open {l2a_path}')
        return None

    # --- cloud / SCL ---
    wanted_masks = ['CLD', 'SCL']
    wanted_swir = _wanted_swir_bands(three_band) if extract_swir else []

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
        print(f'[ERROR] Band(s) {missing} not found in {l2a_path}')
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

    # B8 (10m native) — resample to B12 grid
    if not three_band:
        b8_arr, b8_sub = swir_found['B8']
        if b8_arr.shape != (swir_ysize, swir_xsize):
            b8_arr = _resample_to_target(
                b8_arr, b8_sub, swir_xsize, swir_ysize,
                tuple(swir_geo), swir_proj, 'bilinear')
        result['b8'] = b8_arr

    return result


# ===========================================================================
# L1C extraction: SWIR bands only
# ===========================================================================

def extract_l1c_swir(l1c_path: str, three_band: bool = False) -> Optional[Dict]:
    """
    Extract B12, B11, B9 [, B8] from a L1C zip/SAFE file into RAM.
    B9 and B8 are resampled to match B12 native resolution.
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

    wanted = _wanted_swir_bands(three_band)
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

    result = dict(
        b12=b12_arr,
        b11=b11_arr,
        b9=b9_arr,
        swir_geo=tuple(swir_geo),
        swir_proj=swir_proj,
        swir_xsize=swir_xsize,
        swir_ysize=swir_ysize,
        swir_res=swir_res,
    )

    if not three_band:
        b8_arr, b8_sub = found['B8']
        if b8_arr.shape != (swir_ysize, swir_xsize):
            b8_arr = _resample_to_target(
                b8_arr, b8_sub, swir_xsize, swir_ysize,
                tuple(swir_geo), swir_proj, 'bilinear')
        result['b8'] = b8_arr

    return result


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

    cld_prob = cld_arr.astype(np.float32)
    if cld_prob.max() > 1.0:
        cld_prob = cld_prob / 100.0

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
# Cloud shadow mask refinement via ABCD RF
# ===========================================================================

def refine_shadow_mask(
    scl_arr: np.ndarray,
    swir_stack: np.ndarray,
    skip_f: int,
    offset: int,
    pkl_path: Optional[str] = None,
) -> np.ndarray:
    """
    Refine the SCL cloud-shadow mask using the ABCD RF method.

    Returns refined shadow probability array (H, W) in [0, 1].
    """
    H, W  = scl_arr.shape
    n_px  = H * W

    shadow_binary = (scl_arr == 3).astype(np.float32).ravel()
    water_mask = (scl_arr == 6).ravel()

    A = (swir_stack.astype(np.float32) / 10_000.0).reshape(n_px, 3).T
    B = shadow_binary[np.newaxis, :]
    np.nan_to_num(B, copy=False, nan=0.0)

    if pkl_path and os.path.isfile(pkl_path):
        print(f'  [shadow] Loading cached model: {pkl_path}')
        with open(pkl_path, 'rb') as fh:
            rf = pickle.load(fh)
    else:
        bp   = _bad_pixel_mask(A)
        idx  = np.arange(offset, n_px, skip_f)
        good = idx[~bp[idx] & ~water_mask[idx]]
        n_train = len(good)
        print(f'  [shadow] Training samples: {n_train:,} '
              f'(skip_f={skip_f}, water-excluded)')
        if n_train == 0:
            print('  [shadow] No valid training pixels — returning raw SCL shadow.')
            return shadow_binary.reshape(H, W)

        X_train = A[:, good].T
        Y_train = B[0, good]
        np.nan_to_num(Y_train, copy=False, nan=0.0)

        rf = _fit_rf(X_train, Y_train)

        if pkl_path:
            os.makedirs(os.path.dirname(pkl_path) or '.', exist_ok=True)
            with open(pkl_path, 'wb') as fh:
                pickle.dump(rf, fh)
            print(f'  [shadow] Model saved → {pkl_path}')

    bp_c   = _bad_pixel_mask(A)
    good_c = np.where(~bp_c)[0]
    D      = np.full(n_px, np.nan, dtype=np.float32)
    if len(good_c):
        preds = rf.predict(A[:, good_c].T)
        if preds.ndim > 1:
            preds = preds[:, 0]
        D[good_c] = preds.astype(np.float32)
        print(f'  [shadow] Predicted {len(good_c):,} pixels')

    refined = np.clip(np.nan_to_num(D, nan=0.0).reshape(H, W), 0.0, 1.0)
    return refined


def write_envi(
    out_path: str,
    data: np.ndarray,
    geo: tuple,
    proj: str,
    band_names: Optional[list[str]] = None,
) -> None:
    """Write a float32 ENVI (.bin + .hdr) file."""
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
    input_swir: np.ndarray,
    output_product: np.ndarray,
    raw_cloud_pct: float = 0.0,
    refined_cld: Optional[np.ndarray] = None,
    refined_cloud_pct: float = 0.0,
) -> None:
    """Save a diagnostic PNG (3- or 4-panel)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not available — skipping PNG')
        return

    full_mode = refined_cld is not None
    n_panels  = 4 if full_mode else 3

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # Use first 3 bands for display
    disp_out = output_product[..., :3] if output_product.ndim == 3 and output_product.shape[2] > 3 else output_product
    disp_in  = input_swir[..., :3] if input_swir.ndim == 3 and input_swir.shape[2] > 3 else input_swir

    n_bands = disp_out.shape[2] if disp_out.ndim == 3 else 1
    lo = np.zeros(n_bands, dtype=np.float32)
    hi = np.ones(n_bands,  dtype=np.float32)
    for b in range(n_bands):
        band = disp_out[..., b] if disp_out.ndim == 3 else disp_out
        valid = band[np.isfinite(band) & (band != 0)]
        if valid.size > 0:
            lo[b] = np.percentile(valid, 1)
            hi[b] = np.percentile(valid, 99)

    def _apply_stretch(arr):
        arr = arr[..., :3] if arr.ndim == 3 and arr.shape[2] > 3 else arr
        arr = arr.astype(np.float32)
        out = np.zeros_like(arr)
        nb  = arr.shape[2] if arr.ndim == 3 else 1
        for b in range(nb):
            layer = arr[..., b] if arr.ndim == 3 else arr
            scaled = (layer - lo[b]) / np.maximum(hi[b] - lo[b], 1e-6)
            out[..., b] = np.where(np.isfinite(layer), np.clip(scaled, 0, 1), 0.0)
        return out

    cld_norm = raw_cld.astype(np.float32)
    if cld_norm.max() > 1.0:
        cld_norm = cld_norm / 100.0

    if full_mode:
        axes[0].imshow(_apply_stretch(disp_in));  axes[0].set_title('Input SWIR (B12/B11/B9)'); axes[0].axis('off')
        axes[1].imshow(cld_norm, cmap='gray', vmin=0, vmax=1); axes[1].set_title(f'Sen2Cor CLD — {raw_cloud_pct:.1f}% cloud'); axes[1].axis('off')
        axes[2].imshow(refined_cld, cmap='gray', vmin=0, vmax=1); axes[2].set_title(f'ABCD-RF refined — {refined_cloud_pct:.1f}% cloud'); axes[2].axis('off')
        axes[3].imshow(_apply_stretch(disp_out)); axes[3].set_title('Output product (black = masked)'); axes[3].axis('off')
    else:
        axes[0].imshow(_apply_stretch(disp_in));  axes[0].set_title('Input SWIR (B12/B11/B9)'); axes[0].axis('off')
        axes[1].imshow(cld_norm, cmap='gray', vmin=0, vmax=1); axes[1].set_title(f'Sen2Cor CLD — {raw_cloud_pct:.1f}% cloud'); axes[1].axis('off')
        axes[2].imshow(_apply_stretch(disp_out)); axes[2].set_title('Output product (black = masked)'); axes[2].axis('off')

    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  PNG → {png_path}')


# ===========================================================================
# Filename matching helpers
# ===========================================================================

def _parse_sentinel2_key(filename: str) -> Optional[Tuple[str, str]]:
    parts = Path(filename).stem.split('_')
    try:
        date_str = parts[2][:8]
        tile_id  = next(p for p in parts if p.startswith('T') and len(p) == 6)
        return tile_id, date_str
    except (IndexError, StopIteration):
        return None


def _build_l1c_lookup(l1c_dir: str) -> Dict[Tuple[str, str], str]:
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

def _load_envi_bands(bin_path: str) -> Optional[np.ndarray]:
    try:
        ds = gdal.Open(str(bin_path), gdal.GA_ReadOnly)
    except RuntimeError:
        return None
    if ds is None:
        return None
    K  = ds.RasterCount
    H  = ds.RasterYSize
    W  = ds.RasterXSize
    out = np.empty((H, W, K), dtype=np.float32)
    for i in range(K):
        out[..., i] = ds.GetRasterBand(i + 1).ReadAsArray()
    ds = None
    return out


def _build_masks_from_cld_scl(
    cld_raw: np.ndarray,
    scl_arr: np.ndarray,
    cloud_threshold: float,
    refined_cld: np.ndarray,
    refined_shadow: Optional[np.ndarray] = None,
    shadow_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    H20, W20 = cld_raw.shape

    cld_raw_norm = cld_raw.astype(np.float32)
    if cld_raw_norm.max() > 1.0:
        cld_raw_norm = cld_raw_norm / 100.0
    raw_cloud_pct = (cld_raw_norm >= cloud_threshold).mean() * 100

    cloud_mask_20m    = refined_cld >= cloud_threshold
    refined_cloud_pct = cloud_mask_20m.mean() * 100

    scl_nodata_20m = np.zeros((H20, W20), dtype=bool)
    for cls in SCL_NODATA_CLASSES:
        scl_nodata_20m |= (scl_arr == cls)

    invalid_20m = cloud_mask_20m | scl_nodata_20m

    if refined_shadow is not None:
        shadow_mask_20m = refined_shadow >= shadow_threshold
        invalid_20m = invalid_20m | shadow_mask_20m
        print(f'  Shadow coverage (RF refined)  : {shadow_mask_20m.mean()*100:.1f}%')

    print(f'  Cloud coverage (Sen2Cor raw)  : {raw_cloud_pct:.1f}%')
    print(f'  Cloud coverage (ABCD refined) : {refined_cloud_pct:.1f}%')
    print(f'  SCL nodata                    : {scl_nodata_20m.mean()*100:.1f}%')
    print(f'  Total invalid                 : {invalid_20m.mean()*100:.1f}%')

    return cloud_mask_20m, invalid_20m, raw_cloud_pct, refined_cloud_pct


def _png_from_existing(
    out_bin: Path,
    out_png: Path,
    l2a_path: str,
    cloud_threshold: float,
) -> bool:
    print('  .bin exists, .png missing — re-extracting CLD/SCL only (no RF) ...')

    l2a_data = extract_l2a_masks_and_swir(l2a_path, extract_swir=False)
    if l2a_data is None:
        print('  [FAIL] Could not re-extract cloud masks for PNG.')
        return False

    cld_raw  = l2a_data['cld']
    scl_arr  = l2a_data['scl']
    cld_geo  = l2a_data['cld_geo']
    cld_proj = l2a_data['cld_proj']
    H20, W20 = cld_raw.shape

    cld_raw_norm = cld_raw.astype(np.float32)
    if cld_raw_norm.max() > 1.0:
        cld_raw_norm = cld_raw_norm / 100.0
    raw_cloud_pct = (cld_raw_norm >= cloud_threshold).mean() * 100

    scl_nodata_20m = np.zeros((H20, W20), dtype=bool)
    for cls in SCL_NODATA_CLASSES:
        scl_nodata_20m |= (scl_arr == cls)
    cloud_mask_20m = cld_raw_norm >= cloud_threshold
    invalid_20m    = cloud_mask_20m | scl_nodata_20m

    print(f'  Cloud coverage (Sen2Cor raw) : {raw_cloud_pct:.1f}%')
    print(f'  SCL nodata                   : {scl_nodata_20m.mean()*100:.1f}%')
    print(f'  Total invalid                : {invalid_20m.mean()*100:.1f}%')

    swir_out = _load_envi_bands(out_bin)
    if swir_out is None:
        print(f'  [FAIL] Could not load {out_bin} for PNG.')
        return False

    Hs, Ws = swir_out.shape[:2]
    if (Hs, Ws) != (H20, W20):
        ds_tmp        = gdal.Open(str(out_bin), gdal.GA_ReadOnly)
        swir_geo_out  = ds_tmp.GetGeoTransform()
        swir_proj_out = ds_tmp.GetProjection()
        ds_tmp        = None
        swir_20m      = np.zeros((H20, W20, min(3, swir_out.shape[2])), dtype=np.float32)
        for band_i in range(swir_20m.shape[2]):
            swir_20m[..., band_i] = _resample_to_target(
                swir_out[..., band_i],
                _make_mem_ds(swir_out[..., band_i], swir_geo_out, swir_proj_out),
                W20, H20, cld_geo, cld_proj, 'bilinear')
    else:
        swir_20m = swir_out[..., :3] if swir_out.shape[2] >= 3 else swir_out

    _save_png(
        str(out_png),
        cld_raw,
        np.nan_to_num(swir_20m, nan=0.0),
        swir_20m,
        raw_cloud_pct=raw_cloud_pct,
        refined_cld=None,
    )
    return True


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
    three_band: bool = False,
) -> bool:
    """
    Process one L2A scene (with optional L1C SWIR source).
    Returns True on success, False on skip/failure.
    """
    stem     = Path(l2a_path).stem
    out_bin  = Path(l2a_path).parent / (stem + '.bin')
    _stem_parts  = stem.split('_')
    _dt_prefix   = _stem_parts[2] if len(_stem_parts) > 2 else stem
    out_png  = Path(l2a_path).parent / f'{_dt_prefix}_{stem}.png'

    print(f'\n{"="*60}')
    print(f'  Scene : {stem}')

    if write_png and out_bin.exists() and not out_png.exists() and not overwrite:
        return _png_from_existing(out_bin, out_png, l2a_path, cloud_threshold)

    bin_done = out_bin.exists() and not overwrite
    png_done = (not write_png) or (out_png.exists() and not overwrite)
    if bin_done and png_done:
        print(f'  [SKIP] Output(s) already exist.')
        return False

    source_label = 'L1C' if l1c_path else 'L2A'
    print(f'  SWIR  : {source_label}')
    print(f'  Cloud : L2A (ABCD-RF{"" if use_rf else " disabled"})')
    print(f'  Bands : {"3 (B12/B11/B9)" if three_band else "4 (B12/B11/B9/B8)"}')

    # 1. Extract cloud masks from L2A
    print('  Extracting L2A cloud masks ...')
    l2a_data = extract_l2a_masks_and_swir(l2a_path, extract_swir=(l1c_path is None),
                                          three_band=three_band)
    if l2a_data is None:
        print(f'  [FAIL] Could not extract from {l2a_path}')
        return False

    cld_raw  = l2a_data['cld']
    scl_arr  = l2a_data['scl']
    cld_geo  = l2a_data['cld_geo']
    cld_proj = l2a_data['cld_proj']
    H20, W20 = cld_raw.shape

    # 2. Extract SWIR bands
    if l1c_path:
        print(f'  Extracting L1C SWIR from {Path(l1c_path).name} ...')
        swir_data = extract_l1c_swir(l1c_path, three_band=three_band)
    else:
        print('  Using L2A SWIR bands ...')
        swir_data = {
            'b12': l2a_data['b12'], 'b11': l2a_data['b11'], 'b9': l2a_data['b9'],
            'swir_geo': l2a_data['swir_geo'], 'swir_proj': l2a_data['swir_proj'],
            'swir_xsize': l2a_data['swir_xsize'], 'swir_ysize': l2a_data['swir_ysize'],
            'swir_res': l2a_data['swir_res'],
        }
        if not three_band:
            swir_data['b8'] = l2a_data['b8']

    if swir_data is None:
        print('  [FAIL] SWIR extraction failed.')
        return False

    b12       = swir_data['b12']
    b11       = swir_data['b11']
    b9        = swir_data['b9']
    b8        = swir_data.get('b8')
    swir_geo  = swir_data['swir_geo']
    swir_proj = swir_data['swir_proj']
    Hs, Ws    = swir_data['swir_ysize'], swir_data['swir_xsize']
    swir_res  = swir_data['swir_res']

    # Build 20m SWIR stack for RF input (always 3-band: B12/B11/B9)
    swir_stack_20m = np.dstack([b12, b11, b9])
    if abs(swir_res - 20.0) > 0.5 or swir_stack_20m.shape[:2] != (H20, W20):
        print(f'  Resampling SWIR stack → 20m for RF ...')
        swir_stack_20m = np.dstack([
            _resample_to_target(b12, _make_mem_ds(b12, swir_geo, swir_proj),
                                W20, H20, cld_geo, cld_proj, 'bilinear'),
            _resample_to_target(b11, _make_mem_ds(b11, swir_geo, swir_proj),
                                W20, H20, cld_geo, cld_proj, 'bilinear'),
            _resample_to_target(b9,  _make_mem_ds(b9,  swir_geo, swir_proj),
                                W20, H20, cld_geo, cld_proj, 'bilinear'),
        ])

    # 3. Refine cloud probability
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

    # 4. Build final invalid-pixel mask
    cloud_mask_20m, invalid_20m, raw_cloud_pct, refined_cloud_pct = (
        _build_masks_from_cld_scl(cld_raw, scl_arr, cloud_threshold, refined_cld))

    # 5. Upsample mask to native SWIR grid
    if invalid_20m.shape != (Hs, Ws):
        print(f'  Resampling mask from 20m → {swir_res:.1f}m SWIR grid ...')
        invalid_float = invalid_20m.astype(np.float32)
        invalid_swir  = _resample_to_target(
            invalid_float, _make_mem_ds(invalid_float, cld_geo, cld_proj),
            Ws, Hs, swir_geo, swir_proj, 'near')
        invalid_swir_bool = invalid_swir >= 0.5
    else:
        invalid_swir_bool = invalid_20m

    # 6. Apply mask and write output
    out_stack = _build_output_stack({'b12': b12, 'b11': b11, 'b9': b9, 'b8': b8}, three_band)
    out_stack[invalid_swir_bool] = np.nan
    out_stack[np.all(out_stack == 0.0, axis=-1)] = np.nan

    date_str = stem.split('_')[2][:8] if len(stem.split('_')) > 2 else stem
    band_names = _build_output_band_names(date_str, swir_res, three_band)
    write_envi(str(out_bin), out_stack, swir_geo, swir_proj, band_names)

    # 7. Optional PNG
    if write_png:
        if out_stack.shape[:2] != (H20, W20):
            out_stack_20m = np.dstack([
                _resample_to_target(
                    out_stack[..., i],
                    _make_mem_ds(out_stack[..., i], swir_geo, swir_proj),
                    W20, H20, cld_geo, cld_proj, 'bilinear')
                for i in range(out_stack.shape[2])
            ])
        else:
            out_stack_20m = out_stack

        _save_png(
            str(out_png),
            cld_raw,
            swir_stack_20m,
            out_stack_20m,
            raw_cloud_pct=raw_cloud_pct,
            refined_cld=refined_cld,
            refined_cloud_pct=refined_cloud_pct,
        )

    return True


# ===========================================================================
# MRAP helpers
# ===========================================================================

def _load_envi_stack(bin_path: str) -> Optional[np.ndarray]:
    try:
        ds = gdal.Open(bin_path)
        if ds is None:
            return None
        K = ds.RasterCount
        H, W = ds.RasterYSize, ds.RasterXSize
        out = np.empty((H, W, K), dtype=np.float32)
        for i in range(K):
            out[..., i] = ds.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
        ds = None
        return out
    except Exception:
        return None


def _composite_mrap(
    prev: Optional[np.ndarray],
    current: np.ndarray,
) -> np.ndarray:
    if prev is None:
        return current.copy()
    composite = prev.copy()
    valid = ~np.all(np.isnan(current), axis=-1)
    composite[valid] = current[valid]
    return composite


def _save_png_mrap(
    png_path: str,
    raw_cld: np.ndarray,
    refined_cld: np.ndarray,
    scl_shadow_raw: np.ndarray,
    refined_shadow: Optional[np.ndarray],
    prev_mrap_dep: Optional[np.ndarray],
    curr_mrap_dep: np.ndarray,
    prev_mrap_rf: Optional[np.ndarray],
    curr_mrap_rf: np.ndarray,
    raw_cloud_pct: float = 0.0,
    refined_cloud_pct: float = 0.0,
    cloud_threshold: float = 0.0001,
) -> None:
    """Save a 2×4 diagnostic PNG for MRAP mode."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not available — skipping PNG')
        return

    ref_h, ref_w = raw_cld.shape[:2]
    panel_inches  = 8.0
    dpi           = 100
    # Use first 3 bands for display regardless of output band count
    display_rf  = curr_mrap_rf[..., :3] if curr_mrap_rf.ndim == 3 and curr_mrap_rf.shape[2] > 3 else curr_mrap_rf
    n_bands     = display_rf.shape[2] if display_rf.ndim == 3 else 1

    lo = np.zeros(n_bands, dtype=np.float32)
    hi = np.ones(n_bands,  dtype=np.float32)
    for b in range(n_bands):
        layer = (display_rf[..., b] if display_rf.ndim == 3
                 else display_rf).astype(np.float32)
        valid = layer[np.isfinite(layer) & (layer != 0)]
        if valid.size > 0:
            lo[b] = np.percentile(valid, 2)
            hi[b] = np.percentile(valid, 98)

    def _resample_arr_to(arr: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
        src_h, src_w = arr.shape[:2]
        row_idx = (np.arange(tgt_h) * src_h / tgt_h).astype(int)
        col_idx = (np.arange(tgt_w) * src_w / tgt_w).astype(int)
        return arr[np.ix_(row_idx, col_idx)]

    def _apply_stretch(stack: np.ndarray) -> np.ndarray:
        stack = stack[..., :3] if stack.ndim == 3 and stack.shape[2] > 3 else stack
        nb  = stack.shape[2] if stack.ndim == 3 else 1
        out = np.zeros((*stack.shape[:2], nb), dtype=np.float32)
        for b in range(nb):
            layer = (stack[..., b] if stack.ndim == 3 else stack).astype(np.float32)
            scaled = (np.nan_to_num(layer, nan=0.0) - lo[b]) / np.maximum(hi[b] - lo[b], 1e-6)
            out[..., b] = np.clip(scaled, 0.0, 1.0)
        return out

    def _show_mask(ax, arr, title, cmap='gray', vmin=0, vmax=1):
        a = arr.astype(np.float32)
        if a.max() > 1.0:
            a = a / 100.0
        if a.shape[:2] != (ref_h, ref_w):
            a = _resample_arr_to(a, ref_h, ref_w)
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=11); ax.axis('off')

    def _show_rgb(ax, stack, title):
        if stack is None:
            ax.set_facecolor('#404040')
            ax.text(0.5, 0.5, 'No previous MRAP', ha='center', va='center', color='white',
                    transform=ax.transAxes, fontsize=11)
            ax.axis('off'); ax.set_title(title, fontsize=11); return
        rgb = _apply_stretch(stack)
        if rgb.shape[:2] != (ref_h, ref_w):
            rgb = np.dstack([_resample_arr_to(rgb[..., b], ref_h, ref_w)
                             for b in range(rgb.shape[2])])
        ax.imshow(rgb, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=11); ax.axis('off')

    fig, axes = plt.subplots(2, 4, figsize=(panel_inches * 4, panel_inches * 2))

    _show_mask(axes[0, 0], raw_cld, f'Sen2Cor CLD  ({raw_cloud_pct:.1f}% cloud)')
    _show_mask(axes[0, 1], scl_shadow_raw, 'SCL shadow (raw, class 3)')
    _show_rgb(axes[0, 2], prev_mrap_dep, 'Deprecated MRAP — previous timestep')
    _show_rgb(axes[0, 3], curr_mrap_dep, 'Deprecated MRAP — current timestep')

    _show_mask(axes[1, 0], refined_cld, f'ABCD-RF refined CLD  ({refined_cloud_pct:.1f}% cloud)')
    if refined_shadow is not None:
        _show_mask(axes[1, 1], refined_shadow, 'RF refined shadow prob')
    else:
        _show_mask(axes[1, 1], scl_shadow_raw, 'SCL shadow (raw, class 3)\n[RF not run]')
    _show_rgb(axes[1, 2], prev_mrap_rf, f'RF MRAP — previous timestep\n(threshold={cloud_threshold:.4g})')
    _show_rgb(axes[1, 3], curr_mrap_rf, f'RF MRAP — current timestep\n(threshold={cloud_threshold:.4g})')

    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  PNG → {png_path}')


def _mrap_compute(task: tuple) -> Tuple[int, dict]:
    """Parallelisable part of MRAP processing."""
    import io, contextlib, os as _os
    buf = io.StringIO()
    pid = _os.getpid()
    res: dict = {'ok': False, 'log': ''}

    (l2a_path, l1c_path, skip_f, offset, cloud_threshold,
     use_rf, save_model, model_dir, write_png, overwrite, write_dep,
     three_band) = task

    with contextlib.redirect_stdout(buf):
        stem        = Path(l2a_path).stem
        _stem_parts = stem.split('_')
        _dt_prefix  = _stem_parts[2] if len(_stem_parts) > 2 else stem
        out_mrap_bin = Path(l2a_path).parent / f'{stem}_MRAP.bin'
        out_png      = Path(l2a_path).parent / f'{_dt_prefix}_{stem}_MRAP.png'

        res.update(dict(stem=stem, out_mrap_bin=str(out_mrap_bin),
                        out_png=str(out_png), write_png=write_png,
                        write_dep=write_dep, l2a_path=l2a_path))

        print(f'\n{"="*60}')
        print(f'  Scene (MRAP compute) : {stem}')
        print(f'  Bands : {"3 (B12/B11/B9)" if three_band else "4 (B12/B11/B9/B8)"}')

        if out_mrap_bin.exists() and not overwrite:
            print(f'  [SKIP] {out_mrap_bin.name} already exists — will seed from it.')
            res['skip'] = True
            res['ok']   = True
            res['log']  = buf.getvalue()
            return pid, res

        res['skip'] = False

        # Step 1
        print('  Extracting L2A cloud masks ...')
        l2a_data = extract_l2a_masks_and_swir(l2a_path, extract_swir=(l1c_path is None),
                                              three_band=three_band)
        if l2a_data is None:
            print(f'  [FAIL] Could not extract from {l2a_path}')
            res['log'] = buf.getvalue(); return pid, res

        cld_raw  = l2a_data['cld']
        scl_arr  = l2a_data['scl']
        cld_geo  = l2a_data['cld_geo']
        cld_proj = l2a_data['cld_proj']
        H20, W20 = cld_raw.shape

        # Step 2
        if l1c_path:
            print(f'  Extracting L1C SWIR from {Path(l1c_path).name} ...')
            swir_data = extract_l1c_swir(l1c_path, three_band=three_band)
            if swir_data is None:
                print('  [WARN] L1C SWIR failed; falling back to L2A SWIR.')
                l2a_swir = extract_l2a_masks_and_swir(l2a_path, extract_swir=True,
                                                      three_band=three_band)
                if l2a_swir is None:
                    print('  [FAIL] L2A SWIR fallback also failed.')
                    res['log'] = buf.getvalue(); return pid, res
                l2a_data.update(l2a_swir)
                swir_data = None
        if not l1c_path or swir_data is None:
            print('  Using L2A SWIR bands ...')
            swir_data = {
                'b12': l2a_data['b12'], 'b11': l2a_data['b11'], 'b9': l2a_data['b9'],
                'swir_geo':   l2a_data['swir_geo'],  'swir_proj':  l2a_data['swir_proj'],
                'swir_xsize': l2a_data['swir_xsize'], 'swir_ysize': l2a_data['swir_ysize'],
                'swir_res':   l2a_data['swir_res'],
            }
            if not three_band:
                swir_data['b8'] = l2a_data['b8']

        b12 = swir_data['b12']; b11 = swir_data['b11']; b9 = swir_data['b9']
        b8  = swir_data.get('b8')
        swir_geo  = swir_data['swir_geo']; swir_proj = swir_data['swir_proj']
        Hs, Ws    = swir_data['swir_ysize'], swir_data['swir_xsize']
        swir_res  = swir_data['swir_res']

        # Step 3: 20m SWIR stack (always 3-band for RF)
        swir_stack_20m = np.dstack([b12, b11, b9])
        if abs(swir_res - 20.0) > 0.5 or swir_stack_20m.shape[:2] != (H20, W20):
            swir_stack_20m = np.dstack([
                _resample_to_target(b12, _make_mem_ds(b12, swir_geo, swir_proj),
                                    W20, H20, cld_geo, cld_proj, 'bilinear'),
                _resample_to_target(b11, _make_mem_ds(b11, swir_geo, swir_proj),
                                    W20, H20, cld_geo, cld_proj, 'bilinear'),
                _resample_to_target(b9,  _make_mem_ds(b9,  swir_geo, swir_proj),
                                    W20, H20, cld_geo, cld_proj, 'bilinear'),
            ])

        # Step 4: RF cloud refinement
        if use_rf:
            print('  Running ABCD-RF cloud refinement ...')
            pkl_path: Optional[str] = None
            if save_model:
                safe_stem = stem.replace(' ', '_')
                pkl_path  = str(Path(model_dir) / f'{safe_stem}_{skip_f}_{offset}_cloud.pkl')
            refined_cld = refine_cloud_mask(cld_raw, swir_stack_20m, skip_f, offset, pkl_path)
        else:
            refined_cld = cld_raw.astype(np.float32)
            if refined_cld.max() > 1.0:
                refined_cld /= 100.0

        scl_shadow_raw: np.ndarray = (scl_arr == 3).astype(np.float32)
        if use_rf:
            print('  Running ABCD-RF shadow refinement ...')
            shadow_pkl: Optional[str] = None
            if save_model:
                shadow_pkl = str(Path(model_dir) / f'{safe_stem}_{skip_f}_{offset}_shadow.pkl')
            refined_shadow: Optional[np.ndarray] = refine_shadow_mask(
                scl_arr, swir_stack_20m, skip_f, offset, shadow_pkl)
        else:
            refined_shadow = None

        # Step 5: Final mask
        cloud_mask_20m, invalid_20m, raw_cloud_pct, refined_cloud_pct = (
            _build_masks_from_cld_scl(cld_raw, scl_arr, cloud_threshold,
                                      refined_cld, refined_shadow))

        # Step 6: Upsample mask
        if invalid_20m.shape != (Hs, Ws):
            invalid_float = invalid_20m.astype(np.float32)
            invalid_swir_bool = _resample_to_target(
                invalid_float, _make_mem_ds(invalid_float, cld_geo, cld_proj),
                Ws, Hs, swir_geo, swir_proj, 'near') >= 0.5
        else:
            invalid_swir_bool = invalid_20m

        # Step 7: Apply mask
        bands_dict = {'b12': b12, 'b11': b11, 'b9': b9, 'b8': b8}
        out_stack = _build_output_stack(bands_dict, three_band)
        out_stack[invalid_swir_bool] = np.nan
        out_stack[np.all(out_stack == 0.0, axis=-1)] = np.nan

        # Step 7b: Deprecated SCL-only stack
        dep_stack: Optional[np.ndarray] = None
        if write_dep:
            dep_invalid_20m = np.zeros((H20, W20), dtype=bool)
            for _cls in SCL_NODATA_CLASSES | SCL_CLOUD_CLASSES:
                dep_invalid_20m |= (scl_arr == _cls)
            if dep_invalid_20m.shape != (Hs, Ws):
                dep_inv_float = dep_invalid_20m.astype(np.float32)
                dep_invalid_swir = _resample_to_target(
                    dep_inv_float, _make_mem_ds(dep_inv_float, cld_geo, cld_proj),
                    Ws, Hs, swir_geo, swir_proj, 'near') >= 0.5
            else:
                dep_invalid_swir = dep_invalid_20m
            dep_stack = _build_output_stack(bands_dict, three_band)
            dep_stack[dep_invalid_swir] = np.nan
            dep_stack[np.all(dep_stack == 0.0, axis=-1)] = np.nan

        res.update(dict(
            ok=True, skip=False,
            out_stack=out_stack, dep_stack=dep_stack,
            cld_raw=cld_raw, refined_cld=refined_cld,
            scl_shadow_raw=scl_shadow_raw, refined_shadow=refined_shadow,
            raw_cloud_pct=raw_cloud_pct, refined_cloud_pct=refined_cloud_pct,
            swir_geo=swir_geo, swir_proj=swir_proj, swir_res=swir_res,
            Hs=Hs, Ws=Ws, cld_geo=cld_geo, cld_proj=cld_proj, H20=H20, W20=W20,
            three_band=three_band,
        ))

    res['log'] = buf.getvalue()
    return pid, res


def _resolve_prev_mrap(prev) -> Optional[np.ndarray]:
    """Resolve a deferred MRAP reference.

    prev can be:
      - None           → return None
      - np.ndarray     → return as-is
      - str (filepath) → load from disk (deferred skip)
    """
    if prev is None:
        return None
    if isinstance(prev, np.ndarray):
        return prev
    # str path — deferred load from a skipped timestep
    print(f'  [SEED] Loading deferred MRAP: {prev}')
    return _load_envi_stack(prev)


def _mrap_composite_and_write(
    res: dict,
    prev_mrap_rf,    # Optional[np.ndarray | str]  — array or deferred path
    prev_mrap_dep,   # Optional[np.ndarray | str]
    write_threads: list,
    cloud_threshold: float,
) -> Tuple[bool, object, object]:
    """Serial part of MRAP processing.

    prev_mrap_rf / prev_mrap_dep may be numpy arrays, filepath strings
    (deferred from a skipped timestep), or None.  When a skip is
    encountered, the path is stored without loading — it is only
    loaded when the next non-skip timestep needs to seed from it.
    This avoids reading every intermediate MRAP.bin during long
    runs of already-processed timesteps.

    Returns (success, new_prev_rf, new_prev_dep) where the prev
    values may again be arrays, paths, or None.
    """
    import threading

    if not res['ok']:
        return False, prev_mrap_rf, prev_mrap_dep

    if res.get('skip'):
        # Defer loading — just remember the path.  The file will only be
        # read if/when a subsequent non-skip timestep needs to seed from it.
        return False, res['out_mrap_bin'], prev_mrap_dep

    # --- Non-skip: we actually need the previous MRAP arrays ---
    prev_mrap_rf  = _resolve_prev_mrap(prev_mrap_rf)
    prev_mrap_dep = _resolve_prev_mrap(prev_mrap_dep)

    out_stack    = res['out_stack']
    dep_stack    = res.get('dep_stack')
    write_dep    = res.get('write_dep', False)
    three_band   = res.get('three_band', False)
    swir_geo     = res['swir_geo'];  swir_proj = res['swir_proj'];  swir_res = res['swir_res']
    Hs, Ws       = res['Hs'], res['Ws']
    cld_geo      = res['cld_geo'];   cld_proj  = res['cld_proj']
    H20, W20     = res['H20'], res['W20']
    stem         = res['stem']
    out_mrap_bin = res['out_mrap_bin']
    out_png      = res['out_png']
    write_png    = res['write_png']

    def _resample_prev(stack):
        if stack is None:
            return None
        if stack.shape[:2] == (Hs, Ws):
            return stack
        return np.dstack([
            _resample_to_target(
                stack[..., i],
                _make_mem_ds(stack[..., i], swir_geo, swir_proj),
                Ws, Hs, swir_geo, swir_proj, 'near')
            for i in range(stack.shape[2])
        ])

    new_mrap_rf  = _composite_mrap(_resample_prev(prev_mrap_rf),  out_stack)
    new_mrap_dep = (_composite_mrap(_resample_prev(prev_mrap_dep), dep_stack)
                    if write_dep and dep_stack is not None else prev_mrap_dep)

    date_str = stem.split('_')[2][:8] if len(stem.split('_')) > 2 else stem
    band_names = [n + ' MRAP' for n in _build_output_band_names(date_str, swir_res, three_band)]

    def _write_bin(_arr, _path, _geo, _proj, _names):
        write_envi(_path, _arr, _geo, _proj, _names)
        print(f'  Written (MRAP) → {_path}')

    t_bin = threading.Thread(
        target=_write_bin,
        args=(new_mrap_rf, out_mrap_bin, swir_geo, swir_proj, band_names),
        daemon=True,
    )
    t_bin.start()
    write_threads.append(t_bin)

    if write_png:
        def _write_png_thread(_prev_rf, _curr_rf, _prev_dep, _curr_dep):
            def _to_20m(stack):
                if stack is None: return None
                if stack.shape[:2] == (H20, W20): return stack
                return np.dstack([
                    _resample_to_target(
                        stack[..., i], _make_mem_ds(stack[..., i], swir_geo, swir_proj),
                        W20, H20, cld_geo, cld_proj, 'bilinear')
                    for i in range(stack.shape[2])
                ])
            _save_png_mrap(
                out_png,
                res['cld_raw'], res['refined_cld'],
                res['scl_shadow_raw'], res.get('refined_shadow'),
                _to_20m(_prev_dep), _to_20m(_curr_dep),
                _to_20m(_prev_rf),  _to_20m(_curr_rf),
                raw_cloud_pct=res['raw_cloud_pct'],
                refined_cloud_pct=res['refined_cloud_pct'],
                cloud_threshold=cloud_threshold,
            )

        t_png = threading.Thread(
            target=_write_png_thread,
            args=(prev_mrap_rf, new_mrap_rf, prev_mrap_dep, new_mrap_dep),
            daemon=True,
        )
        t_png.start()
        write_threads.append(t_png)

    return True, new_mrap_rf, new_mrap_dep


# ===========================================================================
# Scene discovery
# ===========================================================================

def find_l2a_files(l2a_dir: str) -> list[str]:
    files = []
    for entry in sorted(Path(l2a_dir).iterdir()):
        name = entry.name
        if ('MSIL2A' in name or 'MSI2A' in name) and (
                name.endswith('.zip') or name.endswith('.SAFE')):
            files.append(str(entry))
    return files


# ===========================================================================
# Simple parallel dispatcher
# ===========================================================================

def _run_parallel(
    tasks: list,
    n_workers: int,
) -> Tuple[int, int, list]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_ok       = 0
    n_skip     = 0
    retry_list = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(_worker, t): t for t in tasks}
        for f in as_completed(future_to_task):
            task = future_to_task[f]
            try:
                pid, ok, log = f.result()
            except Exception as exc:
                ok  = False
                log = f'  [EXCEPTION in future] {exc}\n'

            sys.stdout.write(log)
            sys.stdout.flush()

            if ok:
                n_ok += 1
            elif '[EXCEPTION' in log:
                retry_list.append(task)
            else:
                n_skip += 1

    return n_ok, n_skip, retry_list


def _worker(task: tuple) -> Tuple[int, bool, str]:
    import io, contextlib, os as _os
    buf    = io.StringIO()
    result = False
    pid    = _os.getpid()
    try:
        with contextlib.redirect_stdout(buf):
            result = process_scene(*task)
    except Exception as exc:
        import traceback
        buf.write(f'  [EXCEPTION] {exc}\n')
        buf.write(traceback.format_exc())
    return pid, result, buf.getvalue()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    import time as _time
    _t_main_start = _time.monotonic()

    parser = argparse.ArgumentParser(
        description='Extract cloud-masked SWIR (B12, B11, B9[, B8]) from Sentinel-2 zips.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--l2_dir', default=None,
        help='Directory of L2A zip/SAFE files. Default: current working directory.')
    parser.add_argument('--l1_dir', default=None,
        help='Optional directory of L1C zip/SAFE files.')
    parser.add_argument('--three_band', action='store_true',
        help='Output only B12, B11, B9 (omit B8). Default is 4 bands.')
    parser.add_argument('--skip_f', type=int, default=5000,
        help='RF training pixel stride (default: 5000).')
    parser.add_argument('--offset', type=int, default=0,
        help='RF training pixel offset (default: 0).')
    parser.add_argument('--cloud_threshold', type=float, default=0.0001,
        help='Probability threshold for cloud masking (default: 0.0001).')
    parser.add_argument('--no_rf', action='store_true',
        help='Skip RF refinement and use raw CLD probability directly.')
    parser.add_argument('--save_model', action='store_true',
        help='Save trained RF model (.pkl) for each scene.')
    parser.add_argument('--model_dir', default='./models',
        help='Directory for saved RF models (default: ./models).')
    parser.add_argument('--png', action='store_true',
        help='Write diagnostic PNG alongside each output .bin.')
    parser.add_argument('--overwrite', action='store_true',
        help='Re-process scenes whose output .bin already exists.')
    parser.add_argument('--mrap', action='store_true',
        help='Write MRAP composites instead of per-scene outputs.')
    parser.add_argument('--N_workers', type=int, default=32,
        help='Number of parallel worker processes (default: 32).')

    args = parser.parse_args()

    l2a_dir = os.path.abspath(args.l2_dir or os.getcwd())
    use_rf  = not args.no_rf
    three_band = args.three_band

    if not os.path.isdir(l2a_dir):
        sys.exit(f'[ERROR] l2_dir not found: {l2a_dir}')

    l2a_files = find_l2a_files(l2a_dir)
    if not l2a_files:
        sys.exit(f'[ERROR] No L2A zip/SAFE files found in {l2a_dir}')
    print(f'Found {len(l2a_files)} L2A scene(s) in {l2a_dir}')
    print(f'Output bands: {"3 (B12/B11/B9)" if three_band else "4 (B12/B11/B9/B8)"}')

    l1c_lookup: Dict[Tuple[str, str], str] = {}
    if args.l1_dir:
        l1c_dir = os.path.abspath(args.l1_dir)
        if not os.path.isdir(l1c_dir):
            sys.exit(f'[ERROR] l1_dir not found: {l1c_dir}')
        l1c_lookup = _build_l1c_lookup(l1c_dir)
        print(f'Found {len(l1c_lookup)} L1C scene(s) in {l1c_dir}')

    if args.save_model:
        os.makedirs(args.model_dir, exist_ok=True)

    write_dep = args.mrap and args.png

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
            args.png, args.overwrite, three_band,
        ))

    n_total = len(tasks)

    # ------------------------------------------------------------------ #
    # MRAP mode
    # ------------------------------------------------------------------ #
    if args.mrap:
        def _dt_key(task):
            parts = Path(task[0]).stem.split('_')
            return parts[2] if len(parts) > 2 else ''

        by_dt: Dict[str, tuple] = {}
        for task in tasks:
            parts = Path(task[0]).stem.split('_')
            dt    = parts[2] if len(parts) > 2 else ''
            proc  = parts[6] if len(parts) > 6 else ''
            if dt not in by_dt or proc > by_dt[dt][1]:
                by_dt[dt] = (task, proc)
        tasks_mrap = [v[0] for v in sorted(by_dt.values(), key=lambda x: x[1])]
        tasks_mrap.sort(key=_dt_key)
        print(f'MRAP mode: {len(tasks_mrap)} timestep(s) (sorted by acquisition datetime)')

        # Rebuild task tuples for _mrap_compute (adds write_dep and three_band)
        tasks_mrap_compute = []
        for t in tasks_mrap:
            (l2a_path, l1c_path, skip_f, offset, cloud_threshold,
             use_rf, save_model, model_dir, write_png, overwrite, _tb) = t
            tasks_mrap_compute.append((
                l2a_path, l1c_path, skip_f, offset, cloud_threshold,
                use_rf, save_model, model_dir, write_png, overwrite, write_dep,
                three_band,
            ))

        from concurrent.futures import ProcessPoolExecutor

        n_mrap     = len(tasks_mrap_compute)
        n_workers  = min(args.N_workers, n_mrap)
        print(f'MRAP compute workers: {n_workers}')

        ordered_futures = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for task in tasks_mrap_compute:
                ordered_futures.append(executor.submit(_mrap_compute, task))

            prev_mrap_rf  = None   # np.ndarray | str (deferred path) | None
            prev_mrap_dep = None   # np.ndarray | str (deferred path) | None
            write_threads: list = []
            n_ok  = 0
            n_skip = 0
            _t_mrap_loop_start = _time.monotonic()
            for future in ordered_futures:
                try:
                    _pid, res = future.result()
                except Exception as exc:
                    import traceback
                    print(f'  [EXCEPTION in MRAP compute] {exc}')
                    traceback.print_exc()
                    n_skip += 1
                    continue

                sys.stdout.write(res['log'])
                sys.stdout.flush()

                ok, prev_mrap_rf, prev_mrap_dep = _mrap_composite_and_write(
                    res, prev_mrap_rf, prev_mrap_dep, write_threads,
                    args.cloud_threshold)
                if ok:
                    n_ok += 1
                else:
                    n_skip += 1

                _done     = n_ok + n_skip
                _remain   = n_mrap - _done
                _elapsed  = _time.monotonic() - _t_mrap_loop_start
                _avg_iter = _elapsed / _done
                _eta_s    = _avg_iter * _remain
                print(
                    f'  [PROGRESS] {_done}/{n_mrap} steps done | '
                    f'{_remain} remaining | '
                    f'avg {_avg_iter:.1f}s/step | '
                    f'ETA {_eta_s:.1f}s',
                    flush=True)

        pending = [t for t in write_threads if t.is_alive()]
        if pending:
            print(f'  Waiting for {len(pending)} background write(s) to finish ...')
            for t in pending:
                t.join()

        _total_s = _time.monotonic() - _t_main_start
        print(f'\n[DONE] {n_ok} MRAP timestep(s) written, {n_skip} skipped '
              f'(total {n_mrap}) | wall time {_total_s:.1f}s')
        return

    # ------------------------------------------------------------------ #
    # Normal mode
    # ------------------------------------------------------------------ #
    n_workers = min(args.N_workers, n_total)
    print(f'Workers: {n_workers}')

    n_ok   = 0
    n_skip = 0

    if n_workers == 1:
        retry_list = []
        for task in tasks:
            _, ok, log = _worker(task)
            sys.stdout.write(log)
            sys.stdout.flush()
            if ok:
                n_ok += 1
            elif '[EXCEPTION' in log:
                retry_list.append(task)
            else:
                n_skip += 1
    else:
        n_ok, n_skip, retry_list = _run_parallel(tasks, n_workers)

    if retry_list:
        print(f'\n[RETRY] {len(retry_list)} crashed job(s) — retrying serially ...')
        for task in retry_list:
            stem = Path(task[0]).stem
            print(f'  Retrying: {stem}')
            _, ok, log = _worker(task)
            sys.stdout.write(log)
            sys.stdout.flush()
            if ok:
                n_ok   += 1
            else:
                n_skip += 1

    _total_s = _time.monotonic() - _t_main_start
    print(f'\n[DONE] {n_ok} succeeded, {n_skip} skipped/failed, '
          f'{len(retry_list)} retried (total {n_total}) | wall time {_total_s:.1f}s')


if __name__ == '__main__':
    main()

