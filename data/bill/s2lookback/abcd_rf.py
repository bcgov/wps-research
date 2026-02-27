#!/usr/bin/env python3
'''abcd_rf.py: "A is to B as C is to D" using Random Forest regression.
Pixel-based analogy: trains RF on (A, B) pairs, applies to C to produce D.
Supports ENVI (.bin/.hdr) and GeoTIFF (.tif) input. Output is ENVI format.
Uses GDAL for raster I/O. Caches trained RF model as .pkl.
Usage: python3 abcd_rf.py A B C skip_f [offset]
A - input image (n bands), training features
B - input image (m bands), training targets
C - input image (n bands), inference features
skip_f - spatial sampling stride for training pixels
offset - spatial sampling offset (default 0)
Output D is written as ENVI .bin/.hdr with CRS/map info copied from C.
'''
import sys
import os
import pickle
import numpy as np
try:
    from osgeo import gdal
except ImportError:
    import gdal
gdal.AllRegister()
gdal.UseExceptions()

def read_raster(path):
    '''Read a raster file (ENVI or TIFF) via GDAL. Returns (data [bands, pixels], ds).'''
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        sys.exit(f"Error: cannot open {path}")
    nb = ds.RasterCount
    nr = ds.RasterYSize
    nc = ds.RasterXSize
    npx = nr * nc
    data = np.empty((nb, npx), dtype=np.float32)
    for b in range(nb):
        band = ds.GetRasterBand(b + 1)
        arr = band.ReadAsArray().astype(np.float32).ravel()
        data[b, :] = arr
    return data, ds

def is_bad(data, i):
    '''Check if pixel i is bad (NaN, Inf, or all-zero for multi-band).'''
    vals = data[:, i]
    if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
        return True
    nb = data.shape[0]
    if nb > 1 and np.all(vals == 0):
        return True
    return False

def write_envi(outpath, data_2d, ref_ds, nb_out):
    '''Write ENVI BSQ output (.bin + .hdr) with map/CRS info from ref_ds.'''
    nr = ref_ds.RasterYSize
    nc = ref_ds.RasterXSize
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(outpath, nc, nr, nb_out, gdal.GDT_Float32)
    if out_ds is None:
        sys.exit(f"Error: cannot create {outpath}")
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    for b in range(nb_out):
        out_band = out_ds.GetRasterBand(b + 1)
        arr = data_2d[b, :].reshape(nr, nc)
        out_band.WriteArray(arr)
        out_band.SetNoDataValue(float('nan'))
    out_ds.FlushCache()
    out_ds = None

def cache_key(path_a, path_b, skip_f, offset):
    '''Build cache filename from basenames of A, B plus skip_f and offset.'''
    a_base = os.path.splitext(os.path.basename(path_a))[0]
    b_base = os.path.splitext(os.path.basename(path_b))[0]
    return f"{a_base}_{b_base}_{skip_f}_{offset}.pkl"


# ---------------------------------------------------------------------------
# Array-level core — used by abcd_mask.py and any other callers that already
# have data in memory.  No file I/O here, no caching.
# ---------------------------------------------------------------------------

def bad_pixel_mask_arrays(data: np.ndarray) -> np.ndarray:
    '''
    Boolean (n_pixels,) — True where pixel should be excluded.
    Excluded if any band is NaN/Inf, or (multi-band only) all bands are zero.
    '''
    bp = np.any(~np.isfinite(data), axis=0)
    if data.shape[0] > 1:
        bp |= np.all(data == 0, axis=0)
    return bp


def fit_rf(X_train: np.ndarray, Y_train: np.ndarray):
    '''
    Fit a RandomForestRegressor on pre-built (X, Y) arrays.

    Parameters
    ----------
    X_train : (n_samples, n_features)
    Y_train : (n_samples,) or (n_samples, n_targets)

    Returns
    -------
    Trained RF model.
    '''
    print("Fitting RandomForestRegressor ...")
    try:
        from cuml.ensemble import RandomForestRegressor
        print("  (using cuML GPU implementation)")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        print("  (cuML not available — falling back to sklearn)")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    rf.fit(X_train, Y_train)
    return rf


def predict_rf(rf, C: np.ndarray, bad_pixels: np.ndarray = None) -> np.ndarray:
    '''
    Run inference with a trained RF on array C.

    Parameters
    ----------
    rf         : trained RF model
    C          : (n_bands, n_pixels) float32 — inference features
    bad_pixels : (n_pixels,) bool — pixels to skip (NaN in output).
                 If None, computed automatically from C.

    Returns
    -------
    D : (n_targets, n_pixels) float32 — predictions, NaN at bad pixels.
        If the model has a single output, shape is (1, n_pixels).
    '''
    n_px = C.shape[1]

    if bad_pixels is None:
        bad_pixels = bad_pixel_mask_arrays(C)

    good_idx = np.where(~bad_pixels)[0]

    # Determine number of outputs from a tiny probe
    probe    = rf.predict(C[:, good_idx[:1]].T)
    n_out    = 1 if probe.ndim == 1 else probe.shape[1]

    D = np.full((n_out, n_px), np.nan, dtype=np.float32)

    if len(good_idx) == 0:
        return D

    preds = rf.predict(C[:, good_idx].T)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    for b in range(n_out):
        D[b, good_idx] = preds[:, b].astype(np.float32)

    print(f"  predicted {len(good_idx)} pixels")
    return D


def abcd_rf_arrays(
    A:      np.ndarray,
    B:      np.ndarray,
    C:      np.ndarray,
    skip_f: int,
    offset: int = 0,
    pkl_path: str = None,
) -> tuple:
    '''
    Array-level A:B::C:D RF regression.  No file I/O.

    Parameters
    ----------
    A        : (n_bands_a, n_pixels) float32 — training features
    B        : (n_bands_b, n_pixels) float32 — training targets
    C        : (n_bands_a, n_pixels) float32 — inference features
    skip_f   : sampling stride
    offset   : sampling offset (default 0)
    pkl_path : if given, save/load the trained model at this path

    Returns
    -------
    (D, rf)
    D  : (n_bands_b, n_pixels) float32 — predictions
    rf : trained RF model
    '''
    # ---------------------------------------------------------------- model
    if pkl_path and os.path.isfile(pkl_path):
        print(f"Loading cached model: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            rf = pickle.load(f)
    else:
        n_px = A.shape[1]
        if A.shape[1] != B.shape[1]:
            raise ValueError("A and B must have the same number of pixels.")
        if skip_f < 1 or skip_f >= n_px:
            raise ValueError(f"Illegal skip_f={skip_f}.")

        # Bad-pixel mask on A
        bp = bad_pixel_mask_arrays(A)
        print(f"  {np.sum(bp)} bad pixels, {np.sum(~bp)} good pixels in A")

        idx      = np.arange(offset, n_px, skip_f)
        good_idx = idx[~bp[idx]]
        n_train  = len(good_idx)
        print(f"Training samples: {n_train} (skip_f={skip_f}, offset={offset})")

        if n_train == 0:
            raise ValueError("No good training pixels after sampling.")

        X_train = A[:, good_idx].T
        Y_train = B[:, good_idx].T
        np.nan_to_num(Y_train, copy=False, nan=0.0)

        rf = fit_rf(X_train, Y_train)

        if pkl_path:
            print(f"Saving model to {pkl_path}")
            with open(pkl_path, 'wb') as f:
                pickle.dump(rf, f)

    # --------------------------------------------------------------- infer
    D = predict_rf(rf, C)
    return D, rf


# ---------------------------------------------------------------------------
# Original file-based entry point — unchanged
# ---------------------------------------------------------------------------

def abcd_rf(path_a, path_b, path_c, skip_f, offset=0, write_output=False):
    pkl_name = cache_key(path_a, path_b, skip_f, offset)

    if os.path.isfile(pkl_name):
        print(f"Loading cached model: {pkl_name}")
        with open(pkl_name, 'rb') as f:
            rf = pickle.load(f)
        print("Reading C ...")
        C, ds_c = read_raster(path_c)
        nb_b = rf.n_features_out_ if hasattr(rf, 'n_features_out_') else rf.n_outputs_
    else:
        print("Reading A ...")
        A, ds_a = read_raster(path_a)
        print("Reading B ...")
        B, ds_b = read_raster(path_b)
        print("Reading C ...")
        C, ds_c = read_raster(path_c)

        nb_b = B.shape[0]

        print("Flagging bad pixels in A ...")
        bp_ab = np.any(np.isnan(A) | np.isinf(A), axis=0)
        if A.shape[0] > 1:
            bp_ab |= np.all(A == 0, axis=0)
        if bp_ab.all():
            sys.exit("Error: no good pixels in A x B")
        print(f"  {np.sum(bp_ab)} bad pixels, {np.sum(~bp_ab)} good pixels in A")

        sample_idx = np.arange(offset, A.shape[1], skip_f)
        good_mask  = ~bp_ab[sample_idx]
        sample_idx = sample_idx[good_mask]
        n_train    = len(sample_idx)
        print(f"Training samples: {n_train} (skip_f={skip_f}, offset={offset})")
        if n_train == 0:
            sys.exit("Error: no good training pixels after sampling")

        X_train = A[:, sample_idx].T
        Y_train = B[:, sample_idx].T
        np.nan_to_num(Y_train, copy=False, nan=0.0)

        print("Fitting RandomForestRegressor ...")
        try:
            from cuml.ensemble import RandomForestRegressor
            print("  (using cuML GPU implementation)")
            rf = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42)
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            print("  (cuML not available — falling back to sklearn)")
            rf = RandomForestRegressor(n_estimators=100, max_depth=None,
                                       random_state=42, n_jobs=-1)
        rf.fit(X_train, Y_train)

        print(f"Saving model to {pkl_name}")
        with open(pkl_name, 'wb') as f:
            pickle.dump(rf, f)

    print("Inference on C ...")
    D = np.full((nb_b, C.shape[1]), np.nan, dtype=np.float32)
    bp_c    = np.any(np.isnan(C) | np.isinf(C), axis=0)
    if C.shape[0] > 1:
        bp_c |= np.all(C == 0, axis=0)
    good_c  = np.where(~bp_c)[0]
    X_infer = C[:, good_c].T
    preds   = rf.predict(X_infer)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    for b in range(nb_b):
        D[b, good_c] = preds[:, b].astype(np.float32)
    print(f"  predicted {len(good_c)} pixels")

    if write_output:
        a_base     = os.path.splitext(os.path.basename(path_a))[0]
        b_base     = os.path.splitext(os.path.basename(path_b))[0]
        c_base     = os.path.splitext(os.path.basename(path_c))[0]
        out_prefix = f"abcd_{a_base}_{b_base}_{c_base}_{skip_f}_{offset}"
        out_bin    = out_prefix + ".bin"
        print(f"Writing {out_bin} ...")
        write_envi(out_bin, D, ds_c, nb_b)
        plot_bands = "1 2 3" if nb_b >= 3 else "1 1 1"
        print("Plotting ...")
        os.system(f"raster_plot.py {out_bin} {plot_bands} 1")
        print("Done.")

    return D


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit("Usage: abcd_rf.py A B C skip_f [offset]")
    abcd_rf(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 0,
            write_output=True)