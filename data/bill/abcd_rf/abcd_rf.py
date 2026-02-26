#!/usr/bin/env python3
'''abcd_rf.py: "A is to B as C is to D" using Random Forest regression.
Pixel-based analogy: trains RF on (A, B) pairs, applies to C to produce D.
Supports ENVI (.bin/.hdr) and GeoTIFF (.tif) input. Output is ENVI format.
Uses GDAL for raster I/O. Caches trained RF model as .pkl.

Usage:
    python3 abcd_rf.py A B C skip_f [offset]

    A      - input image (n bands), training features
    B      - input image (m bands), training targets
    C      - input image (n bands), inference features
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
    # Copy georeferencing from reference (C) dataset
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


def abcd_rf(path_a, path_b, path_c, skip_f, offset=0, write_output=False):

    # ------------------------------------------ train or load cached model
    pkl_name = cache_key(path_a, path_b, skip_f, offset)

    if os.path.isfile(pkl_name):
        print(f"Loading cached model: {pkl_name}")
        with open(pkl_name, 'rb') as f:
            rf = pickle.load(f)
        nb_b = rf.n_features_out_ if hasattr(rf, 'n_features_out_') else rf.n_outputs_
    else:
        # Only read A, B when we need to train
        print("Reading A ...")
        A, ds_a = read_raster(path_a)
        print("Reading B ...")
        B, ds_b = read_raster(path_b)

        nb_a, np_ab = A.shape
        nb_b, np_ab2 = B.shape

        if np_ab != np_ab2:
            sys.exit("Error: A.shape != B.shape (pixel count mismatch)")
        if skip_f < 1 or skip_f >= np_ab:
            sys.exit("Error: illegal skip_f")

        # Bad pixels in A, B
        print("Flagging bad pixels in A, B ...")
        bp_ab = np.array([is_bad(A, i) or is_bad(B, i) for i in range(np_ab)], dtype=bool)
        if bp_ab.all():
            sys.exit("Error: no good pixels in A x B")

        # Sample training set
        sample_idx = np.arange(offset, np_ab, skip_f)
        good_mask = ~bp_ab[sample_idx]
        sample_idx = sample_idx[good_mask]
        n_train = len(sample_idx)
        print(f"Training samples: {n_train}  (skip_f={skip_f}, offset={offset})")
        if n_train == 0:
            sys.exit("Error: no good training pixels after sampling")

        X_train = A[:, sample_idx].T   # (n_train, nb_a)
        Y_train = B[:, sample_idx].T   # (n_train, nb_b)

        print("Fitting RandomForestRegressor ...")
        try:
            from cuml.ensemble import RandomForestRegressor
            print("  (using cuML GPU implementation)")
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            print("  (cuML not available — falling back to sklearn)")

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1 if hasattr(RandomForestRegressor, 'n_jobs') else None,
        )
        # cuML RF doesn't have n_jobs; remove if it errors
        try:
            rf.fit(X_train, Y_train)
        except TypeError:
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42,
            )
            rf.fit(X_train, Y_train)

        print(f"Saving model to {pkl_name}")
        with open(pkl_name, 'wb') as f:
            pickle.dump(rf, f)

    # ---------------------------------------------------------------- read C
    print("Reading C ...")
    C, ds_c = read_raster(path_c)
    nb_c, np_c = C.shape

    # Bad pixels in C
    print("Flagging bad pixels in C ...")
    bp_c = np.array([is_bad(C, i) for i in range(np_c)], dtype=bool)
    if bp_c.all():
        sys.exit("Error: no good pixels in C")

    # ---------------------------------------------------------- inference
    print("Inference on C ...")
    D = np.full((nb_b, np_c), np.nan, dtype=np.float32)

    good_c = np.where(~bp_c)[0]
    X_infer = C[:, good_c].T  # (n_good, nb_a)

    preds = rf.predict(X_infer)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    for b in range(nb_b):
        D[b, good_c] = preds[:, b].astype(np.float32)
    print(f"  predicted {len(good_c)} pixels")

    # ------------------------------------------------------------- output
    if write_output:
        a_base = os.path.splitext(os.path.basename(path_a))[0]
        b_base = os.path.splitext(os.path.basename(path_b))[0]
        c_base = os.path.splitext(os.path.basename(path_c))[0]
        out_prefix = f"abcd_{a_base}_{b_base}_{c_base}_{skip_f}_{offset}"
        out_bin = out_prefix + ".bin"

        print(f"Writing {out_bin} ...")
        write_envi(out_bin, D, ds_c, nb_b)

        print("Plotting ...")
        os.system(f"raster_plot.py {out_bin} 1 2 3 1")

    print("Done.")
    return D


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit("Usage: abcd_rf.py A B C skip_f [offset]")
    abcd_rf(sys.argv[1], sys.argv[2], sys.argv[3],
            int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 0,
            write_output=True)

