#!/usr/bin/env python3
import sys, os
import warnings 
from osgeo import gdal, ogr
import numpy as np
from joblib import Parallel, delayed
import pickle

gdal.UseExceptions()
warnings.filterwarnings("ignore")
gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.SetConfigOption('CPL_DEBUG', 'OFF')
gdal.PushErrorHandler('CPLQuietErrorHandler')

# Define polygon size threshold (in pixels)
MIN_POLY_DIMENSION = 15

# ---------- classification (from Script 2) ----------
def print_progress_bar(i, total, prefix='', suffix='', length=40, fill='█'):
    pct = f"{100 * (i / float(total)):.1f}"
    filled = int(length * i // total)
    bar = fill * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {pct}% {suffix}', end='\r')
    if i == total: print()

def classify_pixel(padded, y, x, patch, mean_covs, inv_cov):
    v = padded[y:y+patch, x:x+patch, :].reshape(-1)
    s0 = s1 = np.inf
    if mean_covs[0][0] is not None and inv_cov[0] is not None:
        d = v - mean_covs[0][0]; s0 = d @ inv_cov[0] @ d
    if mean_covs[1][0] is not None and inv_cov[1] is not None:
        d = v - mean_covs[1][0]; s1 = d @ inv_cov[1] @ d
    return 1 if s1 < s0 else 0

def classify_by_gaussian_parallel(image, mean_covs, patch_size=7):
    h, w, _ = image.shape
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros((h, w), dtype=np.uint8)

    inv_cov = {}
    for lbl in (0, 1):
        mean, cov = mean_covs[lbl]
        if mean is None or cov is None:
            inv_cov[lbl] = None
        else:
            try: inv_cov[lbl] = np.linalg.inv(cov)
            except np.linalg.LinAlgError: inv_cov[lbl] = np.linalg.pinv(cov)

    def classify_row(y):
        r = np.zeros(w, dtype=np.uint8)
        for x in range(w):
            r[x] = classify_pixel(padded, y, x, patch_size, mean_covs, inv_cov)
        return r

    rows = Parallel(n_jobs=-1, backend="loky")(delayed(classify_row)(y) for y in range(h))
    for y, r in enumerate(rows, 1):
        out[y-1, :] = r
        print_progress_bar(y, h, prefix='Progress:', suffix='Done')
    return out

def compute_patch_mean_cov(image, rectangles, labels, patch_size=7):
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w, _ = image.shape
    S = {0: [], 1: []}

    for (x0, y0, x1, y1), lbl in zip(rectangles, labels):
        # Skip polygons smaller than the defined minimum size
        width = x1 - x0
        height = y1 - y0
        if width < MIN_POLY_DIMENSION or height < MIN_POLY_DIMENSION:
            print(f"Skipping small polygon: {x0, y0, x1, y1}, Width: {width}, Height: {height}")
            continue

        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(w, int(x1)); y1 = min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            continue
        for y in range(y0, y1):
            for x in range(x0, x1):
                S[lbl].append(padded[y:y+patch_size, x:x+patch_size, :].reshape(-1))

    out = {}
    for lbl in (0, 1):
        if S[lbl]:
            D = np.vstack(S[lbl])
            mean = D.mean(axis=0)
            cov = np.cov(D, rowvar=False) + np.eye(D.shape[1]) * 1e-5
            out[lbl] = (mean, cov)
        else:
            out[lbl] = (None, None)
    return out

def save_envi_classification(dataset, classification):
    driver = gdal.GetDriverByName('ENVI')
    h, w = classification.shape
    in_path = dataset.GetDescription()
    base, _ = os.path.splitext(in_path)
    out_file = f"{base}_classification.bin"
    ds_out = driver.Create(out_file, w, h, 1, gdal.GDT_Float32)
    if ds_out is None:
        print(f"Error creating {out_file}")
        return None
    gt = dataset.GetGeoTransform(); pr = dataset.GetProjection()
    if gt: ds_out.SetGeoTransform(gt)
    if pr: ds_out.SetProjection(pr)
    b = ds_out.GetRasterBand(1)
    b.WriteArray(classification.astype(np.float32))
    b.FlushCache(); ds_out.FlushCache()
    print(f"[DONE] Saved: {out_file}", flush=True)
    return out_file

def locate_image_by_basename(basename, dirs):
    for d in dirs:
        p = os.path.join(d, basename)
        if os.path.isfile(p): return p
    for d in dirs:
        for root, _, files in os.walk(d):
            if basename in files: return os.path.join(root, basename)
    return None

def load_image_stack(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None: raise RuntimeError(f"Failed to open image: {path}")
    w, h, b = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    data = np.stack([ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32) for i in range(b)], axis=-1)
    assert data.shape == (h, w, b)
    return data, ds

# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 classify.py /path/to/annotation_labels.shp image_to_classify.tif")
        sys.exit(1)

    shp = sys.argv[1]
    target_image = sys.argv[2] if len(sys.argv) > 2 else None

    shp_dir = os.path.dirname(os.path.abspath(shp))
    cwd = os.getcwd()

    training = read_training_from_shapefile(shp)
    if not training:
        print("No training rectangles found.")
        sys.exit(1)

    # GLOBAL stats from all training rectangles
    S0, S1 = [], []
    for src, d in training.items():
        img_path = locate_image_by_basename(src, [shp_dir, cwd])
        if img_path is None:
            continue
        try:
            img, _ds = load_image_stack(img_path)
        except Exception:
            continue

        pad = 7 // 2
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        h, w, _ = img.shape
        rects, labels = d['rectangles'], d['labels']
        for (x0, y0, x1, y1), lbl in zip(rects, labels):
            x0 = max(0, int(x0)); y0 = max(0, int(y0))
            x1 = min(w, int(x1)); y1 = min(h, int(y1))
            if x1 <= x0 or y1 <= y0:
                continue
            tgt = S1 if lbl == 1 else S0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    tgt.append(padded[y:y+7, x:x+7, :].reshape(-1))

    mean_covs = {0:(None,None), 1:(None,None)}
    for lbl, S in ((0, S0), (1, S1)):
        if S:
            D = np.vstack(S)
            mean = D.mean(axis=0)
            cov = np.cov(D, rowvar=False) + np.eye(D.shape[1]) * 1e-5
            mean_covs[lbl] = (mean, cov)

    if mean_covs[0][0] is None and mean_covs[1][0] is None:
        print("No valid samples across images.")
        sys.exit(1)

    # --- classify exactly ONE image (second CLI arg) ---
    if target_image:
        fname = target_image
        print(f"[START] {fname}", flush=True)

        try:
            img, ds = load_image_stack(fname)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}", flush=True)
            sys.exit(1)

        cls = classify_by_gaussian_parallel(img, mean_covs, patch_size=7)
        out = save_envi_classification(ds, cls)

        print(f"[DONE] {fname} -> {out if out else 'save_failed'}")
    else:
        # Save the global stats if no target image is provided
        print("No image provided, regenerating global stats...")
        with open("global_stats.pkl", "wb") as f:
            pickle.dump(mean_covs, f)
        print("[DONE] Global stats saved to global_stats.pkl")


if __name__ == "__main__":
    main()


