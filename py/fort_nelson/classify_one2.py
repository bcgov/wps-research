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

MIN_POLY_DIMENSION = 15
PATCH_SIZE = 7
GLOBAL_STATS_FILE = "global_stats.pkl"

# ---------- classification ----------
def print_progress_bar(i, total, prefix='', suffix='', length=40, fill='█'):
    pct = f"{100 * (i / float(total)):.1f}"
    filled = int(length * i // total)
    bar = fill * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {pct}% {suffix}', end='\r')
    if i == total:
        print()

def classify_pixel(padded, y, x, patch, mean_covs, inv_cov):
    v = padded[y:y+patch, x:x+patch, :].reshape(-1)
    s0 = s1 = np.inf
    if mean_covs[0][0] is not None:
        d = v - mean_covs[0][0]
        s0 = d @ inv_cov[0] @ d
    if mean_covs[1][0] is not None:
        d = v - mean_covs[1][0]
        s1 = d @ inv_cov[1] @ d
    return 1 if s1 < s0 else 0

def classify_by_gaussian_parallel(image, mean_covs, patch_size=PATCH_SIZE):
    h, w, _ = image.shape
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros((h, w), dtype=np.uint8)

    inv_cov = {}
    for lbl in (0, 1):
        mean, cov = mean_covs[lbl]
        if mean is None:
            inv_cov[lbl] = None
        else:
            try:
                inv_cov[lbl] = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov[lbl] = np.linalg.pinv(cov)

    def classify_row(y):
        r = np.zeros(w, dtype=np.uint8)
        for x in range(w):
            r[x] = classify_pixel(padded, y, x, patch_size, mean_covs, inv_cov)
        return r

    rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(classify_row)(y) for y in range(h)
    )

    for y, r in enumerate(rows, 1):
        out[y-1, :] = r
        print_progress_bar(y, h, prefix="Classifying:", suffix="Done")

    return out

def compute_patch_mean_cov(image, rectangles, labels, patch_size=PATCH_SIZE):
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w, _ = image.shape
    S = {0: [], 1: []}

    for (x0, y0, x1, y1), lbl in zip(rectangles, labels):
        if (x1 - x0) < MIN_POLY_DIMENSION or (y1 - y0) < MIN_POLY_DIMENSION:
            continue

        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(w, int(x1)); y1 = min(h, int(y1))

        for y in range(y0, y1):
            for x in range(x0, x1):
                S[lbl].append(
                    padded[y:y+patch_size, x:x+patch_size, :].reshape(-1)
                )

    out = {}
    for lbl in (0, 1):
        if S[lbl]:
            D = np.vstack(S[lbl])
            out[lbl] = (
                D.mean(axis=0),
                np.cov(D, rowvar=False) + np.eye(D.shape[1]) * 1e-5
            )
        else:
            out[lbl] = (None, None)

    return out

def save_envi_classification(dataset, classification):
    driver = gdal.GetDriverByName('ENVI')
    h, w = classification.shape
    base, _ = os.path.splitext(dataset.GetDescription())
    out_file = f"{base}_classification.bin"

    ds_out = driver.Create(out_file, w, h, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(dataset.GetGeoTransform())
    ds_out.SetProjection(dataset.GetProjection())
    ds_out.GetRasterBand(1).WriteArray(classification.astype(np.float32))
    ds_out.FlushCache()

    print(f"[DONE] Saved: {out_file}")
    return out_file

def locate_image_by_basename(basename, dirs):
    for d in dirs:
        p = os.path.join(d, basename)
        if os.path.isfile(p):
            return p
    for d in dirs:
        for root, _, files in os.walk(d):
            if basename in files:
                return os.path.join(root, basename)
    return None

def load_image_stack(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {path}")
    data = np.stack(
        [ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
         for i in range(ds.RasterCount)],
        axis=-1
    )
    return data, ds

# ---------- shapefile parsing ----------
def parse_coords_img_to_rect(s):
    if not s or str(s).upper() == "NO_RASTER":
        return None
    pts = [p.split(',') for p in str(s).split(';') if p.strip()]
    xs = [int(round(float(p[0]))) for p in pts]
    ys = [int(round(float(p[1]))) for p in pts]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)

def read_training_from_shapefile(shp):
    ds = ogr.Open(shp, 0)
    lyr = ds.GetLayer(0)
    out = {}
    for f in lyr:
        cls = f.GetField("CLASS")
        src = f.GetField("SRC_IMAGE")
        rect = parse_coords_img_to_rect(f.GetField("COORDS_IMG"))
        if not src or rect is None:
            continue
        lbl = 1 if cls.strip().upper() == "POSITIVE" else 0
        out.setdefault(src, {'rectangles': [], 'labels': []})
        out[src]['rectangles'].append(rect)
        out[src]['labels'].append(lbl)
    return out

# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: classify.py labels.shp [image.tif]")
        sys.exit(1)

    shp = sys.argv[1]
    target_image = sys.argv[2] if len(sys.argv) > 2 else None

    shp_dir = os.path.dirname(os.path.abspath(shp))
    cwd = os.getcwd()

    # --- classification mode ---
    if target_image:
        if not os.path.isfile(GLOBAL_STATS_FILE):
            raise RuntimeError("global_stats.pkl not found")

        print("Loading global stats...")
        with open(GLOBAL_STATS_FILE, "rb") as f:
            global_stats = pickle.load(f)

        mean_covs = {0: global_stats['mean_covs'][0],
                     1: global_stats['mean_covs'][1]}

        img, ds = load_image_stack(target_image)
        cls = classify_by_gaussian_parallel(img, mean_covs)
        save_envi_classification(ds, cls)
        return

    # --- training mode ---
    training = read_training_from_shapefile(shp)
    S = {0: [], 1: []}

    for src, d in training.items():
        img_path = locate_image_by_basename(src, [shp_dir, cwd])
        if img_path is None:
            continue
        img, _ = load_image_stack(img_path)
        out = compute_patch_mean_cov(img, d['rectangles'], d['labels'])
        for lbl in (0, 1):
            if out[lbl][0] is not None:
                S[lbl].append(out[lbl][0])

    mean_covs = {lbl: (np.mean(S[lbl], axis=0), None) for lbl in (0, 1)}
    with open(GLOBAL_STATS_FILE, "wb") as f:
        pickle.dump({'mean_covs': mean_covs}, f)

    print(f"Saved {GLOBAL_STATS_FILE}")

if __name__ == "__main__":
    main()



