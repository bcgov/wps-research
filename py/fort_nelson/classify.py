#!/usr/bin/env python3
# Script 3 — Command-line classifier using rectangles from Script 1's shapefile.
# Usage: python3 classify.py /path/to/annotation_labels.shp

import sys, os
from osgeo import gdal, ogr
import numpy as np
from joblib import Parallel, delayed

# ---------------- Classification functions (from Script 2) ----------------

def print_progress_bar(i, total, prefix='', suffix='', length=50, fill='█'):
    pct = f"{100 * (i / float(total)):.1f}"
    filled = int(length * i // total)
    bar = fill * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {pct}% {suffix}', end='\r')
    if i == total: print()

def classify_pixel(padded, y, x, patch_size, mean_covs, inv_cov):
    patch_flat = padded[y:y+patch_size, x:x+patch_size, :].flatten()
    scores = {}
    for lbl in (0, 1):
        mean, _ = mean_covs[lbl]
        if mean is None or inv_cov[lbl] is None:
            scores[lbl] = np.inf
        else:
            diff = patch_flat - mean
            scores[lbl] = diff @ inv_cov[lbl] @ diff.T
    return 1 if scores[1] < scores[0] else 0

def classify_by_gaussian_parallel(image, mean_covs, patch_size=7):
    h, w, b = image.shape
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
        row = np.zeros(w, dtype=np.uint8)
        for x in range(w):
            row[x] = classify_pixel(padded, y, x, patch_size, mean_covs, inv_cov)
        return row
    print("Starting classification using all CPU cores...")
    results = Parallel(n_jobs=-1, backend="loky")(delayed(classify_row)(y) for y in range(h))
    for y, row in enumerate(results, 1):
        out[y-1, :] = row
        print_progress_bar(y, h, prefix='Classification Progress:', suffix='Complete')
    print("Classification completed.")
    return out

def compute_patch_mean_cov(image, rectangles, labels, patch_size=7):
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w, _ = image.shape
    samples = {0: [], 1: []}
    for (x0, y0, x1, y1), lbl in zip(rectangles, labels):
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(w, int(x1)); y1 = min(h, int(y1))
        if x1 <= x0 or y1 <= y0: continue
        for y in range(y0, y1):
            for x in range(x0, x1):
                samples[lbl].append(padded[y:y+patch_size, x:x+patch_size, :].flatten())
    mean_covs = {}
    for lbl in (0, 1):
        if samples[lbl]:
            data = np.vstack(samples[lbl])
            mean = np.mean(data, axis=0)
            cov = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * 1e-5
            mean_covs[lbl] = (mean, cov)
        else:
            mean_covs[lbl] = (None, None)
    return mean_covs

def save_envi_classification(dataset, classification):
    # Output: <original_basename>_classification.bin (Float32), copy GT/Projection
    driver = gdal.GetDriverByName('ENVI')
    h, w = classification.shape
    in_path = dataset.GetDescription()
    base, _ = os.path.splitext(in_path)
    out_file = f"{base}_classification.bin"
    out_ds = driver.Create(out_file, w, h, 1, gdal.GDT_Float32)
    if out_ds is None:
        print(f"Error creating output file {out_file}")
        return None
    gt = dataset.GetGeoTransform(); proj = dataset.GetProjection()
    if gt: out_ds.SetGeoTransform(gt)
    if proj: out_ds.SetProjection(proj)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(classification.astype(np.float32))
    band.FlushCache(); out_ds.FlushCache()
    print(f"Classification saved to {out_file}")
    return out_file

# ---------------- Shapefile parsing (Script 1 format) ----------------

def parse_coords_img_to_rect(coords_img_str):
    # COORDS_IMG: "c,r;c,r;..." → rectangle via min/max of vertices (x=c, y=r)
    if not coords_img_str or coords_img_str.upper() == "NO_RASTER":
        return None
    try:
        pts = []
        for tok in coords_img_str.split(';'):
            tok = tok.strip()
            if not tok: continue
            c_str, r_str = tok.split(',')
            c = int(round(float(c_str))); r = int(round(float(r_str)))
            pts.append((c, r))
        if not pts: return None
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        if x1 == x0 or y1 == y0: return None
        return (x0, y0, x1, y1)
    except Exception:
        return None

def read_training_from_shapefile(shp_path):
    # Returns { SRC_IMAGE_basename: {'rectangles':[...], 'labels':[...]} }
    ds = ogr.Open(shp_path, 0)
    if ds is None: raise RuntimeError(f"Failed to open shapefile: {shp_path}")
    lyr = ds.GetLayer(0)
    training = {}
    for feat in lyr:
        cls = feat.GetField("CLASS")
        src = feat.GetField("SRC_IMAGE")
        coords_img = feat.GetField("COORDS_IMG")
        if not src: continue
        lbl = 1 if isinstance(cls, str) and cls.strip().upper() == "POSITIVE" else \
              0 if isinstance(cls, str) and cls.strip().upper() == "NEGATIVE" else None
        if lbl is None: continue
        rect = parse_coords_img_to_rect(coords_img)
        if rect is None: continue
        if src not in training: training[src] = {'rectangles': [], 'labels': []}
        training[src]['rectangles'].append(rect)
        training[src]['labels'].append(lbl)
    return training

# ---------------- Image I/O ----------------

def locate_image_by_basename(basename, dirs):
    for d in dirs:
        p = os.path.join(d, basename)
        if os.path.isfile(p): return p
    for d in dirs:
        for root, _, files in os.walk(d):
            if basename in files: return os.path.join(root, basename)
    return None

def load_image_stack(image_path):
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None: raise RuntimeError(f"Failed to open image: {image_path}")
    w, h, b = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    data = np.stack([ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32) for i in range(b)], axis=-1)
    assert data.shape == (h, w, b)
    return data, ds

# ---------------- Main ----------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 classify.py /path/to/annotation_labels.shp"); sys.exit(1)
    shp_path = sys.argv[1]
    shp_dir = os.path.dirname(os.path.abspath(shp_path)); cwd = os.getcwd()
    print(f"Reading training rectangles from shapefile: {shp_path}")
    training = read_training_from_shapefile(shp_path)
    if not training:
        print("No valid training rectangles found in shapefile."); sys.exit(1)
    print(f"Found training for {len(training)} image(s): {list(training.keys())}")
    for src_base, data in training.items():
        print(f"\nProcessing image: {src_base}")
        image_path = locate_image_by_basename(src_base, [shp_dir, cwd])
        if image_path is None:
            print(f"  Could not locate image '{src_base}' in {shp_dir} or {cwd}. Skipping."); continue
        try:
            image, ds = load_image_stack(image_path)
        except Exception as e:
            print(f"  Failed to open image '{image_path}': {e}. Skipping."); continue
        rects, labels = data['rectangles'], data['labels']
        print(f"  Using {len(rects)} rectangles ({labels.count(1)} POS, {labels.count(0)} NEG)")
        mean_covs = compute_patch_mean_cov(image, rects, labels, patch_size=7)
        if mean_covs[0][0] is None and mean_covs[1][0] is None:
            print("  No valid samples to compute statistics. Skipping."); continue
        classification = classify_by_gaussian_parallel(image, mean_covs, patch_size=7)
        out_file = save_envi_classification(ds, classification)
        if out_file: print(f"  Done: {out_file}")
    print("\nAll done.")

if __name__ == "__main__":

    main()



