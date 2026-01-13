'''20260112: this file exports the pkl file used in classify_nn.py to the bin format expected by 
mahalanobis_classify.py 
'''

#!/usr/bin/env python3
"""
Compute Gaussian statistics (mean, covariance) for each training rectangle
and export in binary format for Mahalanobis CUDA classifier
"""

import sys, os
import numpy as np
from osgeo import gdal, ogr
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

gdal.UseExceptions()

MIN_POLY_DIMENSION = 15
PATCH_SIZE = 7
MAX_WORKERS = 512

print_lock = threading.Lock()

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

def load_image_stack(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {path}")
    data = np.stack(
        [ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
         for i in range(ds.RasterCount)],
        axis=-1
    )
    if data.shape[2] == 4:
        data = data[:, :, :3]
    return data

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

def compute_rectangle_stats(img, rectangle):
    pad = PATCH_SIZE // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w, _ = img.shape

    x0, y0, x1, y1 = rectangle

    if (x1 - x0) < MIN_POLY_DIMENSION or (y1 - y0) < MIN_POLY_DIMENSION:
        return None

    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))

    patches = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            p = padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :].reshape(-1)
            if not np.isnan(p).any():
                patches.append(p)

    if not patches:
        return None

    patches = np.vstack(patches)
    mean = patches.mean(axis=0)
    cov = np.cov(patches, rowvar=False) + np.eye(patches.shape[1]) * 1e-5

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    return mean.astype(np.float32), inv_cov.astype(np.float32)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 export_mahalanobis_training.py annotations.shp output.bin")
        sys.exit(1)

    shp_file = sys.argv[1]
    output_file = sys.argv[2]

    training = read_training_from_shapefile(shp_file)

    shp_dir = os.path.dirname(os.path.abspath(shp_file))
    cwd = os.getcwd()

    total_rectangles = sum(len(v['rectangles']) for v in training.values())
    global_processed = 0

    exemplars = []

    image_keys = list(training.keys())

    for img_idx, src in enumerate(image_keys, 1):
        data = training[src]
        rects = data['rectangles']
        labels = data['labels']

        img_path = locate_image_by_basename(src, [shp_dir, cwd])
        if img_path is None:
            print(f"[WARNING] Could not find image: {src}")
            continue

        print(f"\n[IMAGE {img_idx}/{len(image_keys)}] Processing: {src}")
        print(f"[IMAGE {img_idx}/{len(image_keys)}] Rectangles in this file: {len(rects)}")
        print(f"[IMAGE {img_idx}/{len(image_keys)}] Loading image...")

        img = load_image_stack(img_path)

        file_processed = 0
        valid = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(compute_rectangle_stats, img, r): lbl
                for r, lbl in zip(rects, labels)
            }

            for f in as_completed(futures):
                result = f.result()
                lbl = futures[f]

                file_processed += 1
                global_processed += 1

                if result is not None:
                    mean, inv_cov = result
                    exemplars.append({
                        'label': lbl,
                        'mean': mean,
                        'inv_cov': inv_cov
                    })
                    valid += 1

                with print_lock:
                    print(
                        f"  [RECT] File: {file_processed}/{len(rects)} | "
                        f"Global: {global_processed}/{total_rectangles}"
                    )

        print(f"[IMAGE {img_idx}/{len(image_keys)}] Completed: {valid} valid exemplars")
        print(f"[TOTAL] Global progress: {global_processed}/{total_rectangles}")

    print(f"\n[STATUS] Writing {len(exemplars)} exemplars to {output_file}")

    patch_dim = exemplars[0]['mean'].shape[0]

    with open(output_file, "wb") as f:
        f.write(struct.pack('i', len(exemplars)))
        f.write(struct.pack('i', patch_dim))
        for ex in exemplars:
            f.write(struct.pack('B', ex['label']))
            ex['mean'].tofile(f)
            np.zeros((patch_dim, patch_dim), np.float32).tofile(f)
            ex['inv_cov'].tofile(f)

    print("[STATUS] Export complete")

if __name__ == "__main__":
    main()


