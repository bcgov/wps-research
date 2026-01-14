'''20260113: Euclidean nearest-neighbor exporter
Each patch from the moving window becomes an exemplar (flattened patch only, no covariance)

Binary format:
- n_exemplars (int)
- patch_dim (int)
- For each exemplar:
  - label (1 byte)
  - patch (patch_dim floats)
'''

#!/usr/bin/env python3
"""
Export individual patches as exemplars for Euclidean NN CUDA classifier

Usage: python3 export_euclidean_training.py annotations.shp training_euclidean.bin
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import numpy as np
from osgeo import gdal, ogr
import struct
from joblib import Parallel, delayed

gdal.UseExceptions()

# ============ GLOBAL PARAMETERS ============
USE_FIRST_N_BANDS = 3
MIN_POLY_DIMENSION = 15
PATCH_SIZE = 7
TARGET_N_EXEMPLARS = 50  # Target number of exemplars per rectangle
N_THREADS = 16
# ===========================================

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
    """Read shapefile and group rectangles by source image"""
    ds = ogr.Open(shp, 0)
    lyr = ds.GetLayer(0)
    by_image = {}
    
    for f in lyr:
        cls = f.GetField("CLASS")
        src = f.GetField("SRC_IMAGE")
        rect = parse_coords_img_to_rect(f.GetField("COORDS_IMG"))
        if not src or rect is None:
            continue
        lbl = 1 if cls.strip().upper() == "POSITIVE" else 0
        
        if src not in by_image:
            by_image[src] = {'rectangles': [], 'labels': []}
        
        by_image[src]['rectangles'].append(rect)
        by_image[src]['labels'].append(lbl)
    
    return by_image

def load_image_stack(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {path}")
    
    n_bands = min(ds.RasterCount, USE_FIRST_N_BANDS)
    data = np.stack(
        [ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
         for i in range(n_bands)],
        axis=-1
    )
    return data, ds

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

def extract_patches_from_rectangle(padded_img, rectangle, label, h, w):
    """Extract individual patches from a training rectangle"""
    x0, y0, x1, y1 = rectangle
    
    if (x1 - x0) < MIN_POLY_DIMENSION or (y1 - y0) < MIN_POLY_DIMENSION:
        return []
    
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))
    
    rect_width = x1 - x0
    rect_height = y1 - y0
    total_pixels = rect_width * rect_height
    
    # Calculate stride to get approximately TARGET_N_EXEMPLARS samples
    if total_pixels <= TARGET_N_EXEMPLARS:
        stride = 1
    else:
        stride = int(np.sqrt(total_pixels / TARGET_N_EXEMPLARS))
        stride = max(1, stride)
    
    exemplars = []
    
    for y in range(y0, y1, stride):
        for x in range(x0, x1, stride):
            patch = padded_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :].reshape(-1)
            
            if not np.isnan(patch).any():
                exemplars.append({
                    'label': label,
                    'patch': patch.astype(np.float32)
                })
    
    return exemplars

def parfor(my_function, my_inputs, n_thread=N_THREADS):
    if n_thread == 1 or len(my_inputs) == 0:
        return [my_function(inp) for inp in my_inputs]
    else:
        return Parallel(n_jobs=n_thread)(delayed(my_function)(inp) for inp in my_inputs)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 export_euclidean_training.py annotations.shp output.bin")
        sys.exit(1)
    
    shp_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"[STATUS] USE_FIRST_N_BANDS = {USE_FIRST_N_BANDS}")
    print(f"[STATUS] TARGET_N_EXEMPLARS = {TARGET_N_EXEMPLARS} per rectangle")
    print(f"[STATUS] Reading shapefile: {shp_file}")
    training_by_image = read_training_from_shapefile(shp_file)
    
    shp_dir = os.path.dirname(os.path.abspath(shp_file))
    cwd = os.getcwd()
    
    image_files = list(training_by_image.keys())
    total_images = len(image_files)
    total_rectangles = sum(len(data['rectangles']) for data in training_by_image.values())
    
    print(f"[STATUS] Found {total_images} image files with {total_rectangles} total rectangles")
    print(f"[STATUS] Processing with {N_THREADS} parallel workers")
    print(f"[STATUS] Each patch becomes an individual exemplar\n")
    
    all_exemplars = []
    global_processed = 0
    
    for img_idx, image_file in enumerate(image_files, 1):
        data = training_by_image[image_file]
        n_rectangles = len(data['rectangles'])
        
        print(f"[IMAGE {img_idx}/{total_images}] Processing: {image_file}")
        print(f"[IMAGE {img_idx}/{total_images}] Rectangles in this file: {n_rectangles}")
        
        img_path = locate_image_by_basename(image_file, [shp_dir, cwd])
        if img_path is None:
            print(f"[WARNING] Could not find image: {image_file}")
            global_processed += n_rectangles
            continue
        
        print(f"[IMAGE {img_idx}/{total_images}] Loading and padding image...")
        img, _ = load_image_stack(img_path)
        h, w, c = img.shape
        print(f"[IMAGE {img_idx}/{total_images}] Using {c} bands")
        
        pad = PATCH_SIZE // 2
        padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        
        print(f"[IMAGE {img_idx}/{total_images}] Extracting patches from {n_rectangles} rectangles...")
        
        inputs = [
            (padded_img, rect, label, h, w)
            for rect, label in zip(data['rectangles'], data['labels'])
        ]
        
        results = parfor(lambda x: extract_patches_from_rectangle(*x), inputs, N_THREADS)
        
        rect_exemplar_counts = []
        for rect_exemplars in results:
            rect_exemplar_counts.append(len(rect_exemplars))
            all_exemplars.extend(rect_exemplars)
        
        global_processed += n_rectangles
        
        total_from_image = sum(rect_exemplar_counts)
        print(f"[IMAGE {img_idx}/{total_images}] Extracted {total_from_image} exemplars from {n_rectangles} rectangles")
        print(f"[TOTAL] Running total: {len(all_exemplars)} exemplars\n")
    
    n_exemplars = len(all_exemplars)
    patch_dim = all_exemplars[0]['patch'].shape[0] if n_exemplars > 0 else PATCH_SIZE * PATCH_SIZE * USE_FIRST_N_BANDS
    
    print(f"\n[STATUS] Total exemplars generated: {n_exemplars}")
    print(f"[STATUS] Patch dimension: {patch_dim}")
    
    # Write binary file (simplified format: no covariance)
    print(f"[STATUS] Writing to {output_file}...")
    
    with open(output_file, "wb") as f:
        f.write(struct.pack('i', n_exemplars))
        f.write(struct.pack('i', patch_dim))
        
        for i, ex in enumerate(all_exemplars):
            if i % 10000 == 0:
                print(f"  Writing exemplar {i}/{n_exemplars}...", end='\r')
            
            f.write(struct.pack('B', ex['label']))
            ex['patch'].tofile(f)
    
    print(f"\n[STATUS] Export complete!")
    
    n_positive = sum(1 for ex in all_exemplars if ex['label'] == 1)
    n_negative = sum(1 for ex in all_exemplars if ex['label'] == 0)
    print(f"[STATUS] Class distribution: {n_positive} positive, {n_negative} negative")
    
    file_size_mb = (8 + n_exemplars * (1 + patch_dim * 4)) / (1024 * 1024)
    print(f"[STATUS] File size: {file_size_mb:.1f} MB")

if __name__ == "__main__":
    main()
