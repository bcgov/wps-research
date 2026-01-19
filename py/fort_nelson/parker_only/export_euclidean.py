'''20260119: Euclidean nearest-neighbor exporter (window-based)
Each window from tiled sampling becomes an exemplar (flattened patch only, no covariance)

Binary format:
- n_exemplars (int)
- patch_dim (int)
- n_bands (int)
- For each exemplar:
  - label (1 byte)
  - patch (patch_dim floats)
'''

#!/usr/bin/env python3
"""
Export individual windows as exemplars for Euclidean NN CUDA classifier

Usage: python3 export_euclidean_training.py output.bin

Assumes:
- Input images are stack*.bin ENVI files in present directory
- Training rectangles in annotation_labels.shp in present directory
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import glob
import numpy as np
from osgeo import gdal, ogr
import struct
from joblib import Parallel, delayed

gdal.UseExceptions()

# ============ GLOBAL PARAMETERS ============
WINDOW_SIZE = 15
TARGET_N_EXEMPLARS = 50  # Target number of exemplars per rectangle
N_THREADS = 16
ANNOTATION_SHP = "annotation_labels.shp"
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
    if ds is None:
        print(f"[ERROR] Could not open shapefile: {shp}")
        sys.exit(1)
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
    """Load all bands from ENVI .bin file"""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {path}")
    
    n_bands = ds.RasterCount  # Use ALL bands
    data = np.stack(
        [ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
         for i in range(n_bands)],
        axis=-1
    )
    return data, ds, n_bands

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

def extract_windows_from_rectangle(img, rectangle, label, h, w, n_bands):
    """Extract individual windows from a training rectangle using tiled sampling"""
    x0, y0, x1, y1 = rectangle
    
    # Clamp to image bounds
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))
    
    rect_width = x1 - x0
    rect_height = y1 - y0
    
    # Skip if rectangle is too small to admit even one window
    if rect_width < WINDOW_SIZE or rect_height < WINDOW_SIZE:
        return []
    
    # Calculate how many complete windows fit in each dimension
    n_windows_x = rect_width // WINDOW_SIZE
    n_windows_y = rect_height // WINDOW_SIZE
    total_windows = n_windows_x * n_windows_y
    
    if total_windows == 0:
        return []
    
    # Calculate stride to get approximately TARGET_N_EXEMPLARS samples
    if total_windows <= TARGET_N_EXEMPLARS:
        # Sample all windows
        sample_indices = list(range(total_windows))
    else:
        # Randomly sample TARGET_N_EXEMPLARS windows
        sample_indices = np.random.choice(total_windows, TARGET_N_EXEMPLARS, replace=False)
    
    exemplars = []
    
    for idx in sample_indices:
        wy = idx // n_windows_x
        wx = idx % n_windows_x
        
        # Top-left corner of this window in image coordinates
        win_y = y0 + wy * WINDOW_SIZE
        win_x = x0 + wx * WINDOW_SIZE
        
        # Extract window (all bands)
        window = img[win_y:win_y+WINDOW_SIZE, win_x:win_x+WINDOW_SIZE, :].reshape(-1)
        
        if not np.isnan(window).any():
            exemplars.append({
                'label': label,
                'patch': window.astype(np.float32)
            })
    
    return exemplars

def parfor(my_function, my_inputs, n_thread=N_THREADS):
    if n_thread == 1 or len(my_inputs) == 0:
        return [my_function(inp) for inp in my_inputs]
    else:
        return Parallel(n_jobs=n_thread)(delayed(my_function)(inp) for inp in my_inputs)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 export_euclidean_training.py output.bin")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # Check for required shapefile
    if not os.path.exists(ANNOTATION_SHP):
        print(f"[ERROR] Required file not found: {ANNOTATION_SHP}")
        sys.exit(1)
    
    print(f"[STATUS] WINDOW_SIZE = {WINDOW_SIZE}")
    print(f"[STATUS] TARGET_N_EXEMPLARS = {TARGET_N_EXEMPLARS} per rectangle")
    print(f"[STATUS] Reading shapefile: {ANNOTATION_SHP}")
    training_by_image = read_training_from_shapefile(ANNOTATION_SHP)
    
    cwd = os.getcwd()
    
    image_files = list(training_by_image.keys())
    total_images = len(image_files)
    total_rectangles = sum(len(data['rectangles']) for data in training_by_image.values())
    
    print(f"[STATUS] Found {total_images} image files with {total_rectangles} total rectangles")
    print(f"[STATUS] Processing with {N_THREADS} parallel workers")
    print(f"[STATUS] Each window becomes an individual exemplar\n")
    
    all_exemplars = []
    global_processed = 0
    n_bands_global = None
    
    for img_idx, image_file in enumerate(image_files, 1):
        data = training_by_image[image_file]
        n_rectangles = len(data['rectangles'])
        
        print(f"[IMAGE {img_idx}/{total_images}] Processing: {image_file}")
        print(f"[IMAGE {img_idx}/{total_images}] Rectangles in this file: {n_rectangles}")
        
        img_path = locate_image_by_basename(image_file, [cwd])
        if img_path is None:
            print(f"[WARNING] Could not find image: {image_file}")
            global_processed += n_rectangles
            continue
        
        print(f"[IMAGE {img_idx}/{total_images}] Loading image...")
        img, _, n_bands = load_image_stack(img_path)
        h, w, c = img.shape
        print(f"[IMAGE {img_idx}/{total_images}] Using all {c} bands")
        
        # Track number of bands (should be consistent across all images)
        if n_bands_global is None:
            n_bands_global = c
        elif n_bands_global != c:
            print(f"[WARNING] Band count mismatch: expected {n_bands_global}, got {c}")
        
        print(f"[IMAGE {img_idx}/{total_images}] Extracting windows from {n_rectangles} rectangles...")
        
        inputs = [
            (img, rect, label, h, w, c)
            for rect, label in zip(data['rectangles'], data['labels'])
        ]
        
        results = parfor(lambda x: extract_windows_from_rectangle(*x), inputs, N_THREADS)
        
        rect_exemplar_counts = []
        for rect_exemplars in results:
            rect_exemplar_counts.append(len(rect_exemplars))
            all_exemplars.extend(rect_exemplars)
        
        global_processed += n_rectangles
        
        total_from_image = sum(rect_exemplar_counts)
        print(f"[IMAGE {img_idx}/{total_images}] Extracted {total_from_image} exemplars from {n_rectangles} rectangles")
        print(f"[TOTAL] Running total: {len(all_exemplars)} exemplars\n")
    
    n_exemplars = len(all_exemplars)
    patch_dim = all_exemplars[0]['patch'].shape[0] if n_exemplars > 0 else WINDOW_SIZE * WINDOW_SIZE * n_bands_global
    
    print(f"\n[STATUS] Total exemplars generated: {n_exemplars}")
    print(f"[STATUS] Patch dimension: {patch_dim}")
    print(f"[STATUS] Number of bands: {n_bands_global}")
    
    # Write binary file (includes n_bands for classifier to use)
    print(f"[STATUS] Writing to {output_file}...")
    
    with open(output_file, "wb") as f:
        f.write(struct.pack('i', n_exemplars))
        f.write(struct.pack('i', patch_dim))
        f.write(struct.pack('i', n_bands_global if n_bands_global else 0))
        
        for i, ex in enumerate(all_exemplars):
            if i % 10000 == 0:
                print(f"  Writing exemplar {i}/{n_exemplars}...", end='\r')
            
            f.write(struct.pack('B', ex['label']))
            ex['patch'].tofile(f)
    
    print(f"\n[STATUS] Export complete!")
    
    n_positive = sum(1 for ex in all_exemplars if ex['label'] == 1)
    n_negative = sum(1 for ex in all_exemplars if ex['label'] == 0)
    print(f"[STATUS] Class distribution: {n_positive} positive, {n_negative} negative")
    
    file_size_mb = (12 + n_exemplars * (1 + patch_dim * 4)) / (1024 * 1024)
    print(f"[STATUS] File size: {file_size_mb:.1f} MB")

if __name__ == "__main__":
    main()

