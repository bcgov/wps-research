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
- Input image is stack.bin ENVI file in present directory
- Training rectangles in annotation_labels.shp in present directory
- All rectangles use the same stack.bin image (SRC_IMAGE field is ignored)
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
WINDOW_SIZE = 15
TARGET_N_EXEMPLARS = 50  # Target number of exemplars per rectangle
N_THREADS = 16
ANNOTATION_SHP = "annotation_labels.shp"
STACK_IMAGE = "stack.bin"
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
    """Read shapefile and extract all rectangles (ignoring SRC_IMAGE)"""
    ds = ogr.Open(shp, 0)
    if ds is None:
        print(f"[ERROR] Could not open shapefile: {shp}")
        sys.exit(1)
    lyr = ds.GetLayer(0)
    
    rectangles = []
    labels = []
    
    for f in lyr:
        cls = f.GetField("CLASS")
        rect = parse_coords_img_to_rect(f.GetField("COORDS_IMG"))
        if rect is None:
            continue
        lbl = 1 if cls.strip().upper() == "POSITIVE" else 0
        
        rectangles.append(rect)
        labels.append(lbl)
    
    return rectangles, labels

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
    
    # Check for required files
    if not os.path.exists(ANNOTATION_SHP):
        print(f"[ERROR] Required file not found: {ANNOTATION_SHP}")
        sys.exit(1)
    
    if not os.path.exists(STACK_IMAGE):
        print(f"[ERROR] Required file not found: {STACK_IMAGE}")
        sys.exit(1)
    
    print(f"[STATUS] WINDOW_SIZE = {WINDOW_SIZE}")
    print(f"[STATUS] TARGET_N_EXEMPLARS = {TARGET_N_EXEMPLARS} per rectangle")
    print(f"[STATUS] Reading shapefile: {ANNOTATION_SHP}")
    
    rectangles, labels = read_training_from_shapefile(ANNOTATION_SHP)
    total_rectangles = len(rectangles)
    
    print(f"[STATUS] Found {total_rectangles} total rectangles")
    print(f"[STATUS] Processing with {N_THREADS} parallel workers")
    print(f"[STATUS] Each window becomes an individual exemplar\n")
    
    # Load the single stack image
    print(f"[STATUS] Loading image: {STACK_IMAGE}")
    img, _, n_bands = load_image_stack(STACK_IMAGE)
    h, w, c = img.shape
    print(f"[STATUS] Image size: {w} x {h} x {c} bands")
    
    print(f"[STATUS] Extracting windows from {total_rectangles} rectangles...")
    
    inputs = [
        (img, rect, label, h, w, c)
        for rect, label in zip(rectangles, labels)
    ]
    
    results = parfor(lambda x: extract_windows_from_rectangle(*x), inputs, N_THREADS)
    
    all_exemplars = []
    n_skipped = 0
    for rect_exemplars in results:
        if len(rect_exemplars) == 0:
            n_skipped += 1
        all_exemplars.extend(rect_exemplars)
    
    n_exemplars = len(all_exemplars)
    
    print(f"[STATUS] Extracted {n_exemplars} exemplars from {total_rectangles - n_skipped} rectangles")
    print(f"[STATUS] Skipped {n_skipped} rectangles (too small for window size {WINDOW_SIZE}x{WINDOW_SIZE})")
    
    if n_exemplars == 0:
        print(f"\n[ERROR] No exemplars were generated!")
        print(f"[ERROR] Check that rectangles in {ANNOTATION_SHP} are large enough for {WINDOW_SIZE}x{WINDOW_SIZE} windows")
        sys.exit(1)
    
    patch_dim = all_exemplars[0]['patch'].shape[0]
    
    print(f"\n[STATUS] Total exemplars generated: {n_exemplars}")
    print(f"[STATUS] Patch dimension: {patch_dim}")
    print(f"[STATUS] Number of bands: {c}")
    
    # Write binary file (includes n_bands for classifier to use)
    print(f"[STATUS] Writing to {output_file}...")
    
    with open(output_file, "wb") as f:
        f.write(struct.pack('i', n_exemplars))
        f.write(struct.pack('i', patch_dim))
        f.write(struct.pack('i', c))
        
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


