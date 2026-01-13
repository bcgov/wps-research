'''20260112: this file exports the pkl file used in classify_nn.py to the bin format expected by 
mahalanobis_classify.py 
'''

#!/usr/bin/env python3
"""
Compute Gaussian statistics (mean, covariance) for each training rectangle
and export in binary format for Mahalanobis CUDA classifier

Usage: python3 export_mahalanobis_training.py annotations.shp training_mahalanobis.bin
"""

import sys, os
import numpy as np
from osgeo import gdal, ogr
import struct
from multiprocessing import Pool, Manager

gdal.UseExceptions()

MIN_POLY_DIMENSION = 15
PATCH_SIZE = 7

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
    data = np.stack(
        [ds.GetRasterBand(i+1).ReadAsArray().astype(np.float32)
         for i in range(ds.RasterCount)],
        axis=-1
    )
    if data.shape[2] == 4:
        data = data[:, :, :3]
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

# Global variables for sharing with worker threads
_padded_img_global = None
_file_counter = None
_file_total = None
_global_counter = None
_global_total = None
_print_lock = None

def _init_worker(padded_img, file_counter, file_total, global_counter, global_total, print_lock):
    """Initialize worker with shared padded image data and counters"""
    global _padded_img_global, _file_counter, _file_total, _global_counter, _global_total, _print_lock
    _padded_img_global = padded_img
    _file_counter = file_counter
    _file_total = file_total
    _global_counter = global_counter
    _global_total = global_total
    _print_lock = print_lock

def _compute_rectangle_stats(args):
    """Worker function to compute stats for one rectangle"""
    rectangle, label, patch_size, h, w = args
    
    padded = _padded_img_global
    
    x0, y0, x1, y1 = rectangle
    
    # Filter small rectangles
    if (x1 - x0) < MIN_POLY_DIMENSION or (y1 - y0) < MIN_POLY_DIMENSION:
        with _file_counter.get_lock():
            _file_counter.value += 1
            file_done = _file_counter.value
        with _global_counter.get_lock():
            _global_counter.value += 1
            global_done = _global_counter.value
        
        with _print_lock:
            print(f"  [RECT] File: {file_done}/{_file_total.value} | Global: {global_done}/{_global_total.value} (too small)")
        return None
    
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))
    
    # Extract all patches from this rectangle (sliding window)
    patches = []
    pad = patch_size // 2
    
    for y in range(y0, y1):
        for x in range(x0, x1):
            patch = padded[y:y+patch_size, x:x+patch_size, :].reshape(-1)
            # Skip patches with NAN
            if not np.isnan(patch).any():
                patches.append(patch)
    
    if len(patches) == 0:
        with _file_counter.get_lock():
            _file_counter.value += 1
            file_done = _file_counter.value
        with _global_counter.get_lock():
            _global_counter.value += 1
            global_done = _global_counter.value
        
        with _print_lock:
            print(f"  [RECT] File: {file_done}/{_file_total.value} | Global: {global_done}/{_global_total.value} (no valid patches)")
        return None
    
    patches = np.vstack(patches)
    
    # Compute mean and covariance
    mean = patches.mean(axis=0)
    cov = np.cov(patches, rowvar=False) + np.eye(patches.shape[1]) * 1e-5
    
    # Compute inverse covariance
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    
    # Update counters and print progress
    with _file_counter.get_lock():
        _file_counter.value += 1
        file_done = _file_counter.value
    
    with _global_counter.get_lock():
        _global_counter.value += 1
        global_done = _global_counter.value
    
    with _print_lock:
        print(f"  [RECT] File: {file_done}/{_file_total.value} | Global: {global_done}/{_global_total.value}")
    
    return {
        'label': label,
        'mean': mean.astype(np.float32),
        'cov': None,
        'inv_cov': inv_cov.astype(np.float32)
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 export_mahalanobis_training.py annotations.shp output.bin")
        sys.exit(1)
    
    shp_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"[STATUS] Reading shapefile: {shp_file}")
    training_by_image = read_training_from_shapefile(shp_file)
    
    shp_dir = os.path.dirname(os.path.abspath(shp_file))
    cwd = os.getcwd()
    
    # Calculate totals
    image_files = list(training_by_image.keys())
    total_images = len(image_files)
    total_rectangles = sum(len(data['rectangles']) for data in training_by_image.values())
    
    print(f"[STATUS] Found {total_images} image files with {total_rectangles} total rectangles")
    
    all_exemplars = []
    n_processes = 128
    
    print(f"[STATUS] Processing with {n_processes} parallel workers\n")
    
    # Create manager for shared counters
    manager = Manager()
    global_counter = manager.Value('i', 0)
    global_total = manager.Value('i', total_rectangles)
    print_lock = manager.Lock()
    
    # Outer loop: process each image file
    for img_idx, image_file in enumerate(image_files, 1):
        data = training_by_image[image_file]
        n_rectangles = len(data['rectangles'])
        
        print(f"[IMAGE {img_idx}/{total_images}] Processing: {image_file}")
        print(f"[IMAGE {img_idx}/{total_images}] Rectangles in this file: {n_rectangles}")
        
        # Locate and load image
        img_path = locate_image_by_basename(image_file, [shp_dir, cwd])
        if img_path is None:
            print(f"[WARNING] Could not find image: {image_file}")
            with global_counter.get_lock():
                global_counter.value += n_rectangles
            continue
        
        print(f"[IMAGE {img_idx}/{total_images}] Loading image...")
        img, _ = load_image_stack(img_path)
        h, w, _ = img.shape
        
        # Pad image ONCE before parallel processing
        print(f"[IMAGE {img_idx}/{total_images}] Padding image...")
        pad = PATCH_SIZE // 2
        padded_img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        
        # Create file-specific counter
        file_counter = manager.Value('i', 0)
        file_total = manager.Value('i', n_rectangles)
        
        # Prepare arguments for parallel processing
        args_list = [
            (rect, label, PATCH_SIZE, h, w)
            for rect, label in zip(data['rectangles'], data['labels'])
        ]
        
        print(f"[IMAGE {img_idx}/{total_images}] Processing rectangles with {n_processes} workers...\n")
        
        # Process all rectangles for this image in parallel
        with Pool(n_processes, initializer=_init_worker, 
                 initargs=(padded_img, file_counter, file_total, global_counter, global_total, print_lock)) as pool:
            results = pool.map(_compute_rectangle_stats, args_list)
        
        # Collect valid exemplars
        valid_exemplars = [r for r in results if r is not None]
        all_exemplars.extend(valid_exemplars)
        
        print(f"\n[IMAGE {img_idx}/{total_images}] Completed: {len(valid_exemplars)} valid exemplars from {n_rectangles} rectangles")
        print(f"[TOTAL] Global progress: {global_counter.value}/{total_rectangles} rectangles processed\n")
    
    n_exemplars = len(all_exemplars)
    patch_dim = all_exemplars[0]['mean'].shape[0] if n_exemplars > 0 else PATCH_SIZE * PATCH_SIZE * 3
    
    print(f"\n[STATUS] Total exemplars generated: {n_exemplars}")
    print(f"[STATUS] Patch dimension: {patch_dim}")
    
    # Write binary file
    print(f"[STATUS] Writing to {output_file}...")
    with open(output_file, "wb") as f:
        # Header
        f.write(struct.pack('i', n_exemplars))
        f.write(struct.pack('i', patch_dim))
        
        # Exemplars
        for ex in all_exemplars:
            f.write(struct.pack('B', ex['label']))
            ex['mean'].tofile(f)
            
            # Write dummy covariance (not used, but keeps format consistent)
            dummy_cov = np.zeros((patch_dim, patch_dim), dtype=np.float32)
            dummy_cov.tofile(f)
            
            ex['inv_cov'].tofile(f)
    
    print(f"[STATUS] Export complete!")
    
    # Report class distribution
    n_positive = sum(1 for ex in all_exemplars if ex['label'] == 1)
    n_negative = sum(1 for ex in all_exemplars if ex['label'] == 0)
    print(f"[STATUS] Class distribution: {n_positive} positive, {n_negative} negative")

if __name__ == "__main__":
    main()

