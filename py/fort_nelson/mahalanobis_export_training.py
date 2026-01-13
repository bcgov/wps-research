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

def compute_rectangle_stats(img, rectangle, patch_size=PATCH_SIZE):
    """Compute mean and covariance for a single training rectangle"""
    pad = patch_size // 2
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w, _ = img.shape
    
    x0, y0, x1, y1 = rectangle
    
    # Filter small rectangles
    if (x1 - x0) < MIN_POLY_DIMENSION or (y1 - y0) < MIN_POLY_DIMENSION:
        return None, None
    
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(w, int(x1))
    y1 = min(h, int(y1))
    
    # Extract all patches from this rectangle
    patches = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            patch = padded[y:y+patch_size, x:x+patch_size, :].reshape(-1)
            # Skip patches with NAN
            if not np.isnan(patch).any():
                patches.append(patch)
    
    if len(patches) == 0:
        return None, None
    
    patches = np.vstack(patches)
    
    # Compute mean and covariance
    mean = patches.mean(axis=0)
    cov = np.cov(patches, rowvar=False) + np.eye(patches.shape[1]) * 1e-5
    
    # Compute inverse covariance
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    
    return mean, inv_cov

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 export_mahalanobis_training.py annotations.shp output.bin")
        sys.exit(1)
    
    shp_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"[STATUS] Reading shapefile: {shp_file}")
    training = read_training_from_shapefile(shp_file)
    
    shp_dir = os.path.dirname(os.path.abspath(shp_file))
    cwd = os.getcwd()
    
    exemplars = []
    
    print(f"[STATUS] Processing training rectangles...")
    total_rectangles = sum(len(d['rectangles']) for d in training.values())
    processed = 0
    
    for src, data in training.items():
        img_path = locate_image_by_basename(src, [shp_dir, cwd])
        if img_path is None:
            print(f"[WARNING] Could not find image: {src}")
            continue
        
        print(f"[STATUS] Loading image: {src}")
        img, _ = load_image_stack(img_path)
        
        for rect, label in zip(data['rectangles'], data['labels']):
            mean, inv_cov = compute_rectangle_stats(img, rect)
            
            if mean is not None and inv_cov is not None:
                exemplars.append({
                    'label': label,
                    'mean': mean.astype(np.float32),
                    'cov': None,  # Not needed for classification
                    'inv_cov': inv_cov.astype(np.float32)
                })
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed {processed}/{total_rectangles} rectangles")
    
    n_exemplars = len(exemplars)
    patch_dim = exemplars[0]['mean'].shape[0] if n_exemplars > 0 else PATCH_SIZE * PATCH_SIZE * 3
    
    print(f"\n[STATUS] Total exemplars: {n_exemplars}")
    print(f"[STATUS] Patch dimension: {patch_dim}")
    
    # Write binary file
    print(f"[STATUS] Writing to {output_file}...")
    with open(output_file, "wb") as f:
        # Header
        f.write(struct.pack('i', n_exemplars))
        f.write(struct.pack('i', patch_dim))
        
        # Exemplars
        for ex in exemplars:
            f.write(struct.pack('B', ex['label']))
            ex['mean'].tofile(f)
            
            # Write dummy covariance (not used, but keeps format consistent)
            dummy_cov = np.zeros((patch_dim, patch_dim), dtype=np.float32)
            dummy_cov.tofile(f)
            
            ex['inv_cov'].tofile(f)
    
    print(f"[STATUS] Export complete!")
    
    # Report class distribution
    n_positive = sum(1 for ex in exemplars if ex['label'] == 1)
    n_negative = sum(1 for ex in exemplars if ex['label'] == 0)
    print(f"[STATUS] Class distribution: {n_positive} positive, {n_negative} negative")

if __name__ == "__main__":
    main()


