

import numpy as np
import os
import sys
import pickle
from joblib import Parallel, delayed
from osgeo import gdal

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def classify_chunk(start_y, end_y, padded, w, patch_size, mean_covs, inv_cov):
    chunk_height = end_y - start_y
    chunk_result = np.zeros((chunk_height, w), dtype=np.uint8)
    for y_idx, y in enumerate(range(start_y, end_y)):
        for x in range(w):
            patch = padded[y:y+patch_size, x:x+patch_size, :]
            patch_flat = patch.flatten()
            scores = {}
            for label in [0, 1]:
                mean, _ = mean_covs[label]
                if mean is None or inv_cov[label] is None:
                    scores[label] = np.inf
                else:
                    diff = patch_flat - mean
                    scores[label] = diff @ inv_cov[label] @ diff.T
            chunk_result[y_idx, x] = 1 if scores[1] < scores[0] else 0
    return (start_y, chunk_result)

def classify_by_gaussian_parallel(image, mean_covs, patch_size=7, chunk_rows=16):
    h, w, b = image.shape
    pad = patch_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    output = np.zeros((h, w), dtype=np.uint8)

    inv_cov = {}
    for label in [0, 1]:
        mean, cov = mean_covs[label]
        if mean is None or cov is None:
            inv_cov[label] = None
        else:
            try:
                inv_cov[label] = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv_cov[label] = np.linalg.pinv(cov)

    chunks = [(y, min(y + chunk_rows, h)) for y in range(0, h, chunk_rows)]
    print(f"Starting classification using all CPU cores on {len(chunks)} chunks...")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(classify_chunk)(start_y, end_y, padded, w, patch_size, mean_covs, inv_cov)
        for (start_y, end_y) in chunks)

    for idx, (start_y, chunk_result) in enumerate(results, 1):
        output[start_y:start_y + chunk_result.shape[0], :] = chunk_result
        print_progress_bar(idx, len(chunks), prefix='Classification Progress:', suffix='Complete', length=50)
    print("Classification completed.")
    return output

def save_envi_image(filename, image_array, prototype_ds):
    """
    Save numpy array as ENVI format file, float32, single band or multiband
    """
    driver = gdal.GetDriverByName('ENVI')
    h, w = image_array.shape[:2]
    bands = 1 if image_array.ndim == 2 else image_array.shape[2]

    out_ds = driver.Create(filename, w, h, bands, gdal.GDT_Float32)
    if out_ds is None:
        print("Failed to create output ENVI file.")
        return False

    # Copy geo transform and projection from prototype
    out_ds.SetGeoTransform(prototype_ds.GetGeoTransform())
    out_ds.SetProjection(prototype_ds.GetProjection())

    if bands == 1:
        out_ds.GetRasterBand(1).WriteArray(image_array.astype(np.float32))
    else:
        for i in range(bands):
            out_ds.GetRasterBand(i + 1).WriteArray(image_array[:, :, i].astype(np.float32))

    out_ds.FlushCache()
    out_ds = None
    return True

def load_envi_image(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"Failed to open ENVI image: {path}")
        sys.exit(1)
    band_count = ds.RasterCount
    width = ds.RasterXSize
    height = ds.RasterYSize

    data = np.stack([
        ds.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
        for i in range(band_count)
    ], axis=-1)

    return ds, data

def main():
    if len(sys.argv) != 3:
        print("Usage: python classify_large_envi.py model.pkl image.bin")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    print("Loading model...")
    with open(model_path, 'rb') as f:
        mean_covs = pickle.load(f)
    print("Model loaded.")

    print("Loading image...")
    ds, image = load_envi_image(image_path)
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels with {image.shape[2]} bands.")

    classified = classify_by_gaussian_parallel(image, mean_covs, patch_size=7, chunk_rows=16)

    # Save output ENVI classification file
    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_classification.bin"
    print(f"Saving classification result to {out_path} ...")
    if save_envi_image(out_path, classified.astype(np.float32), ds):
        print("Classification saved successfully.")
    else:
        print("Failed to save classification.")

if __name__ == "__main__":
    main()

