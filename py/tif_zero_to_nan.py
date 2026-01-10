'''20260109: open a tif file. Write back a new version with NAN instead of 0 vectors.
'''

#!/usr/bin/env python3

import sys
import os
import glob
import tempfile
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def process_tif(path):
    print(f"Processing: {path}")

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        print("  ERROR: Could not open file")
        return

    raster_count = ds.RasterCount
    needs_conversion = False

    for i in range(1, raster_count + 1):
        band = ds.GetRasterBand(i)
        if band.DataType != gdal.GDT_Float32:
            needs_conversion = True
            break

    if needs_conversion:
        print("  WARNING: Input is not Float32 — converting to Float32")

    # Create temporary output
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd)

    driver = gdal.GetDriverByName("GTiff")

    out_ds = driver.CreateCopy(
        tmp_path,
        ds,
        strict=0,
        options=["TILED=YES", "COMPRESS=LZW"]
    )

    ds = None  # close input

    if out_ds is None:
        print("  ERROR: Failed to create temporary copy")
        os.remove(tmp_path)
        return

    # Process bands (now guaranteed writable Float32)
    for band_idx in range(1, raster_count + 1):
        band = out_ds.GetRasterBand(band_idx)
        arr = band.ReadAsArray()

        if arr is None:
            print(f"  WARNING: Could not read band {band_idx}")
            continue

        if band.DataType != gdal.GDT_Float32:
            arr = arr.astype(np.float32)

        arr[arr == 0.0] = np.nan

        band.WriteArray(arr)
        band.SetNoDataValue(np.nan)
        band.FlushCache()

    out_ds.FlushCache()
    out_ds = None

    # Atomically replace original file
    os.replace(tmp_path, path)

    print("  Done.")


def main():
    if len(sys.argv) == 1:
        files = sorted(
            glob.glob("*.tif") + glob.glob("*.TIF") +
            glob.glob("*.tiff") + glob.glob("*.TIFF")
        )
        if not files:
            print("No TIF files found in current directory.")
            return
    else:
        files = [f for f in sys.argv[1:] if os.path.isfile(f)]
        if not files:
            print("No valid input TIF files found.")
            return

    for f in files:
        process_tif(f)


if __name__ == "__main__":
    main()

