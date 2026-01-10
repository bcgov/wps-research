'''20260109: open a tif file. Write back a new version with NAN instead of 0 vectors.
'''
#!/usr/bin/env python3
import sys
import os
import glob
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def process_tif(path):
    print(f"Processing: {path}")

    ds = gdal.Open(path, gdal.GA_Update)
    if ds is None:
        print(f"  ERROR: Could not open file")
        return

    # Verify all bands are Float32
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        if band.DataType != gdal.GDT_Float32:
            dtype_name = gdal.GetDataTypeName(band.DataType)
            print(
                f"  WARNING: Skipping file — band {band_idx} is {dtype_name}, "
                f"not Float32"
            )
            ds = None
            return

    # All bands verified as Float32
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        arr = band.ReadAsArray()

        if arr is None:
            print(f"  WARNING: Could not read band {band_idx}")
            continue

        # Replace zeros with NaN
        arr[arr == 0.0] = np.nan

        band.WriteArray(arr)
        band.SetNoDataValue(np.nan)
        band.FlushCache()

    ds.FlushCache()
    ds = None
    print("  Done.")


def main():
    if len(sys.argv) == 1:
        # No arguments: process all TIF files in current directory
        files = sorted(
            glob.glob("*.tif") + glob.glob("*.TIF") +
            glob.glob("*.tiff") + glob.glob("*.TIFF")
        )
        if not files:
            print("No TIF files found in current directory.")
            return
    else:
        # Process only files explicitly provided
        files = [f for f in sys.argv[1:] if os.path.isfile(f)]
        if not files:
            print("No valid input TIF files found.")
            return

    for f in files:
        process_tif(f)


if __name__ == "__main__":
    main()

