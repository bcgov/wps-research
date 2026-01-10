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

    src_ds = gdal.Open(path, gdal.GA_ReadOnly)
    if src_ds is None:
        print("  ERROR: Could not open file")
        return

    band_count = src_ds.RasterCount

    needs_conversion = any(
        src_ds.GetRasterBand(i).DataType != gdal.GDT_Float32
        for i in range(1, band_count + 1)
    )

    if needs_conversion:
        print("  WARNING: Input is not Float32 — converting to Float32")

    # Temp file in SAME directory (avoids cross-device error)
    src_dir = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tif",
        dir=src_dir
    )
    os.close(fd)

    driver = gdal.GetDriverByName("GTiff")

    out_ds = driver.CreateCopy(
        tmp_path,
        src_ds,
        strict=0,
        options=["TILED=YES", "COMPRESS=LZW"]
    )

    src_ds = None  # close input

    if out_ds is None:
        print("  ERROR: Failed to create temporary copy")
        os.remove(tmp_path)
        return

    # Process bands
    for band_idx in range(1, band_count + 1):
        band = out_ds.GetRasterBand(band_idx)

        arr = band.ReadAsArray()
        if arr is None:
            print(f"  WARNING: Could not read band {band_idx}")
            continue

        if band.DataType != gdal.GDT_Float32:
            arr = arr.astype(np.float32)

        # Replace zeros with NaN
        arr[arr == 0.0] = np.nan

        band.WriteArray(arr)

        # IMPORTANT: this is the ONLY supported way to write GDAL nodata
        band.SetNoDataValue(np.nan)

        band.FlushCache()

    out_ds.FlushCache()
    out_ds = None

    # Atomic replace (same filesystem, so this works)
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


