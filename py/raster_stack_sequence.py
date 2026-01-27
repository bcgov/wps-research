'''20260127 raster stack sequence: '''
#!/usr/bin/env python3

import os
import re
import sys
import subprocess

from osgeo import gdal


def main():
    # Find all .bin files
    bin_files = [f for f in os.listdir(".") if f.endswith(".bin")]

    if not bin_files:
        print("ERROR: No .bin files found")
        sys.exit(1)

    # Regex: leading digits followed by anything, ending in .bin
    pattern = re.compile(r"^(\d+).*\.bin$")

    files_with_prefix = []
    prefix_lengths = set()

    for f in bin_files:
        m = pattern.match(f)
        if not m:
            continue
        prefix = m.group(1)
        prefix_lengths.add(len(prefix))
        files_with_prefix.append((int(prefix), f))

    if not files_with_prefix:
        print("ERROR: No .bin files with numeric prefix found")
        sys.exit(1)

    if len(prefix_lengths) != 1:
        print("ERROR: Files have different digit-length numeric prefixes:")
        print(sorted(prefix_lengths))
        sys.exit(1)

    # Sort by numeric prefix
    files_with_prefix.sort(key=lambda x: x[0])
    file_names = [f for _, f in files_with_prefix]

    print("Stacking files in order:")
    for f in file_names:
        print(" ", f)

    # Build and run stacking command
    cmd = " ".join(["raster_stack.py"] + file_names + ["stack.bin"])
    print("\nRunning command:")
    print(cmd)

    subprocess.run(cmd, shell=True, check=True)

    # ------------------------------------------------------------
    # Update output band names
    # ------------------------------------------------------------
    print("\nUpdating output band names...")

    prefixed_band_names = []

    # Read band names from each input file
    for fname in file_names:
        ds = gdal.Open(fname, gdal.GA_ReadOnly)
        if ds is None:
            print(f"ERROR: Cannot open {fname}")
            sys.exit(1)

        for b in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(b)
            band_name = band.GetDescription()
            if not band_name:
                band_name = f"Band {b}"
            prefixed_band_names.append(f"{fname}:{band_name}")

        ds = None

    # Open output dataset for update
    out_ds = gdal.Open("stack.bin", gdal.GA_Update)
    if out_ds is None:
        print("ERROR: Cannot open stack.bin for update")
        sys.exit(1)

    if out_ds.RasterCount != len(prefixed_band_names):
        print("ERROR: Band count mismatch between inputs and output")
        sys.exit(1)

    # Write updated band names
    for i, name in enumerate(prefixed_band_names, start=1):
        band = out_ds.GetRasterBand(i)
        band.SetDescription(name)

    out_ds = None
    print("Band names updated successfully.")


if __name__ == "__main__":
    main()

