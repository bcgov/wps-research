#!/usr/bin/env python3
"""
Concatenate ENVI-format .bin files into a near-square spatial grid for ML training.

Finds all S2*.bin files in the current directory, sorts by acquisition datetime
(third underscore-delimited field in the filename), computes a near-square grid
layout, discards the fewest trailing images necessary, then writes a single
ENVI output band-by-band.

Output is deliberately non-spatial (no map info / CRS).
"""

import glob
import math
import os
import re
import sys
import numpy as np

try:
    from osgeo import gdal
except ImportError:
    import gdal

gdal.UseExceptions()


# ---------------------------------------------------------------------------
# 1. Discover and sort files
# ---------------------------------------------------------------------------
bin_files = sorted(glob.glob("S2*.bin"))

if not bin_files:
    sys.exit("ERROR: No S2*.bin files found in the current directory.")

print(f"Found {len(bin_files)} .bin files")


def sort_key(fname):
    """Extract the datetime field (3rd underscore-delimited token) for sorting."""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    # e.g. S2B_MSIL2A_20250909T191909_N0511_R099_T09UYU_...
    if len(parts) >= 3:
        return parts[2]
    return base  # fallback


bin_files.sort(key=sort_key)

# ---------------------------------------------------------------------------
# 2. Compute grid dimensions
# ---------------------------------------------------------------------------
n = len(bin_files)
grid_rows = int(math.floor(math.sqrt(n)))

# Add columns until we pack as many images as possible
grid_cols = grid_rows
while (grid_cols + 1) * grid_rows <= n:
    grid_cols += 1
# Also check if adding a row instead would use more images
alt_cols = grid_rows
alt_rows = grid_rows + 1
while (alt_cols + 1) * alt_rows <= n:
    alt_cols += 1
if alt_rows * alt_cols > grid_rows * grid_cols:
    grid_rows, grid_cols = alt_rows, alt_cols

used = grid_rows * grid_cols
discarded = n - used

print(f"Grid layout: {grid_rows} rows x {grid_cols} cols = {used} images "
      f"({discarded} discarded from end of sorted list)")

# Keep only the images we need (discard from the tail of the sorted list)
bin_files = bin_files[:used]

# ---------------------------------------------------------------------------
# 3. Read reference image metadata
# ---------------------------------------------------------------------------
ref_ds = gdal.Open(bin_files[0], gdal.GA_ReadOnly)
if ref_ds is None:
    sys.exit(f"ERROR: Cannot open {bin_files[0]}")

nrow = ref_ds.RasterYSize
ncol = ref_ds.RasterXSize
nbands = ref_ds.RasterCount
dtype_gdal = ref_ds.GetRasterBand(1).DataType
dtype_np = gdal.GetDataTypeName(dtype_gdal)  # for reporting

# Collect band names from the reference image
band_names = []
for b in range(1, nbands + 1):
    band = ref_ds.GetRasterBand(b)
    desc = band.GetDescription()
    band_names.append(desc if desc else f"Band {b}")

ref_ds = None  # close

print(f"Image dimensions: {ncol} cols x {nrow} rows x {nbands} bands  "
      f"(GDAL type: {dtype_np})")
print(f"Output dimensions: {ncol * grid_cols} cols x {nrow * grid_rows} rows "
      f"x {nbands} bands")

# ---------------------------------------------------------------------------
# 4. Create the output ENVI file
# ---------------------------------------------------------------------------
out_name = "grid_concat.bin"
out_cols = ncol * grid_cols
out_rows = nrow * grid_rows

driver = gdal.GetDriverByName("ENVI")
out_ds = driver.Create(out_name, out_cols, out_rows, nbands, dtype_gdal)
if out_ds is None:
    sys.exit("ERROR: Could not create output dataset.")

# ---------------------------------------------------------------------------
# 5. Fill the output band-by-band
# ---------------------------------------------------------------------------
for b_idx in range(1, nbands + 1):
    print(f"  Processing band {b_idx}/{nbands} ({band_names[b_idx-1]}) ...")
    out_band = out_ds.GetRasterBand(b_idx)

    for img_i, fpath in enumerate(bin_files):
        gr = img_i // grid_cols   # row in the grid
        gc = img_i % grid_cols    # col in the grid

        ds = gdal.Open(fpath, gdal.GA_ReadOnly)
        if ds is None:
            sys.exit(f"ERROR: Cannot open {fpath}")
        data = ds.GetRasterBand(b_idx).ReadAsArray()
        ds = None

        y_off = gr * nrow
        x_off = gc * ncol
        out_band.WriteArray(data, x_off, y_off)

    out_band.FlushCache()

# ---------------------------------------------------------------------------
# 6. Write band names, then close (triggers .hdr write)
# ---------------------------------------------------------------------------
for b_idx in range(1, nbands + 1):
    out_ds.GetRasterBand(b_idx).SetDescription(band_names[b_idx - 1])

out_ds.FlushCache()
out_ds = None

# ---------------------------------------------------------------------------
# 7. Post-process the .hdr: strip map info / coordinate system, ensure
#    band names are present
# ---------------------------------------------------------------------------
hdr_path = out_name.replace(".bin", ".hdr")

if os.path.exists(hdr_path):
    with open(hdr_path, "r") as f:
        hdr_text = f.read()

    # Remove map info and coordinate system info lines (may be multi-line with {})
    hdr_text = re.sub(
        r"(?m)^map info\s*=\s*\{[^}]*\}\s*\n?", "", hdr_text
    )
    hdr_text = re.sub(
        r"(?m)^coordinate system string\s*=\s*\{[^}]*\}\s*\n?", "", hdr_text
    )
    # Also remove simple single-line variants just in case
    hdr_text = re.sub(r"(?m)^map info\s*=.*\n?", "", hdr_text)
    hdr_text = re.sub(r"(?m)^coordinate system string\s*=.*\n?", "", hdr_text)

    # If GDAL didn't write band names, inject them
    if "band names" not in hdr_text.lower():
        names_block = "band names = {\n"
        for i, bn in enumerate(band_names):
            sep = "," if i < len(band_names) - 1 else ""
            names_block += f"  {bn}{sep}\n"
        names_block += "}\n"
        # Insert before closing (or just append)
        hdr_text = hdr_text.rstrip("\n") + "\n" + names_block

    with open(hdr_path, "w") as f:
        f.write(hdr_text)

    print(f"\nCleaned .hdr: removed map info / CRS, ensured band names present.")

print(f"\nDone.  Output: {out_name}  ({out_cols} x {out_rows} x {nbands})")
print(f"Grid: {grid_rows} rows x {grid_cols} cols,  {used}/{n} images used.")

