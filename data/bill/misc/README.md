# Miscellaneous Utilities

Shared helper functions used across the repository. Organised into focused modules and re-exported from `__init__.py` so everything is available from a single `from misc import ...` statement.

<!-- IMAGE: insert misc utilities screenshot here -->

---

## File Overview

| File | Purpose |
|------|---------|
| `general.py` | Core array utilities: percentile contrast stretching, NaN filtering, polygon border extraction/drawing, boolean matrix checks, and combinatorics helpers. |
| `sen2.py` | Sentinel-2 specific helpers: timestamp extraction from filenames, band index/name lookup by Sentinel-2 band number, and ENVI file writing (new file or append bands). |
| `date_time.py` | Date conversion utilities: parse Sentinel-2 `YYYYMMDD` strings to Python `date` objects, and convert Julian day + year to `datetime` (used for VIIRS filenames). |
| `files.py` | File system helper: iterate over all files of a given extension in a directory. |
| `photos.py` | Convert `.bin` ENVI rasters in a directory to `.png` files using percentile normalization from the first file encountered. |

---

## Function Reference

### `general.py`

| Function | Description |
|----------|-------------|
| `htrim_1d(X, p=1.0)` | Percentile contrast stretch on a 1D array — clips the bottom/top `p`% and rescales to `[0, 1]`. |
| `htrim_3d(X, p=1.0)` | Same as `htrim_1d` but applied independently per channel on a 3D `(H, W, C)` array. |
| `ignore_nan_2D(X, axis=1)` | Returns a boolean mask and filtered matrix with all NaN-containing rows (`axis=1`) or columns (`axis=0`) removed. |
| `extract_border(mask, thickness=1)` | Uses binary erosion to extract the border pixels of a boolean polygon mask, leaving only a ring of the given pixel thickness. |
| `draw_border(img, border, color=(255,0,0))` | Paints border pixels onto a copy of an image in the specified colour (default red). |
| `is_boolean_matrix(A)` | Returns `True` if an array is of boolean dtype or contains only 0/1 values. |
| `get_combinations(val_lst, least=1, most=None)` | Returns all unordered combinations of a list within a given size range. |

### `sen2.py`

| Function | Description |
|----------|-------------|
| `read_timestamp_filename(filename)` | Extracts the acquisition timestamp from a Sentinel-2 filename (e.g. `S2A_MSIL2A_20250502T193831_...` → `"20250502T193831"`). |
| `band_index(band_info_list, band)` | Finds the array index (0-based) of a Sentinel-2 band number (e.g. `8` → NIR) by searching GDAL band descriptions with regex. |
| `band_name(band_info_list, band_index)` | Returns the short band label (e.g. `"B8"`) at a given 0-based index. |
| `writeENVI(output_filename, data, *, mode, ref_filename, ...)` | Writes a NumPy array to an ENVI `.bin` file. `mode="new"` creates a fresh file copying georeferencing from a reference; `mode="add"` appends new bands to an existing file. |

### `date_time.py`

| Function | Description |
|----------|-------------|
| `date_str2obj(date_str, format="%Y%m%d")` | Parses a date string (e.g. `"20250502"`) into a Python `date` object. |
| `julian_to_date(year, jday)` | Converts a Julian day number and year into a `datetime` object (used for VIIRS file naming). |

### `files.py`

| Function | Description |
|----------|-------------|
| `iter_files(folder, file_type, full_path=True)` | Generator that yields all files with the given extension (e.g. `".bin"`) in a directory. Returns full paths by default, filenames only if `full_path=False`. |

### `photos.py`

| Function | Description |
|----------|-------------|
| `save_png_same_dir(dir, band_list=None, p_trim=1)` | Converts all `.bin` files in `dir` to `.png` using percentile normalization derived from the first file. Defaults to bands `[1, 2, 3]`. |

---

## Quick Usage

```python
# Everything is available from the top-level package
from misc import (
    htrim_3d,
    extract_border,
    draw_border,
    ignore_nan_2D,
    is_boolean_matrix,
    get_combinations,
    read_timestamp_filename,
    band_index,
    band_name,
    writeENVI,
    date_str2obj,
    julian_to_date,
    iter_files,
)

# Or import from the specific module
from misc.general import htrim_3d, extract_border
from misc.sen2 import writeENVI, read_timestamp_filename
from misc.date_time import date_str2obj, julian_to_date
from misc.files import iter_files

# Contrast stretch a 3D raster (H, W, C)
stretched = htrim_3d(raster_data, p=1.0)

# Extract and draw a polygon border on an image
border = extract_border(polygon_mask, thickness=3)
annotated = draw_border(rgb_image, border, color=(255, 0, 0))

# Extract timestamp from a Sentinel-2 filename
ts = read_timestamp_filename("S2A_MSIL2A_20250502T193831_N0511_R142_T09UYU.bin")
# → "20250502T193831"

# Lookup the array index of Band 8 (NIR)
idx = band_index(raster.band_info_list, 8)

# Write a NumPy array to a new georeferenced ENVI file
writeENVI("output.bin", data, mode="new", ref_filename="reference.bin", band_names=["burned(bool)"])

# Append a new band to an existing ENVI file
writeENVI("output_with_extra.bin", new_band, mode="add", ref_filename="output.bin")

# Convert Sentinel-2 date string to Python date
d = date_str2obj("20250502")   # → datetime.date(2025, 5, 2)

# Convert a VIIRS Julian date
dt = julian_to_date(2025, 263)  # → datetime(2025, 9, 20)

# Iterate over all .bin files in a directory
for filepath in iter_files("/data/images", ".bin"):
    print(filepath)
```

### Julian date CLI

```bash
python3 -m misc.date_time 2025 263
# → 2025-263 → 2025-09-20 (Saturday)
# → Leap year: no
```

---

## Notes

- All functions are re-exported from `__init__.py` — prefer `from misc import ...` for brevity.
- `writeENVI` always writes `Float32`. The `same_hdr=True` flag copies the `.hdr` file verbatim from the reference, which is useful when the output needs identical ENVI metadata.
- `photos.py` in this package is a simpler sequential version of the one in `fire_mapping/` — it normalizes using percentile bounds from the first file only. For a parallel, two-pass version use `fire_mapping.photos` instead.
