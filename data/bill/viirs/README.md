# viirs — VIIRS Fire Pixel Processing Toolkit

*Last updated: March 10, 2026*

An end-to-end pipeline for downloading, converting, and visualising VIIRS fire pixel data. Designed specifically for **VNP14IMG** products.

---

## Overview

This toolkit provides a complete workflow for working with VIIRS active fire data: pulling raw NetCDF files from NASA, converting them to shapefiles in an appropriate Sentinel-2 projection, and optionally rasterizing or accumulating the results for burn mapping. A graphical interface is included for interactive exploration and visualisation.

---

## Workflow

### 1. Launch the GUI

```bash
python -m viirs.fp_gui
```

**Load Raster image:** The reference raster is needed for everything. To improve pan and zoom performance, navigate to the **Config** tab and reduce the **Max Raster Display** value (default is full resolution).

### 2. Download VIIRS VNP14IMG Data

> This step can be skipped if data was already downloaded in a previous session.

A reference raster file (`.bin` format; additional extensions may be supported in future releases) is required for:

- Extracting the spatial projection (e.g. EPSG:32609)
- Defining the bounding box of the raster image

**Steps:**

1. A LAADS DAAC authentication token is required. Store it at `/data/.tokens/laads` for automatic loading (displayed as `***...`), or paste it in the key box.
3. Select the date range of interest (`YYYY-MM-DD` format).
4. The save directory will be in `VNP14IMG` in the same folder as the raster image. Removal of downloaded data is recommended for each download session (bug is not fixed) — writing to an existing directory may cause errors.
5. Click **Download**. A summary is displayed in the GUI panel; full output is printed to the originating terminal.

**What the download does:**

- Retrieves VNP14IMG data from the NASA LAADS DAAC using the provided authentication token.
- Fire pixel data are sourced in EPSG:4326 (geographic coordinates).
- During the *shapify* process, `.nc` files are extracted and converted to `.shp` files, retaining only fire pixels within the specified bounding box. Coordinates are automatically reprojected from EPSG:4326 to match the reference raster's projection, ensuring correct spatial alignment in the GUI.

### 3. Load Data

**Shapefiles:** Select a directory (if you just download, the dir will be auto-pasted) — the engine will discover all shapefiles within it and load them ordered by detection datetime (extracted from the third field of each filename, in UTC). Some processing is performed on load to optimise rendering; progress is shown in the bottom-left panel.

> **Note:** In this version, you cannot load shapefiles before the raster image. It needs an image to project onto.

### 4. Explore the GUI

Once raster and fire pixel data are loaded, the GUI is ready for interactive use.

In the **Config** tab:

- **Fire pixel size** — adjust the display size of fire pixels (integer, minimum 1).
- **Colour levels** — configure up to 500 colour levels. Pixel colour shifts along the colour bar to reflect the age of detection (in days), with older detections displayed in progressively distinct colours.

### 5. Accumulate & Rasterize

The **Accumulate & Rasterize** button (on the navigation/tools row) runs the full accumulation and rasterization pipeline directly from the GUI.

**Setup:**

1. **Reference raster** — set the raster image to use as the grid template for rasterized outputs. This field automatically mirrors the visualization raster loaded in step 3; it can be changed independently if a different base grid is needed.
2. **Output directory** — browse or type the directory where all output shapefiles and rasterized `.bin`/`.hdr` files will be saved (flat, no subdirectories).
3. **Date range** — the start and end dates from the GUI date fields are used. Apply a date filter first if the range needs to be narrowed.

**Running:**

1. Click **Accumulate & Rasterize**. A confirmation popup summarises the shapefile directory, reference raster, date range, and output directory.
2. Click the green **Confirm** button to start (or close the popup to cancel).
3. Progress is reported in the bottom-left status panel.

**What it does:**

- **Accumulate:** scans the loaded shapefile directory, filters by the GUI date range, sorts all files chronologically by detection datetime, and writes one cumulative shapefile per unique detection date into the output directory. Output filenames use the actual detection timestamps from the source files: `VIIRS_VNP14IMG_{first_dt}_{batch_end_dt}.shp`. The first output file has identical start and end timestamps. Each cumulative file includes an `age_days` column (fractional days relative to that batch's end datetime).
- **Rasterize:** after accumulation completes, each accumulated shapefile is rasterized in parallel onto the base raster grid, producing a binary fire mask (`.bin`/`.hdr`) for each file in the same output directory.

**Output example** (for data spanning 2025-09-20 to 2025-09-25):

```
output_dir/
    VIIRS_VNP14IMG_20250920T0840_20250920T0840.shp   (+ .dbf .prj .shx)
    VIIRS_VNP14IMG_20250920T0840_20250920T0840.bin    (+ .hdr)
    VIIRS_VNP14IMG_20250920T0840_20250921T1430.shp
    VIIRS_VNP14IMG_20250920T0840_20250921T1430.bin
    ...
    VIIRS_VNP14IMG_20250920T0840_20250925T2010.shp
    VIIRS_VNP14IMG_20250920T0840_20250925T2010.bin
```

---

## Additional Utilities

### Rasterize Shapefiles for Burn Mapping

Generate binary fire masks on the Sentinel-2 grid (e.g. as inputs for burn severity mapping):

```bash
# Process all shapefiles in a directory
python -m viirs.utils.rasterize /data/viirs/shapefiles sentinel2.bin /output/rasters --buffer 375 -w 8

# Process a single shapefile
python -m viirs.utils.rasterize fire.shp sentinel2.bin /output/rasters
```

### Accumulate into a Single Shapefile

Merge all fire pixels from a date range into one shapefile, including an `age_days` column (fractional days, e.g. `34.06`):

```bash
python -m viirs.utils.accumulate /data/viirs/shapefiles 20250401 20250930 -r <reference_file>
```

Pixel age is computed as `(end_date − detection_datetime)` in fractional days.