# viirs — VIIRS Fire Pixel Processing Toolkit

*Last updated: March 9, 2026*

An end-to-end pipeline for downloading, converting, and visualising VIIRS fire pixel data. Designed specifically for **VNP14IMG** products.

---

## Overview

This toolkit provides a complete workflow for working with VIIRS active fire data: pulling raw NetCDF files from NASA, converting them to shapefiles in a Sentinel-2 UTM projection, and optionally rasterizing or accumulating the results for burn mapping. A graphical interface is included for interactive exploration and visualisation.

---

## Workflow

### 1. Launch the GUI

```bash
python -m viirs.fp_gui
```

### 2. Download VIIRS VNP14IMG Data

> This step can be skipped if data was already downloaded in a previous session.

A reference raster file (`.bin` format; additional extensions may be supported in future releases) is required for:

- Extracting the spatial projection (e.g. EPSG:32609)
- Defining the bounding box of the raster image

**Steps:**

1. Browse to or paste in the raster file path, then click **Load Reference**. The detected projection and bounding box will be displayed. These values can be edited manually if needed (e.g. using QGIS to verify correct extents).
2. A LAADS DAAC authentication token is required. Store it at `/data/.tokens/laads` for automatic loading (displayed as `***...`), or paste it in directly.
3. Select the date range of interest (`YYYY-MM-DD` format).
4. Specify a save directory (e.g. `/data/users/viirs_T09UYU`). A new directory is recommended for each download session — writing to an existing directory may cause errors.
5. Click **Download**. A summary is displayed in the GUI panel; full output is printed to the originating terminal.

**What the download does:**

- Retrieves VNP14IMG data from the NASA LAADS DAAC using the provided authentication token.
- Fire pixel data are sourced in EPSG:4326 (geographic coordinates).
- During the *shapify* process, `.nc` files are extracted and converted to `.shp` files, retaining only fire pixels within the specified bounding box. Coordinates are automatically reprojected from EPSG:4326 to match the reference raster's projection, ensuring correct spatial alignment in the GUI.

### 3. Load Data

**Raster image:** The reference raster can be used here, or a raster from a different timestamp. To improve pan and zoom performance, navigate to the **Config** tab and reduce the **Max Raster Display** value (default is full resolution).

**Shapefiles:** Select a directory — the engine will discover all shapefiles within it and load them ordered by detection datetime (extracted from the third field of each filename, in UTC). Some processing is performed on load to optimise rendering; progress is shown in the bottom-left panel.

> **Note:** For correct visualisation, always load the raster before loading fire pixel data. Loading shapefiles first may result in distorted display scaling.

### 4. Explore the GUI

Once raster and fire pixel data are loaded, the GUI is ready for interactive use.

In the **Config** tab:

- **Fire pixel size** — adjust the display size of fire pixels (integer, minimum 1).
- **Colour levels** — configure up to 500 colour levels. Pixel colour shifts along the colour bar to reflect the age of detection (in days), with older detections displayed in progressively distinct colours.

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

---

## Package Structure

```
viirs/
├── __init__.py
├── README.md
│
├── fp_gui/                            # GUI viewer
│   ├── __main__.py                    # Entry point: python -m viirs.fp_gui
│   ├── config_dialog.py               # Settings/configuration dialog
│   ├── config.py                      # Tuneable constants
│   ├── fire_animation_controller.py   # Play/pause/step timing
│   ├── fire_data_manager.py           # Shapefile loading and NumPy frame cache
│   ├── fire_gui.py                    # Main window (orchestrator)
│   ├── fire_map_canvas.py             # Matplotlib rendering and click popups
│   ├── download_dialog.py             # Download UI dialog
│   ├── raster.py                      # GDAL raster reader
│   └── raster_loader.py               # Raster display wrapper
│
└── utils/                             # CLI tools (also importable as modules)
    ├── accumulate.py                  # Merge shapefiles with age tracking
    ├── bc_alber_to_latlon.py          # BC Albers projection → lat/lon conversion
    ├── download.py                    # Pull VNP14IMG from LAADS DAAC
    ├── laads_data_download_v2.py      # LAADS DAAC download helper (v2)
    ├── rasterize.py                   # Convert .shp to binary raster on Sentinel-2 grid
    ├── shapify.py                     # Convert .nc to UTM-projected .shp
    └── utm_to_latlon.py               # UTM coordinates → lat/lon conversion
```