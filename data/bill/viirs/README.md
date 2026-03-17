# viirs — VIIRS Fire Pixel Processing Toolkit

*Last updated: March 17, 2026*

Pipeline for downloading, converting, accumulating, and visualising [VNP14IMG](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP14IMG/#product-information) (VIIRS/NPP Active Fires 375m) data from NASA LAADS DAAC.

---

## Overview

1. **Download** raw NetCDF (`.nc`) files from LAADS DAAC.
2. **Shapify** — convert `.nc` to shapefiles (`.shp`) in the reference raster's projection.
3. **Accumulate** — merge shapefiles chronologically into cumulative snapshots with `age_days`.
4. **Rasterize** — burn each accumulated shapefile onto the reference raster grid as a binary fire mask (`.bin`/`.hdr`).

All four steps run automatically from a single **Download** button in the GUI. Command-line utilities are also available.

```bash
python -m viirs.fp_gui
```

---

## Requirements

`gdal`/`osgeo`, `netCDF4`, `geopandas`, `shapely`, `pyproj`, `numpy`, `matplotlib`, `tkinter`.

---

## LAADS DAAC Token

Downloads require a free Earthdata Login account and application token.

1. Register at **https://urs.earthdata.nasa.gov/**.
2. Log in at **https://ladsweb.modaps.eosdis.nasa.gov**, go to your profile, and click **Generate Token**.
3. Provide the token to the GUI via one of:
   - **Token file (recommended):** `echo -n "YOUR_TOKEN" > /data/.tokens/laads` — auto-loaded on startup.
   - **Paste in GUI:** clicking **Download** without a token file prompts for manual entry (session only).

---

## Workflow

### 1 — Load a Raster

The reference raster defines the projection and bounding box. Load it first via **Browse** or by pasting a path into the **Raster** field and pressing **Enter**. Supported formats: `.bin` (ENVI), `.hdr`, `.tif`/`.tiff`, and most GDAL-readable formats.

After loading:
- **Ref** (row 3) is set to the same raster. It controls the accumulation output grid/directory — change it independently if needed.
- If `<raster_name>_VIIRS/` already exists, shapefiles are auto-loaded (green **loaded** status).
- Band selection is reset. Use the **Band** button to pick up to 3 bands for RGB display.

> Reduce **Max Raster Display Dim** in the **Config** dialog if panning feels slow (display only, does not affect outputs).

### 2 — Download VIIRS Data

> Skip if data already exists — the GUI auto-detects the `_VIIRS` folder on raster load.

1. Enter **Start** and **End** dates (`YYYY-MM-DD`) in row 2.
2. Click **Download**, review the confirmation popup, and click **Confirm**.

The pipeline runs automatically: Download (16 concurrent workers, skips existing days) → Shapify → Accumulate → Rasterize. Progress is shown in the status bar.

**Smart accumulation** when re-downloading:
- **Exact match** → skipped.
- **Same start, further end** → existing folder is renamed and accumulation extends incrementally.
- **Same start, shorter end** → prompts whether to create a new folder.

### 3 — Explore Loaded Data

| Control | Function |
|---|---|
| **Start/End + Apply** | Filter animation to a date sub-range |
| **Play/Pause, Step, Slider** | Animate fire pixel detections over time |
| **Band** | Select up to 3 bands for RGB display |
| **Pan / Zoom+ / Zoom− / Home** | Map navigation |
| **Fire Pixels / Background Image** | Layer visibility toggles |

**Config** (gear button): fire pixel size, colour levels (age gradient), max raster display dimension.

### 4 — Accumulate & Rasterize

Runs automatically after download. To re-run manually, adjust dates and click **Download** again — existing `.nc` files are skipped.

The **Ref** field controls the output grid template and directory. Output folder:

```
<ref_basename>_<YYYYMMDD>_<YYYYMMDD>_ACCUMULATED/
```

Each snapshot contains all fire pixels from start to that detection time, with `age_days` (fractional days relative to the snapshot's end datetime).

### 5 — Sentinel-2 Fire Mapping

Launches `fire_mapping.py` with the loaded raster and a matching `.bin` fire mask.

**Prerequisites:** raster loaded + shapefiles loaded (green status).

The GUI searches the **main raster's directory** for an `_ACCUMULATED` folder whose start date matches and end date covers the requested range, then selects the `.bin` with end date closest to (but not exceeding) the End date in the box.

If no matching folder or `.bin` is found, a dialog offers **Download and Launch Again** — this triggers the full download pipeline and automatically re-launches fire mapping on completion.

---

## Folder Naming Conventions

| Folder | Contents |
|---|---|
| `<raster_name>_VIIRS/YYYY/JJJ/` | Downloaded `.nc` files + converted `.shp` files |
| `<ref_name>_<START>_<END>_ACCUMULATED/` | Accumulated shapefiles + rasterized `.bin`/`.hdr` |

---

## Command-Line Utilities

### Download

```bash
python -m viirs.utils.laads_data_download_v2 <url> <output_dir> <token>
```

### Shapify

```bash
python -m viirs.utils.shapify /data/viirs/my_raster_VIIRS -r my_raster.bin
python -m viirs.utils.shapify /data/viirs/my_raster_VIIRS -r my_raster.bin \
    --bbox -126.07 52.18 -124.37 53.21 -w 8
```

### Accumulate

```bash
python -m viirs.utils.accumulate /data/viirs/my_raster_VIIRS 20250901 20250930 \
    -r my_raster.bin
```

### Rasterize

```bash
python -m viirs.utils.rasterize /data/viirs/accumulated my_raster.bin /output/rasters \
    --buffer 375 -w 8
```

---

## GUI Reference

```
Row 1:  [Config]  Raster: [____] [Browse]  |  Shapefiles: loaded          [Sentinel-2 Fire Mapping]
Row 2:  Start: [____]  End: [____]  [Apply]  [Download]  |  [Play] [Reset] [Step] ms:[__] [slider]
Row 3:  Ref: [____] [Browse]  |  [Band] [Pan] [Zoom+] [Zoom-] [Home]  |  Fire Pixels  Background Image
```

**Status bar:** status message, current date, frame counter, pixel count, pixels in viewport, CRS.
