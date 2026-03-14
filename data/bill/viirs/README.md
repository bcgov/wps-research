# viirs — VIIRS Fire Pixel Processing Toolkit

*Last updated: March 14, 2026*

An end-to-end pipeline for downloading, converting, accumulating, and visualising VIIRS active fire pixel data from NASA. Designed specifically for [VNP14IMG](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP14IMG/#product-information) — the VIIRS/NPP Active Fires 6-Min L2 Swath 375m product.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Getting a LAADS DAAC Token](#getting-a-laads-daac-token)
4. [Launching the GUI](#launching-the-gui)
5. [Complete Workflow](#complete-workflow)
   - [Step 1 — Load a Raster](#step-1--load-a-raster)
   - [Step 2 — Download VIIRS Data](#step-2--download-viirs-data)
   - [Step 3 — Explore Loaded Data](#step-3--explore-loaded-data)
   - [Step 4 — Accumulate & Rasterize](#step-4--accumulate--rasterize)
6. [Folder Naming Conventions](#folder-naming-conventions)
7. [Command-Line Utilities](#command-line-utilities)
8. [GUI Reference](#gui-reference)

---

## Overview

This toolkit provides a complete, integrated workflow for working with VIIRS active fire data:

1. **Download** raw NetCDF (`.nc`) files from the NASA LAADS DAAC.
2. **Shapify** — convert `.nc` files to shapefiles (`.shp`) in the reference raster's projection.
3. **Accumulate** — merge shapefiles chronologically into cumulative snapshots, each including pixel age in days.
4. **Rasterize** — burn each accumulated shapefile onto the reference raster grid as a binary fire mask (`.bin`/`.hdr`).

A Tkinter graphical interface drives all four steps from a single **Download** button. Command-line utilities are also available for scripted use.

---

## Requirements

- Python 3.9+
- `gdal` / `osgeo`
- `netCDF4`
- `geopandas`, `shapely`, `pyproj`
- `numpy`, `matplotlib`
- `tkinter` (usually bundled with Python)

---

## Getting a LAADS DAAC Token

All downloads from NASA LAADS DAAC require an **Earthdata Login** account and an **application token**. This is free and takes a few minutes.

### 1 — Create an Earthdata Login account

1. Go to **https://urs.earthdata.nasa.gov/** and click **Register**.
2. Fill in the form (name, email, username, password) and verify your email.
3. Log in at **https://urs.earthdata.nasa.gov/**.

### 2 — Generate a LAADS DAAC application token

1. Go to **https://ladsweb.modaps.eosdis.nasa.gov**.
2. Click **Login** (top-right) and sign in with your Earthdata credentials.
3. After logging in, click your **username** (top-right) to open your profile.
4. Select **Generate Token** (or navigate to **My Account → App Keys / Tokens**).
5. Click **Generate Token**. A long alphanumeric string will be displayed — this is your token.
6. Copy the token immediately; it is only shown once. You can regenerate it at any time.

> **Note:** Tokens expire after a period of inactivity (typically 1 year). If downloads suddenly start failing with authentication errors, regenerate your token and update it in the GUI or token file.

### 3 — Make the token available to the GUI

The GUI looks for the token in two ways, in order:

**Option A — Token file (recommended, auto-loads on startup):**

Place the token in the file `/data/.tokens/laads` (no extension, no newline):

```bash
echo -n "YOUR_TOKEN_HERE" > /data/.tokens/laads
```

The GUI will read this file at startup and the token will be silently applied to every download session. No manual entry is required.

**Option B — Paste it in the GUI:**

If no token file is found, clicking **Download** opens a popup titled *LAADS DAAC Token Required*. Paste your token into the masked field and click **Set Token**. The token is held in memory for the session but is not saved to disk automatically.

---

## Launching the GUI

```bash
python -m viirs.fp_gui
```

The GUI opens maximised. Three toolbar rows appear at the top; the map canvas fills the rest of the window.

---

## Complete Workflow

### Step 1 — Load a Raster

The reference raster defines the spatial projection and bounding box for everything else. It must be loaded first.

**Load via the browser:**

1. In row 1, click **Browse** next to the **Raster** field.
2. Navigate to your raster file in the file browser. Supported formats: `.bin` (ENVI), `.hdr`, `.tif`/`.tiff`, and most GDAL-readable formats.
3. Click **Load** (the renamed Select button inside the browser).
4. The raster is displayed on the map canvas. The CRS is shown in the status bar (bottom-right).

**Load by pasting a path:**

You can also paste or type a full path directly into the **Raster** entry box and press **Enter**.

**What happens after loading:**

- The **Ref** field (row 3) is automatically set to the same raster. The Ref controls where accumulation output is saved; you can change it independently later.
- The expected VIIRS download folder (`<raster_name>_VIIRS/`) is computed and checked. If it already exists, shapefiles inside it are loaded automatically and the **Shapefiles** status changes to **loaded** (green).
- The CRS EPSG code is extracted and used for the download bounding box.

> **Performance tip:** Open the ⚙ Config dialog and reduce **Max Raster Display Dim** if panning and zooming feels slow. This downsamples the display only — it does not affect any analysis outputs.

---

### Step 2 — Download VIIRS Data

> Skip this step if you have already downloaded data for this raster in a previous session. On load, the GUI auto-detects the `_VIIRS` folder and reloads the shapefiles.

1. In row 2, enter the **Start** and **End** dates in `YYYY-MM-DD` format (e.g. `2025-09-01` to `2025-09-30`).
2. Click **⬇ Download**. A confirmation popup appears summarising:
   - The raster CRS
   - The date range
   - The download folder (named `<raster_basename>_VIIRS/`)
   - The accumulation output folder that will be created
3. Click the green **✔ Confirm** button to start.

**What the download pipeline does (fully automatic after Confirm):**

| Stage | What happens |
|---|---|
| **Download** | Queries the LAADS DAAC API for each day in the date range, downloads `.nc` files into `<raster_name>_VIIRS/YYYY/JJJ/`. Days that already have `.nc` files are skipped. Up to 16 concurrent downloads run in parallel. |
| **Shapify** | Converts each `.nc` file to a `.shp` file, reprojects fire pixel coordinates from EPSG:4326 to the reference raster's CRS, and filters to the raster bounding box. |
| **Accumulate** | Merges shapefiles chronologically into cumulative snapshots. Each snapshot includes all pixels from the start date up to that detection time, with an `age_days` column. |
| **Rasterize** | Burns each accumulated shapefile onto the reference raster grid, producing a binary fire mask (`.bin`/`.hdr`) for each snapshot. |

Progress is reported continuously in the bottom-left status bar and printed to the terminal.

**Smart accumulation — re-downloading or extending a date range:**

- **Exact match exists** (same start and end) → a popup informs you and no re-accumulation is run.
- **Same start, further end** (e.g. you extend from Sep 25 to Oct 5) → the existing accumulation folder is renamed to the new end date and accumulation runs incrementally.
- **Same start, shorter end** (e.g. you choose a narrower window when a longer one already exists) → a popup asks: *"An accumulation with the same start date and a further end date already exists. Do you want to create a new folder for this end date?"* Confirming creates a fresh folder; cancelling aborts.

---

### Step 3 — Explore Loaded Data

Once raster and shapefile data are loaded, the map canvas is interactive.

**Shapefiles status (row 1):**
- **loaded** (green) — shapefiles found and loaded successfully.
- **not found** (gray) — no shapefile directory found for this raster yet.

**Date filter (row 2):**

- The **Start** and **End** fields are pre-filled with the min/max dates from the loaded data.
- Edit either field and click **Apply** to filter the animation to a sub-range.

**Animation (row 2):**

| Control | Function |
|---|---|
| **▶ Play / ⏸ Pause** | Start or pause the fire pixel animation |
| **⏮** | Reset to frame 0 |
| **← / →** | Step back or forward by N frames (set N in the spinbox) |
| **ms** | Animation speed in milliseconds per frame |
| **Slider** | Scrub directly to any frame |

**Layer toggles (row 3, right side):**

- **Fire Pixels** — show/hide the fire pixel scatter overlay.
- **Background Image** — show/hide the raster background.

**Navigation (row 3):**

| Button | Mode |
|---|---|
| **Pan** | Click-drag to pan the map |
| **Zoom+** | Click to zoom in |
| **Zoom−** | Click to zoom out |
| **⌂** | Reset to full extent |

**Config (⚙ button, row 1):**

- **Fire pixel size** — display size of each fire pixel (auto-computed from raster resolution on load; integer, minimum 1).
- **Colour levels** — number of colour steps across the age gradient (up to 500). Newest detections are bright red; oldest fade toward warm brown.
- **Max Raster Display Dim** — maximum pixels in either dimension for raster display (reduce for performance).

---

### Step 4 — Accumulate & Rasterize

Accumulation and rasterization run **automatically** at the end of every download. If you have already downloaded data and want to re-run accumulation manually (e.g. after changing the date range), adjust the **Start/End** dates in row 2 and click **Download** again — existing `.nc` files are skipped, only the accumulation/rasterize stages re-run.

**Changing the reference raster (Ref field, row 3):**

The **Ref** field controls which raster is used as the grid template for rasterized output, and determines where the accumulation output folder is created.

- By default it mirrors the main raster loaded in row 1.
- To use a different grid: click **Browse** next to **Ref**, or paste a full path and press **Enter**.
- The accumulation output folder is always created **in the same directory as the Ref raster**.

**Output folder naming:**

```
<ref_basename>_<YYYYMMDD>_<YYYYMMDD>_ACCUMULATED/
```

For example, if the Ref is `/data/BC_2025.bin` and the date range is 2025-09-01 → 2025-09-30:

```
/data/BC_2025_20250901_20250930_ACCUMULATED/
    VIIRS_VNP14IMG_20250901T0840_20250901T0840.shp   (+ .dbf .prj .shx)
    VIIRS_VNP14IMG_20250901T0840_20250901T0840.bin   (+ .hdr)
    VIIRS_VNP14IMG_20250901T0840_20250902T1430.shp
    VIIRS_VNP14IMG_20250901T0840_20250902T1430.bin
    ...
    VIIRS_VNP14IMG_20250901T0840_20250930T2010.shp
    VIIRS_VNP14IMG_20250901T0840_20250930T2010.bin
```

Each shapefile contains all fire pixels from the start date up to that snapshot's end time. The `age_days` column is fractional days relative to each snapshot's end datetime (e.g. `34.06` means detected 34.06 days before the snapshot).

---

## Folder Naming Conventions

| Folder | Contents |
|---|---|
| `<raster_name>_VIIRS/` | Downloaded `.nc` files, organised as `YYYY/JJJ/` |
| `<raster_name>_VIIRS/<YYYY>/<JJJ>/` | Raw NetCDF files for one Julian day |
| Shapefiles are written alongside the `.nc` files (in the same `YYYY/JJJ/` directories) | |
| `<ref_name>_<START>_<END>_ACCUMULATED/` | Accumulated shapefiles + rasterized `.bin`/`.hdr` outputs |

---

## Command-Line Utilities

All utilities are also runnable from the command line independently of the GUI.

### Download

```bash
python -m viirs.utils.laads_data_download_v2 <url> <output_dir> <token>
```

### Shapify (NetCDF → Shapefile)

Convert VNP14IMG `.nc` files to shapefiles in the reference raster's CRS:

```bash
# Match a reference raster (recommended)
python -m viirs.utils.shapify /data/viirs/my_raster_VIIRS -r my_raster.bin

# With bounding box filter and 8 worker threads
python -m viirs.utils.shapify /data/viirs/my_raster_VIIRS -r my_raster.bin \
    --bbox -126.07 52.18 -124.37 53.21 -w 8

# Auto UTM (no reference raster)
python -m viirs.utils.shapify /data/viirs/my_raster_VIIRS

# Explicit CRS
python -m viirs.utils.shapify /data/viirs/my_raster_VIIRS --crs EPSG:3005
```

### Accumulate

Merge shapefiles into chronological cumulative snapshots:

```bash
python -m viirs.utils.accumulate /data/viirs/my_raster_VIIRS 20250901 20250930 \
    -r my_raster.bin
```

Output filenames use the actual detection timestamps from the source files. Each file includes an `age_days` column (fractional days relative to that snapshot's end datetime).

### Rasterize

Burn accumulated shapefiles onto the reference raster grid as binary fire masks:

```bash
# All shapefiles in a directory
python -m viirs.utils.rasterize /data/viirs/accumulated my_raster.bin /output/rasters \
    --buffer 375 -w 8

# Single shapefile
python -m viirs.utils.rasterize fire.shp my_raster.bin /output/rasters
```

---

## GUI Reference

### Toolbar layout

```
Row 1:  [⚙]  Raster: [____________] [Browse]  |  Shapefiles: loaded
Row 2:  Start: [__________]  End: [__________]  [Apply]  [⬇ Download]  |  [▶ Play] [⏮] [←] [N] [→]  ms:[___]  [slider]
Row 3:  Ref: [____________] [Browse]  |  [Pan] [Zoom+] [Zoom−] [⌂]  |  ☑ Fire Pixels  ☑ Background Image
```

### Status bar (bottom)

From left to right: current status message · current date · frame counter · pixel count · pixels in viewport · CRS.
