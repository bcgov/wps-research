# VIIRS Fire Pixel Accumulation Viewer

A tkinter + matplotlib GUI for visualising the temporal accumulation of VIIRS VNP14IMG fire detections. Optionally displays a raster background (e.g. Sentinel-2 ENVI/GeoTIFF); falls back to a black canvas when no raster is provided.

---

## Quick Start

```bash
# Install dependencies
pip install geopandas matplotlib numpy pandas gdal

# Run the viewer
cd fire_viewer
python main.py
```

---

## How It Works

1. **Select a shapefile directory** — the app recursively scans all subdirectories for `.shp` files. Files whose names match `VIIRS_VNP14IMG_UTM<YYYYMMDD>T<HHMM>` get their datetime parsed from the filename. Files that don't match the naming convention are still loaded, using their file modification time as a fallback datetime.

2. **Select a raster file** (optional) — any GDAL-readable format (ENVI `.hdr`/`.bin`, GeoTIFF, etc.) displayed as the map background. The raster is loaded via the included `Raster` class (GDAL-based, no rasterio dependency). If provided, the raster defines the viewport and fire pixels outside its extent are discarded. If omitted, the viewer derives the extent from the fire pixel coordinates with 5% padding and uses a black background.

3. **Set start / end dates** — auto-populated from the data range. Edit and click *Apply Date Filter* to restrict the time window.

4. **Play / Pause / Step** — the animation steps through **every calendar day** in the range (even days with no new detections). Fire pixels accumulate and never disappear.

5. **Skip N days** — jump forward or backward by a configurable number of days (default 7). Useful for quickly scrubbing through long time ranges.

6. **Age-based colouring** — uses a discrete colour level system (configurable, default 100 levels). Each day ages the pixel by one level. Day 0 = bright red, day 100+ = palest yellow. The transition is consistent — each day shifts exactly one colour step.

7. **Click a pixel** — a resizable popup table shows all attributes (lat, lon, UTM, FRP, confidence, T4, T5, age_days, etc.). The popup remembers its size, position, and column widths across clicks.

8. **Toggle layers** — checkboxes with green/black indicators to independently show/hide the background image and fire pixels.

---

## Controls Reference

| Control | Description |
|---|---|
| **Shapefile Dir** | Browse for directory; recursively finds all `.shp` files |
| **Raster File** | Browse for ENVI (`.hdr`, `.bin`) or GeoTIFF (`.tif`). Optional |
| **Load Data** | Loads raster (if any) and shapefiles, clips, precomputes frames |
| **Start / End Date** | YYYY-MM-DD; auto-filled, editable |
| **Apply Date Filter** | Reload only files within the date window |
| **▶ Play / ⏸ Pause** | Start/stop the day-by-day animation |
| **← −1 Day / +1 Day →** | Step one day forward or backward |
| **← Skip N Days / Skip N Days →** | Jump by N days (configurable spinner) |
| **⏹ Reset** | Stop animation and clear display |
| **Speed (ms)** | Delay between frames (lower = faster) |
| **Frame slider** | Drag to scrub to any date |
| **Scatter Size** | Point size for fire pixels |
| **Colour Levels** | Number of discrete red→yellow steps (default 100) |
| **Show Background Image** | Toggle raster visibility (green = on, black = off) |
| **Show Fire Pixels** | Toggle scatter visibility |

---

## Architecture

```
fire_viewer/
├── main.py                      # Entry point
├── config.py                    # All tuneable constants
│                                   - Scatter size, animation speed
│                                   - Colour levels, endpoints (red → yellow)
│                                   - Filename regex, popup columns
│                                   - Max parallel workers (default 64)
│
├── raster.py                    # Raster class (user's GDAL-based reader)
│                                   - Reads ENVI / GeoTIFF via osgeo.gdal
│                                   - Band selection, metadata extraction
│
├── raster_loader.py             # RasterLoader (wraps Raster class)
│                                   - Computes extent from GeoTransform
│                                   - Normalises image for display
│
├── fire_data_manager.py         # FireDataManager
│                                   - Recursive shapefile scanning
│                                   - Parallel loading (ThreadPoolExecutor)
│                                   - Extracts numpy arrays for animation
│                                   - Frame precomputation (pure numpy)
│                                   - FrameData: lightweight per-frame container
│
├── fire_map_canvas.py           # FireMapCanvas
│                                   - matplotlib figure in tkinter
│                                   - Raster background or black canvas
│                                   - Scatter with discrete colour lookup table
│                                   - Click → resizable popup with saved layout
│                                   - Layer visibility toggles
│
├── fire_animation_controller.py # FireAnimationController
│                                   - Play / pause / step / jump / jump_by(N)
│                                   - tkinter .after() scheduling
│
└── fire_gui.py                  # FireAccumulationGUI
                                    - Main window (starts maximised)
                                    - File browsers, date inputs, playback
                                    - Skip-N-days controls
                                    - Scatter size, colour levels spinners
                                    - Green/black toggle checkboxes
                                    - Status bar with date, frame, pixel count
```

---

## Performance

Shapefile loading is parallelised across up to 64 threads (configurable via `MAX_WORKERS` in `config.py`). After loading, all coordinate and date data is extracted into pure numpy arrays. Frame precomputation runs once — computing the index mask and age array for every calendar day in the range. During animation, no pandas or GeoDataFrame operations occur. Each frame is a dictionary lookup returning a `FrameData` object with numpy arrays that feed directly into matplotlib's scatter. The GeoDataFrame is only touched when you click a pixel to show the popup.

---

## Customisation

### Hardcode paths for debugging
In `fire_gui.py` `__init__`, set default values:
```python
self._shapefile_dir_var = tk.StringVar(value="/path/to/shapefiles")
self._raster_path_var = tk.StringVar(value="/path/to/raster.hdr")
```

### Colour scheme
Edit in `config.py`:
```python
N_COLOUR_LEVELS = 100                       # steps from red to yellow
COLOUR_NEWEST = (1.0, 0.0, 0.0, 1.0)       # bright red (RGBA)
COLOUR_OLDEST = (1.0, 1.0, 0.6, 1.0)       # pale yellow (RGBA)
```

### Popup columns
Control which attributes appear first in the click-popup:
```python
# config.py
DEFAULT_POPUP_COLUMNS = [
    "latitude", "longitude", "utm_x", "utm_y",
    "FRP_MW", "confidence", "T4", "T5",
    "day", "line", "sample",
    "detection_datetime", "age_days",
]
```
Any columns in the shapefile not listed here will still appear below these.

### Filename pattern
If your shapefiles use a different naming convention, edit in `config.py`:
```python
FILENAME_DATETIME_PATTERN = r"(\d{8}T\d{4})"
FILENAME_DATETIME_FORMAT = "%Y%m%dT%H%M"
```
Files that don't match will still be loaded using their file modification time.

---

## Dependencies

| Package    | Purpose                          |
|------------|----------------------------------|
| geopandas  | Shapefile I/O                    |
| osgeo/gdal | Raster (ENVI/TIFF) I/O          |
| matplotlib | Plotting & embedded canvas       |
| numpy      | Array operations & frame cache   |
| pandas     | DataFrame for popup lookups      |
| tkinter    | GUI (ships with Python)          |

---

## Notes

- The **date is extracted from the filename**, not from any attribute column inside the shapefile.
- `age_days` in the popup is computed live from the current animation date, not from a stored column.
- Even calendar days with **no new fire data** are animated (existing pixels age by one day and shift one colour level).
- The raster and shapefile CRS must already match (no on-the-fly reprojection).
- The raster is optional. Without it, fire pixels display on a black background with extent derived from the data.
- All `.shp` files found recursively are loaded, regardless of naming convention.