# viirs VNP14 — VIIRS Fire Pixel Processing Toolkit

End-to-end pipeline: download → convert → (optionally rasterize / accumulate) → visualise.

---

## Workflow

The idea: you start with nothing, pull raw VIIRS NetCDF data from NASA, convert it to shapefiles in your Sentinel-2 UTM projection, and then either view the fire pixel accumulation in the GUI or rasterize them for burn mapping.

### 1. Figure out your bounding box

You have a Sentinel-2 scene in UTM. You need lat/lon for the LAADS DAAC download URL.

```bash
python -m viirs.utils.utm_to_latlon
```
`current version`: specify utm coordinates in file.

This prints `W`, `S`, `E`, `N` in lat/lon and a ready-to-paste `regions=` URL parameter for LAADS DAAC.

### 2. Download VIIRS VNP14IMG data

Edit `download_vnp14.py` — set your LAADS token, date range (`downloadStartDay` / `downloadEndDay`), output directory, and paste the `regions=` bbox from step 1 into the download URL. Then run:

```bash
python -m viirs.utils.download
```

You get a directory tree: `VNP14IMG/YYYY/DDD/*.nc` — one NetCDF per granule.

### 3. Convert NetCDF → shapefiles

Each `.nc` becomes a shapefile with fire points reprojected to your UTM zone.



```bash
# Point at the entire download directory — finds all .nc recursively
python -m viirs.utils.shapify /data/viirs/VNP14IMG \
       --bbox -126.07 52.18 -124.37 53.21 \
       -w 8

# Or a single file
python -m viirs.utils.shapify VNP14IMG.A2025245.1012.002.nc

# Or just the current directory (default)
python -m viirs.utils.shapify (same as python -m viirs.utils.shapify .)
```

**Recommendation for multiple zone usage**: flags of `utm-zone` and `hemisphere` are optional, if data span across multiple zones, this should not be used (it might distort the points).

Output: `VIIRS_VNP14IMG_<YYYYMMDD>T<HHMM>.shp` next to each `.nc`.  
The datetime in the filename is what the GUI uses for temporal ordering.

### 4. View fire accumulation in the GUI

Now you have a directory of shapefiles with datetime-stamped names. Launch:

```bash
python -m viirs.fp_gui
```

In the GUI:
- **Shapefile Dir** — browse to where the `.shp` files are (scans recursively)
- **Raster File** — optionally pick your Sentinel-2 ENVI `.bin` as background. If you skip this, the background is black and no pixels are clipped
- **Load Data** — loads everything, clips to raster extent if provided, precomputes all frames
- **Play / Step / Skip N days** — watch fire pixels accumulate day by day. Newest = red, oldest = pale yellow
- **Click any pixel** — popup shows lat, lon, UTM, FRP, confidence, T4, T5, age, source file

### 5. (Optional) Rasterize shapefiles for burn mapping

If you need binary fire masks on the Sentinel-2 grid (e.g. as hints for burn severity mapping):

```bash
# All shapefiles in a directory
python -m viirs.utils.rasterize /data/viirs/shapefiles sentinel2.bin /output/rasters --buffer 375 -w 8

# Single shapefile
python -m viirs.utils.rasterize fire.shp sentinel2.bin /output/rasters
```

Each shapefile becomes a `.bin`/`.hdr` ENVI raster with 1 = fire, 0 = no fire, buffered by 375m (one VIIRS pixel) by default.

### 6. (Optional) Accumulate into a single shapefile

If you want one merged shapefile with all fire pixels and an `age_days` column (decimal days, e.g. `34.06`):

```bash
python -m viirs.utils.accumulate /data/viirs/shapefiles 20250401 20250930
```

Age is computed as `(end_date − detection_datetime)` in fractional days.

---

## Package Structure

```
viirs/
├── __init__.py
├── README.md
│
├── fp_gui/                          # GUI viewer
│   ├── __main__.py                  # python -m viirs.fp_gui
│   ├── config.py                    # Tuneable constants
│   ├── fire_gui.py                  # Main window (orchestrator)
│   ├── fire_data_manager.py         # Shapefile loading + numpy frame cache
│   ├── fire_animation_controller.py # Play/pause/step timing
│   ├── fire_map_canvas.py           # matplotlib rendering + click popups
│   ├── raster.py                    # GDAL raster reader (your class)
│   └── raster_loader.py             # Wraps raster.py for display
│
└── utils/                           # CLI tools (also importable)
    ├── download_vnp14.py            # Pull VNP14IMG from LAADS DAAC
    ├── vnp14_to_shp.py              # .nc → .shp (UTM projected)
    ├── rasterize_batch.py           # .shp → binary raster on Sentinel-2 grid
    ├── accumulate_fp.py             # Merge shapefiles + age tracking
    └── utm_to_latlon.py             # UTM bbox → lat/lon for LAADS URL
```

---

## Using utils as Python functions

Every util works both from the command line and as an import:

```python
from viirs.utils.shapify import process_file
process_file("file.nc", utm_zone=9, hemisphere="N", bbox=[-126, 52, -124, 53])

from viirs.utils.accumulate import accumulate
gdf = accumulate("/data/shp", "20250401", "20250930")

from viirs.utils.utm_to_latlon import utm_bbox_to_latlon
w, s, e, n = utm_bbox_to_latlon(699960, 5790240, 809760, 5900040, zone=9)
```

---

## Dependencies

```bash
pip install geopandas matplotlib numpy pandas netCDF4 pyproj shapely
# GDAL: install system-wide or via conda (needed for raster.py and rasterize_batch)
```

No `pip install` for the package itself — just have `viirs/` on your Python path.