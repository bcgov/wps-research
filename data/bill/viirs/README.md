# viirs — VIIRS Fire Pixel Processing Toolkit 

<--- Last Update March 9, 2026 --->

End-to-end pipeline: download → convert → (optionally rasterize / accumulate) → visualise.

This package is designed specifically for `VNP14IMG data`

## Workflow

The idea: you start with nothing, pull raw VIIRS NetCDF data from NASA, convert it to shapefiles in your Sentinel-2 UTM projection, and then either view the fire pixel accumulation in the GUI or rasterize them for burn mapping.

If you don't need further conversion, you can handle everything in the GUI with ease.

### 1. Run the GUI

Launch:

```bash
python -m viirs.fp_gui
```

### 2. Download VIIRS VNP14IMG data (Can skip if you had the data downloaded in the previous run)

Use the `download` button to download the data, the reference file (.bin but can add more extension later) is needed for:

+ Extracting the projection (e.g EPSG 32609).

+ The bounding box of the raster image.

Steps:

1. After browsing (or pasting) the raster path in, click `Load Reference` for extraction, you will see the projection, bounding box in its projection.

    You can edit the bounding box (by using QGIS to see the correct values.)

2. `LAADS token` should be store in <u>/data/.tokens/laads</u>, it will be autoloaded as '***...' (or paste it in if you have your own.)

3. Select your date range of interest (standard format YYYY-MM-DD)

4. Your save directory (e.g /data/users/viirs_T09UYU). I recommend to have new dir for each download (don't overwrite on the existing data, may show `core dump` error)

5. Click download (The summary will be shown in the panel, the full prints are in the same terminal as the GUI.)

What does the download do:

+ It will pull the data from NASA portal using your LAADS tokens.

+ The fire pixel data are in EPSG 4326 (basically logitude/lattitude).

+ In shapify process, `.nc` files will be extracted and converted to `.shp` files, with only fire pixels in the `Bounding Box`. Most importantly, it will automatically transform the projection from 4326 to whatever the reference raster image is in, so that the fire scatters in the GUI will align with the raster image.

### 3. Load Data.

Raster Image:

+ It could be the same as the reference file (or at different timestamp).

+ A tip, if you want a lighter load (pan and zoom will be much faster), go into `config` and change `Max Raster Display` (default is full resolution, or 99999).

Shapefiles:

+ You need to load a directory, the engine will find all shapefiles and load by date time (datetime is in 3rd field of the file name (in UTC)).

+ As you load, it will perform some calculation so that your visualization will be more efficient (displayed in the bottom left panel).

+ If you load fire pixels before the raster, it will still work but it will be stretched too much. For the best visualization, always add a raster.

### 4. Play around with the GUI.

After raster and fire pixels are loaded. You can play around to get familiar with the GUI.

In the `config` tab, 

+ You can change the fire pixel size (Integer minimum 1).

+ You can config your colour level to max of 500 (the older the fire pixels from detection (in days), the colour will shift accordingly to the colour bar).


## Extra offers in the package.

### 1. Rasterize shapefiles for burn mapping

If you need binary fire masks on the Sentinel-2 grid (e.g. as hints for burn severity mapping):

```bash
# All shapefiles in a directory
python -m viirs.utils.rasterize /data/viirs/shapefiles sentinel2.bin /output/rasters --buffer 375 -w 8

# Single shapefile
python -m viirs.utils.rasterize fire.shp sentinel2.bin /output/rasters
```

### 2. Accumulate into a single shapefile

If you want one merged shapefile with all fire pixels and an `age_days` column (decimal days, e.g. `34.06`):

```bash
python -m viirs.utils.accumulate /data/viirs/shapefiles 20250401 20250930 -r <reference file>
```

Age is computed as `(end_date − detection_datetime)` in fractional days.

---

## Package Structure

```
viirs/
├── __init__.py
├── README.md
│
├── fp_gui/                            # GUI viewer
│   ├── __main__.py                    # python -m viirs.fp_gui
│   ├── config_dialog.py               # Settings/config dialog
│   ├── config.py                      # Tuneable constants
│   ├── fire_animation_controller.py   # Play/pause/step timing
│   ├── fire_data_manager.py           # Shapefile loading + numpy frame cache
│   ├── fire_gui.py                    # Main window (orchestrator)
│   ├── fire_map_canvas.py             # matplotlib rendering + click popups
│   ├── download_dialog.py             # Download UI dialog
│   ├── raster.py                      # GDAL raster reader
│   └── raster_loader.py               # Wraps raster.py for display
│
└── utils/                             # CLI tools (also importable)
    ├── accumulate.py                  # Merge shapefiles + age tracking
    ├── bc_alber_to_latlon.py          # BC Albers projection → lat/lon
    ├── download.py                    # Pull VNP14IMG from LAADS DAAC
    ├── laads_data_download_v2.py      # LAADS DAAC download v2
    ├── rasterize.py                   # .shp → binary raster on Sentinel-2 grid
    ├── shapify.py                     # .nc → .shp (UTM projected)
    └── utm_to_latlon.py               # UTM coordinates → lat/lon
```