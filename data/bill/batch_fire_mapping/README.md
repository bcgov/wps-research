# batch_fire_mapping

Batch fire-mapping pipeline for Sentinel-2 imagery using VIIRS active fire detections.

Given a shapefile of fire perimeters and a Sentinel-2 raster, this package:

1. Downloads VIIRS VNP14IMG data for the full date range spanned by the shapefile's `FIRE_YEAR` values.
2. Converts granule `.nc` files to point shapefiles.
3. For each fire polygon, crops the raster, accumulates VIIRS detections, and runs the fire-mapping model via `fire_mapping_cli.py`.
4. Saves a single-panel comparison figure per fire (mapping outline | VIIRS outline | traditional perimeter outline) and a before/after class-brush figure.

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (for T-SNE and Random Forest via `cuml`)
- GDAL / `osgeo.gdal`
- `geopandas`, `numpy`, `matplotlib`, `shapely`
- `viirs` package (in the project root)
- `fire_mapping` package (in the project root)
- `cpp/class_brush.exe` compiled binary
- NASA LAADS DAAC token (see [Authentication](#authentication))

---

## Authentication

VIIRS data is downloaded from [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/).
An authentication token is required.

The script looks for the token in `/data/.tokens/laads`.
If that file is not found, you will be prompted to paste the token in the terminal,
and you can optionally save it there for future runs.

To create the token file manually:

```bash
mkdir -p /data/.tokens
echo "YOUR_TOKEN_HERE" > /data/.tokens/laads
chmod 600 /data/.tokens/laads
```

---

## Usage

```
python batch_fire_mapping/run_fire_mapping.py  POLYGONS.shp  RASTER.bin  [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `POLYGONS.shp` | BC historical fire perimeters shapefile. Must contain columns `FIRE_NUMBE`, `FIRE_DATE` (YYYY-MM-DD), and `FIRE_YEAR`. |
| `RASTER.bin` | Sentinel-2 ENVI `.bin` raster (with accompanying `.hdr`). |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--year` | (all years) | Only process fires from this `FIRE_YEAR`. |
| `--out_dir` | same directory as `RASTER` | Root directory for all outputs. |
| `--padding` | `0.1` | Fractional padding on each side of the fire bounding box. `0.1` = 10% of fire width added left and right, 10% of fire height added top and bottom. |
| `--skip_download` | off | Skip VIIRS download and shapify — go straight to mapping. |
| `--shapify_workers` | `8` | Parallel shapify processes. |

**Sampling** (computed per fire, then forwarded to `fire_mapping_cli.py`):

| Flag | Default | Description |
|---|---|---|
| `--sample_rate` | `0.05` | Fraction of crop pixels to sample. `0.05` = 5% of the cropped raster area. The actual count is `clip(crop_w × crop_h × sample_rate, min_samples, max_samples)`, so small fires automatically get fewer samples and large fires are capped to stay fast. |
| `--min_samples` | `500` | Floor on sample count — ensures T-SNE has enough points even for tiny fires. |
| `--max_samples` | `30000` | Ceiling on sample count — prevents T-SNE from being slow on very large crops. |
| `--seed` | `123` | Sampling random seed. |

**Model hyperparameters** (forwarded to `fire_mapping_cli.py`):

| Flag | Default | Description |
|---|---|---|
| `--embed_bands` | (all bands) | 1-indexed comma-separated band list for T-SNE embedding. |
| `--rf_n_estimators` | `100` | RF number of trees. |
| `--rf_max_depth` | `15` | RF max tree depth. |
| `--rf_max_features` | `sqrt` | RF feature selection strategy. |
| `--rf_random_state` | `42` | RF random seed. |
| `--controlled_ratio` | `0.5` | HDBSCAN min-cluster-size ratio. |
| `--hdbscan_min_samples` | `20` | HDBSCAN min samples. |
| `--tsne_perplexity` | `60.0` | T-SNE perplexity. |
| `--tsne_learning_rate` | `200.0` | T-SNE learning rate. |
| `--tsne_max_iter` | `2000` | T-SNE iterations. |
| `--tsne_init` | `pca` | T-SNE initialisation (`pca` or `random`). |
| `--tsne_n_components` | `2` | T-SNE output dimensions. |
| `--tsne_random_state` | `42` | T-SNE random seed. |
| `--plot_downsample` | `2` | Downsample factor for the comparison PNG. |

---

## Example

```bash
python batch_fire_mapping/run_fire_mapping.py \
    data/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    data/S2C_MSIL1C_20251014T192401_20m.bin \
    --year 2025 \
    --padding 0.1
```

Outputs will be written to `data/fire_mapping_results/` (next to the raster).

---

## Output structure

```
<raster_dir>/                              # or --output_dir if specified
    fire_mapping_results/
        <FIRE_NUMBE>/
            <FIRE_NUMBE>_crop.bin               # Cropped Sentinel-2 subscene (ENVI)
            <FIRE_NUMBE>_crop.hdr
            VIIRS_VNP14IMG_<start>_<end>.shp    # Final accumulated VIIRS shapefile
            VIIRS_VNP14IMG_<start>_<end>.bin    # Rasterized VIIRS hint (0/1)
            <FIRE_NUMBE>_perimeter.bin          # Rasterized traditional perimeter (0/1)
            <FIRE_NUMBE>_crop_classified.bin    # Raw fire-mapping classification output
            <FIRE_NUMBE>_comparison.png         # Comparison figure (after class-brush)
            <FIRE_NUMBE>_brush_comparison.png   # Before vs after class-brush figure
```

Re-running the script **replaces** existing fire folders — each fire is always processed fresh.

---

### Comparison figure

Single panel with three polygon outlines (no fill) on a false-colour background (B12/B11/B9 post-fire bands):

| Outline | Colour | Metric |
|---|---|---|
| Our mapping (after class-brush) | Red contour | IoU vs perimeter, pixel accuracy |
| VIIRS accumulation | Orange | IoU vs perimeter, pixel accuracy |
| Traditional perimeter | Cyan | — (reference) |

The figure title includes the fire number and the accumulation date range.
The start date is the earliest VIIRS detection inside the polygon (or `FIRE_DATE − 5 days` if none found).

---

## VIIRS download structure

Downloaded granules are stored next to the raster under `<raster_name>_VIIRS/`:

```
<raster_dir>/<raster_name>_VIIRS/
    VNP14IMG/
        YYYY/
            DDD/
                VNP14IMG.AYYYYDDD.HHMM.*.nc
```

This mirrors the layout used by the VIIRS GUI so that existing downloads are
automatically detected and skipped on subsequent runs.
The download date range spans from January 1 of the earliest `FIRE_YEAR` to
December 31 of the latest `FIRE_YEAR` in the shapefile (filtered by `--year` if given).

---

## Notes

- Processing is sequential (one fire at a time) because the GPU is used by `fire_mapping_cli.py`.
- VIIRS download uses multi-threaded `ThreadPoolExecutor`; shapify runs in parallel via `ProcessPoolExecutor` (GDAL is not thread-safe).
- If a fire polygon lies entirely outside the raster, or the cropped region has no raster data, that fire is skipped with a status message.
- Already-downloaded granules and already-shapified files are detected and skipped; only the fire output folders are always recreated from scratch.

---

## Related files

| File | Location | Description |
|---|---|---|
| `run_fire_mapping.py` | `batch_fire_mapping/` | This script (entry point) |
| `fire_mapping_cli.py` | `py/fire_mapping/` | Non-GUI fire-mapping pipeline |
| `fire_mapping.py` | `py/fire_mapping/` | Interactive GUI version |
| `class_brush.exe` | `cpp/` | Connected-component brush binary |
| `download.py` | `viirs/utils/` | LAADS DAAC download utility |
| `accumulate.py` | `viirs/utils/` | VIIRS point accumulation |
| `rasterize.py` | `viirs/utils/` | Shapefile → raster conversion |
| `shapify.py` | `viirs/utils/` | `.nc` → `.shp` conversion |
