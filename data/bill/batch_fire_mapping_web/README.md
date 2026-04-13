# Batch Fire Mapping - Web Interface

Interactive web interface for Sentinel-2 fire mapping using T-SNE, Random Forest, and HDBSCAN.

Wraps the same GPU-accelerated pipeline as `batch_fire_mapping` (`fire_mapping_cli.py`) but replaces the sequential CLI batch with a browser-based workflow where you can inspect each fire, tune parameters, and accept or retry results one at a time.

Last updated: 2026-04-13

---

## Requirements

- Python 3.10+
- All packages required by the batch pipeline (numpy, GDAL, scipy, matplotlib, geopandas, shapely, cuml, pyyaml)
- A CUDA-capable GPU (required by cuML for T-SNE and HDBSCAN)
- No additional web framework packages needed -- the server uses Python's built-in `http.server`

---

## Quick Start

```bash
python -m batch_fire_mapping_web \
    /path/to/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    /path/to/stack.bin \
    --skip_download \
    --out_dir ./results \
    --padding 0.2
```

Then open `http://localhost:8765` in your browser.

---

## Command Reference

```
python -m batch_fire_mapping_web  POLYGON_FILE  RASTER_FILE  [options]
```

### Required Arguments

| Argument | Description |
|---|---|
| `POLYGON_FILE` | BC historical fire perimeters shapefile (`.shp`). Must contain columns: `FIRE_NUMBE`, `FIRE_DATE`, `FIRE_YEAR`, `FIRE_SIZE_`. |
| `RASTER_FILE` | Sentinel-2 ENVI raster (`.bin` with accompanying `.hdr`). |

### Optional Arguments

#### Output

| Flag | Default | Description |
|---|---|---|
| `--out_dir DIR` | same directory as the raster | Root directory for all outputs (cache, accepted results, status index). |
| `--padding FLOAT` | `0.2` | Default fractional padding around the fire bounding box (0.2 = 20%). Can be changed per-fire in the web UI via the Re-crop button. |

#### Perimeter / VIIRS

| Flag | Default | Description |
|---|---|---|
| `--perimeter_mode {viirs,traditional}` | `viirs` | Classification hint source. `viirs`: use VIIRS active fire detections when available, fall back to the traditional polygon perimeter. `traditional`: always use the fire polygon perimeter. |
| `--skip_download` | off | Skip VIIRS download and shapify steps. Use this when VIIRS data was already downloaded by a previous run of `batch_fire_mapping` or `batch_fire_mapping_web`. |
| `--shapify_workers N` | `8` | Number of parallel workers for VIIRS `.nc` to `.shp` conversion. |

#### Sampling

| Flag | Default | Description |
|---|---|---|
| `--sample_rate FLOAT` | `0.05` | Fraction of crop pixels to sample for T-SNE embedding. Adjustable per-fire in the UI. |
| `--min_samples N` | `500` | Minimum number of samples (floor). |
| `--max_samples N` | `30000` | Maximum number of samples (ceiling). |

#### Server

| Flag | Default | Description |
|---|---|---|
| `--host ADDR` | `0.0.0.0` | Bind address. Use `127.0.0.1` to restrict to localhost only. |
| `--port N` | `8765` | Server port number. |

#### Authentication

| Flag | Default | Description |
|---|---|---|
| `--user USERNAME` | none | Username for HTTP Basic Auth. Can also be set via the `FIRE_AUTH_USER` environment variable. When omitted, no authentication is required. |
| `--password PASSWORD` | none | Password for HTTP Basic Auth. Can also be set via the `FIRE_AUTH_PASSWORD` environment variable. Prefer environment variables over CLI flags on shared systems (CLI args are visible in `ps`). |

---

## Examples

### First run (downloads VIIRS data)

```bash
python -m batch_fire_mapping_web \
    /data/GIS/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    /ram/T09UYU_prepost/stack.bin \
    --out_dir ./pgfc_results \
    --padding 0.2
```

### Skip VIIRS download (data already exists)

```bash
python -m batch_fire_mapping_web \
    /data/GIS/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    /ram/T09UYU_prepost/stack.bin \
    --skip_download \
    --out_dir ./pgfc_results
```

### Use traditional perimeters only (no VIIRS)

```bash
python -m batch_fire_mapping_web \
    /data/GIS/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    /ram/T09UYU_prepost/stack.bin \
    --perimeter_mode traditional \
    --out_dir ./pgfc_results
```

### With authentication (on HPC)

Using environment variables (recommended on shared systems):

```bash
export FIRE_AUTH_USER=admin
export FIRE_AUTH_PASSWORD=mysecretpassword

python -m batch_fire_mapping_web \
    perimeters.shp stack.bin \
    --skip_download --out_dir ./results
```

Or using CLI flags:

```bash
python -m batch_fire_mapping_web \
    perimeters.shp stack.bin \
    --skip_download --out_dir ./results \
    --user admin --password mysecretpassword
```

### Access from another machine on the same network

The server binds to `0.0.0.0` by default, so it is accessible from other machines on the same network. On startup, the server prints its network address:

```
============================================================
  Server ready!
  Local:   http://localhost:8765
  Network: http://142.36.19.133:8765
           http://heartofgold:8765
  Auth:    user=admin
  45 fire(s) available
============================================================
```

From another machine, open `http://142.36.19.133:8765` (or `http://heartofgold:8765`) in a browser.

Alternatively, use an SSH tunnel for encrypted access:

```bash
ssh -L 8765:localhost:8765 user@heartofgold
# Then open http://localhost:8765 on your local machine
```

---

## Web UI Workflow

### 1. Fire List (home page)

- Sortable table of all fires from the shapefile. Click any column header to sort.
- Collapsible **Filters** panel with:
  - **Year**: checkboxes (auto-populated from shapefile)
  - **Fire Number**: regex text input (e.g., `C11.*`, `G8`)
  - **Min Size (ha)**: number input
  - **Status**: checkboxes (pending / ready / mapped / accepted / error)
  - Live match count updates as you change filter values. Click **Apply** to filter the table, **Reset** to clear.
- **Open**: navigate to the fire mapping page.
- **Remove**: hide a fire from the list (confirm dialog). Does not delete any files.
- Status badges: green = accepted, blue = mapped, yellow = ready, gray = pending, red = error.
- A green checkmark badge indicates a fire that was previously accepted but is now being re-worked with different parameters.

### 2. Fire Mapping Page

**On page load**, the server automatically:
1. Crops the raster to the fire bounding box (with padding).
2. Runs VIIRS accumulation and rasterizes the hint perimeter.
3. Rasterizes the traditional polygon perimeter.
4. Generates preview images (post-fire, pre-fire, diff views).

**Image viewer** (left panel):
- Dropdown to switch between preview images (post, pre, diff, comparison).
- Scroll wheel to zoom in/out (zooms towards cursor).
- Click and drag to pan.
- Toolbar buttons: **+** (zoom in), **-** (zoom out), **Fit** (reset to fit).
- Images use nearest-neighbor rendering for crisp pixels at all zoom levels.

**Parameters** (right panel, collapsible sections):
- **Crop & Sampling**: padding (+ Re-crop button), sample rate, min/max samples, seed.
- **T-SNE**: perplexity, learning rate, max iterations, init method, components, random state, embed bands.
- **Random Forest**: number of estimators, max depth, max features, random state.
- **HDBSCAN**: controlled ratio, min samples.
- **Display**: contour width (integer pixels).

**Mapping workflow**:
1. Adjust parameters as needed.
2. Click **Map Fire**.
3. The console streams `fire_mapping_cli.py` output in real time.
4. When complete, the comparison figure appears in the image viewer.
5. Click **Accept** to save results, or adjust parameters and map again.

**GPU queue**: only one fire processes at a time. If another mapping is already running, the console shows the queue position and waits automatically.

**Re-mapping accepted fires**: changing padding and clicking Re-crop resets the fire to "ready" status. The previous accepted results remain on disk and are not overwritten until you Accept the new mapping.

**Remove**: the Remove button in the header hides the fire from the list (with confirmation). Accepted results on disk are not deleted.

---

## Output Structure

All outputs are written under `--out_dir` (or the raster directory if not set).

```
<out_dir>/
    fire_status.yaml                        # Status index for all fires
    .web_cache/                             # Temporary working files (safe to delete)
        <FIRE_NUMBE>/
            *_crop.bin / .hdr               # Cropped raster
            *_perimeter.bin                 # Rasterized polygon perimeter
            VIIRS_*.bin                     # Rasterized VIIRS hint
            *_classified.bin                # Classification output
            *_comparison.png                # Comparison figure
            *_brush_comparison.png          # Brush comparison figure
            previews/                       # Resampled PNGs for web display
    <FIRE_NUMBE>/                           # Accepted results (created on Accept)
        *_crop.bin / .hdr
        *_perimeter.bin
        *_classified.bin
        *_comparison.png
        *_brush_comparison.png
        *_params.yaml                       # Full parameter record + ML area estimate
```

| What | Location | Created when |
|---|---|---|
| Working files | `.web_cache/<FIRE_NUMBE>/` | Fire page is opened (prepare step). Overwritten on every re-map. |
| Preview images | `.web_cache/<FIRE_NUMBE>/previews/` | Prepare step. For web display only, not copied on Accept. |
| Accepted results | `<FIRE_NUMBE>/` | Accept button is clicked. Includes `_params.yaml` with ML area. |
| VIIRS downloads | `<raster_dir>/<raster_name>_VIIRS/` | VIIRS download step. Stored next to the raster, shared with `batch_fire_mapping`. |

Accepted fire folders have the same structure and file names as the `batch_fire_mapping` CLI output and are fully compatible. The `fire_status.yaml` index is also compatible -- web entries include `source: web`.

---

## Model Parameters

All parameters are forwarded to `fire_mapping_cli.py` and match the batch pipeline CLI flags:

| Parameter | UI Location | CLI Flag | Default |
|---|---|---|---|
| Padding | Crop & Sampling | `--padding` | 0.2 |
| Sample rate | Crop & Sampling | `--sample_rate` | 0.05 |
| Seed | Crop & Sampling | `--seed` | 123 |
| T-SNE perplexity | T-SNE | `--tsne_perplexity` | 60 |
| T-SNE learning rate | T-SNE | `--tsne_learning_rate` | 200 |
| T-SNE max iterations | T-SNE | `--tsne_max_iter` | 2000 |
| T-SNE init | T-SNE | `--tsne_init` | pca |
| T-SNE components | T-SNE | `--tsne_n_components` | 2 |
| T-SNE random state | T-SNE | `--tsne_random_state` | 42 |
| T-SNE embed bands | T-SNE | `--embed_bands` | all |
| RF estimators | Random Forest | `--rf_n_estimators` | 100 |
| RF max depth | Random Forest | `--rf_max_depth` | 15 |
| RF max features | Random Forest | `--rf_max_features` | sqrt |
| RF random state | Random Forest | `--rf_random_state` | 42 |
| HDBSCAN ctrl ratio | HDBSCAN | `--controlled_ratio` | 0.5 |
| HDBSCAN min samples | HDBSCAN | `--hdbscan_min_samples` | 20 |
| Contour width | Display | `--contour_width` | 1 |

---

## Known Limitations

- **HDBSCAN non-determinism**: cuML's GPU HDBSCAN does not support a random seed parameter. Identical runs may produce slightly different cluster assignments due to GPU parallelism. All other pipeline stages (T-SNE, Random Forest, sampling) are seeded and deterministic.
- **No TLS**: the server uses plaintext HTTP. On internal networks this is acceptable. For exposure over untrusted networks, use an SSH tunnel or place behind a TLS-terminating reverse proxy (e.g., nginx).
- **Single GPU queue**: only one fire maps at a time. Additional requests are queued and wait automatically.
