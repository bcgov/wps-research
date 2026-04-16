# batch_fire_mapping_web

Interactive web interface for mapping wildfire burn areas from Sentinel-2 satellite imagery. Uses a machine learning pipeline (T-SNE dimensionality reduction, Random Forest classification, HDBSCAN clustering) accelerated on GPU to classify burned vs. unburned pixels, then lets users visually review and accept results through a browser.

This is the web companion to the `batch_fire_mapping` CLI. It wraps the same underlying pipeline but replaces the sequential batch workflow with an interactive one: users can inspect each fire, tune parameters, compare results side-by-side, and build up a parameter knowledge base over time.

---

## What this tool does

1. **Loads** a shapefile of historical fire perimeters and a Sentinel-2 raster stack.
2. **Optionally downloads** VIIRS active fire satellite detections to use as classification hints.
3. **Serves a web UI** where users can:
   - Browse all fires in the shapefile, sorted and filtered by year, size, status, etc.
   - Open any fire to see post-fire, pre-fire, and difference imagery at full pixel resolution.
   - Run the ML classification pipeline with adjustable parameters.
   - Run **serial mapping** (multiple parameter sets at once) and compare results in a gallery.
   - Accept the best result, which saves it to disk and logs the parameters for future use.
   - Batch-map many fires at once using recommended settings.
4. **Learns over time**: accepted parameters are logged. Future serial mappings rank parameter sets by how well they performed on similar fires (same region, similar size), so results improve as more fires are processed.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (required by cuML for T-SNE and HDBSCAN)
- Python packages: numpy, GDAL, scipy, matplotlib, geopandas, shapely, cuml, pyyaml
- No web framework needed -- the server uses Python's built-in `http.server`

---

## Quick start

```bash
python -m batch_fire_mapping_web \
    /path/to/fire_perimeters.shp \
    /path/to/sentinel2_stack.bin \
    --skip_download \
    --out_dir ./results
```

Then open `http://localhost:8765` in your browser.

Authentication is optional. To enable it, set passwords via environment variables:

```bash
export FIRE_ADMIN_PASSWORD=<your-admin-password>
export FIRE_USER_PASSWORD=<your-user-password>

python -m batch_fire_mapping_web \
    fire_perimeters.shp sentinel2_stack.bin \
    --skip_download --out_dir ./results
```

> **Tip**: Use environment variables instead of CLI flags on shared systems, since CLI arguments are visible in process listings.

---

## Command reference

```
python -m batch_fire_mapping_web  POLYGON_FILE  RASTER_FILE  [options]
```

### Required arguments

| Argument | Description |
|---|---|
| `POLYGON_FILE` | Fire perimeters shapefile (`.shp`). Must have columns: `FIRE_NUMBE`, `FIRE_DATE`, `FIRE_YEAR`, `FIRE_SIZE_`. |
| `RASTER_FILE` | Sentinel-2 ENVI raster (`.bin` with companion `.hdr`). |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--out_dir DIR` | raster directory | Root directory for all outputs. |
| `--perimeter_mode {viirs,traditional}` | `viirs` | Hint source. `viirs` uses VIIRS active fire data when available, falls back to polygon perimeter. `traditional` always uses the polygon. |
| `--skip_download` | off | Skip VIIRS download (use when data already exists from a previous run). |
| `--shapify_workers N` | `8` | Parallel workers for VIIRS NetCDF-to-shapefile conversion. |
| `--sample_rate FLOAT` | `0.05` | Fraction of crop pixels to sample for T-SNE. Adjustable per-fire in the UI. |
| `--min_samples N` | `500` | Minimum samples (floor). |
| `--max_samples N` | `30000` | Maximum samples (ceiling). |
| `--host ADDR` | `0.0.0.0` | Bind address. Use `127.0.0.1` to restrict to localhost. |
| `--port N` | `8765` | Server port. |
| `--admin_password` | none | Admin password (or env `FIRE_ADMIN_PASSWORD`). |
| `--user_password` | none | User password (or env `FIRE_USER_PASSWORD`). Requires admin password to also be set. |

---

## Authentication and access control

Authentication is **off by default**. When both an admin and user password are configured, the server enables a two-role system:

- **Admin**: full access. Can approve or block other users' IP addresses via the admin dashboard (`/admin`). Can edit recommended settings.
- **User**: fire mapping access. Must have their IP approved by an admin before they can see any content.

New user IPs appear as "pending" on the admin dashboard. The admin approves or blocks them. Approved/blocked state persists across server restarts.

When no passwords are set, everyone has full access with no login required.

---

## How the web UI works

### Fire list (home page)

A sortable, filterable table of every fire in the shapefile.

- **Columns**: fire number, date, year, size (ha), agreement %, ML area, status.
- **Filters**: year, fire number (regex), minimum size, status. Filter state persists across page navigation.
- **Batch actions**: select fires with checkboxes, then "Map Selected (with settings)" to batch-process them, or "Remove Selected" to hide them from the list.
- **PDF report**: generates a downloadable PDF of all accepted fires.
- **Status badges**: pending (gray), ready (yellow), mapping (blue), mapped (blue), accepted (green), error (red).

### Fire mapping page

Opening a fire takes you to the mapping page. On load, the server automatically:

1. Crops the raster to the fire's bounding box (with configurable padding).
2. Rasterizes the VIIRS hint or polygon perimeter.
3. Generates preview images (post-fire, pre-fire, difference).

#### Image viewer (left side)

- **Split view**: side-by-side comparison. Post-fire on the left, ML classification on the right (opens automatically when results are available).
- **Synced zoom/pan**: both panes move together by default. Zoom preserves your position when toggling split or collapsing the control panel -- you won't lose your place on large fires.
- **Pixel-perfect rendering**: `image-rendering: pixelated` at all zoom levels.
- **View dropdown**: switch between post-fire, pre-fire, difference, ML classification, comparison figure, brush comparison.

#### Parameters (right side)

Collapsible sections for every pipeline parameter:

- **Crop & Sampling**: padding, sample rate, min/max samples, seed.
- **T-SNE**: perplexity, learning rate, max iterations, init method, components, random state, embed bands.
- **Random Forest**: estimators, max depth, max features, random state.
- **HDBSCAN**: controlled ratio, min samples.
- **Display**: contour width.
- **Notes**: free-text annotations per fire (e.g., "cloud contamination"). Persisted on accept.

#### Mapping

- **Map Fire**: runs the pipeline with the parameters currently in the form.
- **Map Fire / with settings**: applies the recommended settings for the fire's size range, then maps.
- **Runs (N)**: set N > 1 to run multiple mappings with varied HDBSCAN parameters. All runs appear as cards in a results gallery, ranked by agreement score.
- **Console**: streams pipeline output in real time. Persists across page navigation.

#### Results gallery

Every mapping (even a single run) produces a result card showing:

- Thumbnail of the ML classification overlay.
- Agreement score (IoU between ML result and hint perimeter).
- ML fire area estimate in hectares.
- Expandable parameter details.
- **Accept** button to save that result.

When re-mapping a previously accepted fire, the old result appears as a "Previously accepted" card (gold border) so you can compare old vs. new before deciding.

#### Accepting

Clicking Accept on a result card:
- Copies all outputs to the canonical output directory.
- Writes a `_params.yaml` file with the full parameter record and ML area estimate.
- Logs parameters to `accepted_params.csv` (feeds the learning system).
- Clears the results gallery and serial cache files.
- Shows a confirmation dialog if overwriting a previously accepted result.

### Navigation

- **Prev / Next** buttons move through fires in the current filtered/sorted order without returning to the list.
- **Back to fire list** preserves your filters.

### GPU queue

Only one fire maps at a time (GPU serialization). Additional requests queue automatically. Queue state is visible in the admin dashboard.

---

## Serial mapping and parameter learning

This is the system's core learning mechanism. Instead of manually tuning parameters for each fire, you run N mappings at once and pick the best.

### How it works

1. Set N (1-10) and click Map Fire.
2. The system finds the top N parameter sets using hierarchical context matching:
   - **Level 1**: fires in the same zone (e.g., C1) and size bucket, if enough accepted examples exist.
   - **Level 2**: same region (e.g., all C fires) and size bucket.
   - **Level 3**: same size bucket, any region.
   - **Level 4**: defaults from `recommended_settings.yaml` (cold start).
3. Run 1 does the full pipeline (T-SNE + Random Forest + HDBSCAN) and caches intermediate state. Runs 2-N load the cache and only re-run HDBSCAN with different `min_samples` values, which is much faster.
4. Results appear in the gallery as they complete. Pick the best and accept it.
5. The accepted parameters feed back into rankings for the next fire.

### Size buckets

| Bucket | Range |
|---|---|
| 1 | 0 - 10 ha |
| 2 | 10 - 50 ha |
| 3 | 50 - 100 ha |
| 4 | 100 - 500 ha |
| 5 | 500 - 1,000 ha |
| 6 | 1,000 - 5,000 ha |
| 7 | 5,000+ ha |

### The learning loop

```
Accept fire -> params logged to CSV -> next serial map reads CSV ->
better rankings -> better results -> more accepts -> ...
```

Early on, the system uses defaults from `recommended_settings.yaml`. After enough accepted fires in a given context, it uses real performance data.

---

## Recommended settings

The file `recommended_settings.yaml` defines parameter presets by fire size range. Loaded automatically on startup.

```yaml
- min_ha: 0
  max_ha: 500
  params:
    padding: 0.1
    sample_rate: 0.05
    embed_bands: '5,6,7,9,10,11,13,14,15'
    tsne_perplexity: 60
    # ... all pipeline parameters

- min_ha: 500
  max_ha: null    # null = no upper limit
  params:
    padding: 0.1
    embed_bands: '1,2,3,5,6,7,9,10,11,13,14,15'
    # ... different bands for larger fires
```

Admins can view and temporarily edit settings in the web UI. **Edits are in-memory only** -- to make permanent changes, edit the YAML file directly.

---

## Typical workflow

1. **Filter** the fire list (e.g., year 2023, minimum 10 ha).
2. **Select all** and click **Map Selected (with settings)** for a batch run.
3. Wait. The fire list auto-refreshes as each fire completes.
4. **Filter by status = mapped** to see fires needing review.
5. **Open** the first fire. The split view shows post-fire imagery alongside the ML classification.
6. If the result looks good, click **Accept** on the result card. If not, tweak parameters and re-map.
7. Click **Next** to move to the next fire.
8. Accepted results are in `<out_dir>/<FIRE_NUMBER>/` with full parameter records.

---

## Output structure

```
<out_dir>/
    fire_status.yaml              # Status index for all fires
    accepted_params.csv           # Parameter log (appended on each Accept)
    notes.yaml                    # Per-fire notes
    .web_cache/                   # Temporary working files (safe to delete)
        <FIRE_NUMBER>/
            *_crop.bin / .hdr     # Cropped raster
            *_perimeter.bin       # Rasterized perimeter
            VIIRS_*.bin           # Rasterized VIIRS hint
            *_classified.bin      # Classification output
            *_comparison.png      # Comparison figure
            previews/             # Preview PNGs for web display
    <FIRE_NUMBER>/                # Accepted results (one directory per fire)
        *_crop.bin / .hdr
        *_classified.bin
        *_comparison.png
        *_brush_comparison.png
        *_params.yaml             # Full parameter record + ML area
```

Accepted fire directories are compatible with `batch_fire_mapping` CLI output.

---

## Network access

The server binds to `0.0.0.0` by default and prints its network address on startup. Other machines on the same network can connect directly.

For encrypted access over untrusted networks, use an SSH tunnel:

```bash
ssh -L 8765:localhost:8765 user@remote-host
# Then open http://localhost:8765 locally
```

Or place the server behind a TLS-terminating reverse proxy.

---

## Known limitations

- **HDBSCAN non-determinism**: cuML's GPU HDBSCAN does not support a random seed. Identical runs may produce slightly different results due to GPU parallelism. All other stages are seeded and deterministic.
- **Single GPU queue**: one fire maps at a time. Additional requests queue automatically.
- **In-memory settings edits**: changes to recommended settings in the web UI are lost on restart. Edit the YAML file for permanent changes.
- **No TLS**: the server uses plaintext HTTP. Use an SSH tunnel or reverse proxy for encrypted access.

---

## File overview

| File | Purpose |
|---|---|
| `__main__.py` | Entry point. Parses arguments, loads data, initializes state, starts server. |
| `app.py` | Web server, all route handlers, mapping orchestration, template rendering. |
| `state.py` | Data classes for per-fire state (`FireInfo`) and global app state (`AppState`). |
| `preview.py` | ENVI header parsing, band detection, preview PNG generation. |
| `templates/fire_list.html` | Fire list page template. |
| `templates/fire_mapping.html` | Fire mapping page template (image viewer, parameters, console, results gallery). |
| `templates/login.html` | Login page template. |
| `templates/admin.html` | Admin dashboard template. |
| `templates/pending.html` | IP approval waiting page template. |
| `static/style.css` | All CSS styles. |
| `recommended_settings.yaml` | Default parameter presets by fire size range. |
