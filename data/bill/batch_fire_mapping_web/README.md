# batch_fire_mapping_web

*Last updated: April 20, 2026*

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
5. **Parameter Analyzer (admin-only)**: a dedicated high-throughput tool for exploring N parameter sets × M HDBSCAN replicates across selected fires. Admins accept one or more runs per fire; accepted parameters and fire characteristics are logged to a master CSV for offline analysis. All outputs live in a separate `analyzing_parameters/` directory and never touch the user-facing accepted fires.

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

Authentication is required by default. Set passwords via environment variables or CLI flags:

```bash
export FIRE_ADMIN_PASSWORD=<your-admin-password>
export FIRE_USER_PASSWORD=<your-user-password>

python -m batch_fire_mapping_web \
    fire_perimeters.shp sentinel2_stack.bin \
    --skip_download --out_dir ./results
```

To run without authentication (e.g. localhost-only testing), pass `--insecure_no_auth`:

```bash
python -m batch_fire_mapping_web \
    fire_perimeters.shp sentinel2_stack.bin \
    --skip_download --out_dir ./results --insecure_no_auth
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
| `--insecure_no_auth` | off | Run without any authentication. All users get full admin access. |
| `--trust_proxy` | off | Trust `X-Forwarded-For` header for client IP (use only behind a trusted reverse proxy). |

---

## Authentication and access control

Authentication is **required by default**. The server will refuse to start without passwords unless `--insecure_no_auth` is explicitly passed. When both an admin and user password are configured, the server enables a two-role system:

- **Admin**: full access. Can approve or block other users' IP addresses via the admin dashboard (`/admin`). Can edit recommended settings. Can trigger and cancel batch mapping. Can restore hidden fires.
- **User**: fire mapping access. Must have their IP approved by an admin before they can see any content.

New user IPs appear as "pending" on the admin dashboard. The admin approves or blocks them. Approved, blocked, and pending IP state all persist across server restarts (stored in `access_control.yaml`). Sessions are cookie-based and persist across restarts (stored in `sessions.yaml`).

Password rules:
- `--admin_password` alone enables admin-only mode (no user role).
- `--user_password` requires `--admin_password` to also be set (otherwise no one can approve IPs).
- Admin and user passwords must be different.

Login is rate-limited to 5 attempts per IP per 5-minute window.

With `--insecure_no_auth`, all authentication is bypassed and every user has full admin access. Use only for localhost-only testing.

---

## How the web UI works

**In-UI help.** Non-obvious concepts (N × M parameter sets, HDBSCAN jitter, padding, agreement %, status badges, the two "Map Fire" modes) carry a small grey "?" next to their label. Click it for a short explanation; click outside or press Escape to close. The tooltips are click-only — nothing pops up on hover — and are implemented as a shared primitive in `static/help.js` + the `.help` / `.help-popover` rules in `static/style.css`.

### Fire list (home page)

A sortable, filterable table of every fire in the shapefile.

- **Columns**: fire number, date, year, size (ha), agreement %, ML area, status.
- **Filters**: year, fire number (regex), minimum size, status. Filter state persists across page navigation, browser tabs, and browser restarts (stored in `localStorage`).
- **Batch actions**: select fires with checkboxes, then "Map Selected (with settings)" to batch-process them, or "Remove Selected" to hide them from the list. Running batches can be **cancelled** via the Cancel button.
- **Removing / restoring fires**: removed fires are hidden from the list and persist across restarts. Admins can restore hidden fires from the admin dashboard.
- **PDF report**: generates a downloadable PDF of all accepted fires.
- **Status badges**: pending (gray), ready (yellow), mapping (blue), mapped (blue), accepted (green), error (red).
- **Smart refresh**: the table auto-refreshes every 5 seconds but only re-renders when data has actually changed, preserving scroll position and avoiding visual flicker.

### Fire mapping page

Opening a fire takes you to the mapping page. On load, the server automatically:

1. Crops the raster to the fire's bounding box (with configurable padding).
2. Rasterizes the VIIRS hint or polygon perimeter.
3. Generates preview images (post-fire, pre-fire, difference).

#### Image viewer (left side)

- **Split view**: side-by-side comparison. Post-fire on the left, ML classification on the right (opens automatically when results are available).
- **Synced zoom/pan**: both panes move together by default. Zoom preserves your position when toggling split or collapsing the control panel -- you won't lose your place on large fires.
- **Pixel-perfect rendering**: `image-rendering: pixelated` at all zoom levels.
- **View dropdown**: switch between post-fire, pre-fire, difference, ML classification, hint perimeter, comparison figure, brush comparison.
- **Geospatial overlay alignment**: when re-cropping with different padding, previously accepted ML classification overlays are placed at the correct geographic position within the new crop using GDAL geotransforms, rather than being stretched to fit.

#### Parameters (right side)

Collapsible sections for every pipeline parameter:

- **Crop & Sampling**: padding, sample rate, min/max samples, seed.
- **T-SNE**: perplexity, learning rate, max iterations, init method, components, random state, embed bands.
- **Random Forest**: estimators, max depth, max features, random state.
- **HDBSCAN**: controlled ratio, min samples.
- **Display**: contour width.
- **Notes**: free-text annotations per fire (e.g., "cloud contamination"). Persisted immediately on change with visual save confirmation (green border flash).

#### Mapping

- **Map Fire**: runs the pipeline with the parameters currently in the form. If the padding value has changed since the last crop, the fire is automatically re-cropped first and all preview images update immediately before mapping begins.
- **Map Fire / with settings**: applies the recommended settings for the fire's size range, re-crops if padding changed, then maps.
- **Re-crop**: manually re-crops the fire with the current padding value and updates all preview images, without running the ML pipeline.
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

All mapping results live in `.web_cache/` until explicitly accepted. Nothing is written to the canonical output directory until you click Accept. Clicking Accept on a result card:
- Copies all outputs to the canonical output directory (`<out_dir>/<FIRE_NUMBER>/`).
- Writes a `_params.yaml` file with the full parameter record and ML area estimate.
- Logs parameters to `accepted_params.csv` (feeds the learning system). Re-accepting a fire replaces its previous CSV entry rather than appending a duplicate.
- Clears the results gallery and serial cache files.
- Shows a confirmation dialog if overwriting a previously accepted result.

Re-cropping or re-mapping a previously accepted fire does not affect the accepted results on disk. You can freely change padding and re-run with different parameters -- the accepted output is only overwritten when you accept a new result.

### Navigation

- **Prev / Next** buttons move through fires in the current filtered/sorted order without returning to the list. Works across browser tabs and bookmarked URLs (navigation order is stored in `localStorage`).
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

## Parameter Analyzer (admin-only)

The Parameter Analyzer is a dedicated high-throughput tool for systematically exploring how different parameter combinations perform across many fires. Where the user-facing serial mapping runs one fire at a time and accepts a single best result, the analyzer runs **N parameter sets × M HDBSCAN replicates** across many fires and lets the admin accept **multiple** runs per fire. The accepted parameters plus fire characteristics (region, zone, year, size bucket, crop dimensions, perimeter type) are logged to `analyzer_accepted.csv` for offline statistical analysis.

**Entry point**: admin dashboard → *Open params_analyzer*, or `/analyzer`.

### Why it exists

There is no unique best parameter set. The user-facing workflow accepts one parameter set per fire, so the learning loop ranks by agreement on a per-fire basis. The analyzer is the dataset-building counterpart: for a given fire, several parameter sets may produce acceptable results (e.g. 68% and 69% agreement), and both may be worth keeping. By recording every accept across many fires, the admin can analyze offline which parameter sets tend to work for which kinds of fires (region, size bucket, hint quality, …) rather than relying on the ranker's inference.

### Configuration

The `/analyzer` page lets the admin define:

- **Parameter sets (N)**: any number of sets. Each set contains a full set of pipeline parameters (padding, sample_rate, embed_bands, t-SNE and Random Forest settings, HDBSCAN controlled_ratio/min_samples). Leave a field blank to fall back to the pipeline default. Sets can be seeded from the recommended tiers for a quick start, then tweaked field-by-field.
- **HDBSCAN runs per set (M)**: how many replicate runs per set, 1-20.
- **HDBSCAN jitter step**: how much to vary `hdbscan_min_samples` across the M replicates (see below).
- **Fires to analyze**: multi-select via a rich filter panel (year, region, zone, size bucket, user status, analyzer status, accepted-runs toggle, fire-number regex) with live count preview. Filter state is persisted in `localStorage`.
- **Description**: free-text label for the config.

Config is persisted to `analyzing_parameters/analyzer_config.yaml` on save.

### HDBSCAN jitter (why replicates need it)

cuML's GPU HDBSCAN has no random seed. For some inputs it is effectively deterministic, which means M replicate runs with byte-identical HDBSCAN parameters produce byte-identical clusters and agreement scores -- useless for statistics. The jitter step automatically perturbs `hdbscan_min_samples` across replicates in a fan-out pattern:

| r | value |
|---|-------|
| 0 | base |
| 1 | base + 1×step |
| 2 | base − 1×step |
| 3 | base + 2×step |
| 4 | base − 2×step |
| … | … |

Because `hdbscan_min_samples` is **not** part of the t-SNE+RF cache signature, all M jittered replicates share one cached `.npz` and the full pipeline only runs once per (padding, signature) group. Jitter gives you statistical variation at zero speed cost. Set jitter to 0 to run true replicates (may yield identical scores on some GPUs).

### Grouping and caching

The worker plans the grid so runs sharing a padding and a t-SNE+RF signature execute contiguously:

```
For each (padding, tsne_rf_signature) group:
  Run 1 of the group: full pipeline (T-SNE + RF + HDBSCAN), --save_state -> cache.npz
  Runs 2..k of the group: --load_state cache.npz, HDBSCAN only
```

Example: 4 parameter sets × 3 replicates with jitter=1, where sets A and B share all non-HDBSCAN params (perplexity, bands, …):

- 4 sets × 3 = 12 runs total
- 3 unique (padding, tsne_rf_signature) groups → **only 3 full pipelines** + 9 HDBSCAN-only re-runs (≈ 10-20× faster than running the grid naively)

### Snapshots and isolation

Each distinct padding gets its own snapshot directory under `.analyzer_cache/<FIRE>/p_<padding>/`. The snapshot is produced by calling the user's `_prepare_fire_sync` under the GPU lock and immediately copying the crop, hint rasters, and preview PNGs into the snapshot. This means a user re-cropping the same fire at a different padding in the main UI cannot disturb analyzer work in progress -- the analyzer runs against its own crop, and writes its classification outputs into its own run subdir.

### Gallery and accepting

Open any fire (via the fires table or the "Currently analyzing" link in the status section) to see its runs gallery:

- Runs are grouped by parameter set, each card showing thumbnail, agreement %, ML area, actual params used (including the jittered `hdbscan_min_samples`).
- Click a thumbnail to open the full comparison figure in a modal viewer.
- **Accept** on any card moves that run's outputs into `analyzing_parameters/<FIRE>/run_XXXX/`, appends a row to `analyzer_accepted.csv`, and updates the composite-overlay backdrop if the accepted run has a larger padding than any previously accepted run.
- **Unaccept** reverses the accept, removes the row, and wipes the canonical fire directory if no accepts remain.
- Admins can accept multiple runs per fire, or zero. There is no "best run" constraint.

### Composite overlay viewer

Once at least one run is accepted, the per-fire page shows a composite overlay: the biggest-padded accepted crop rendered as the backdrop, with every accepted run's 1-pixel perimeter outline drawn in a distinct color. Outlines are aligned via geotransforms, so a run accepted at padding 0.1 correctly lines up with a backdrop that was saved at padding 0.3. The legend below the image maps colors to `run_XXXX` IDs, agreement scores, and padding values. The backdrop only ever grows -- unaccepting a run never shrinks it, because a bigger backdrop is harmless.

### CSV preview and download

The main analyzer page has two CSV outputs:

- **Accepted runs table**: the full CSV rendered as a sortable table on the page. Click any column header to sort. Toggle columns via the 35-box column picker. Summary chips above the table break down accepts by region / year / size bucket, plus agreement stats (min, median, mean, max). Auto-refreshes every 7 seconds.
- **Download full CSV**: link in the action bar returns `analyzer_accepted.csv` verbatim for offline pandas / Excel / BigQuery use.

The CSV schema is a superset of the user-facing `accepted_params.csv` -- overlapping parameter column names are identical, so the two files can be unioned offline for combined analysis. Extra analyzer columns: `accept_id`, `set_idx`, `run_idx`, `fire_region`, `fire_zone`, `size_bucket_lo`, `size_bucket_hi`, `perimeter_type`, `crop_w`, `crop_h`, `accepted_at`.

### Resumability and cancel

- Every completed run writes an `agreement.json` sidecar in its run subdir. On re-run, the worker skips any `(set_idx, run_idx)` whose sidecar exists.
- Grid cells with an accepted twin are also skipped (unaccept to re-run).
- **Cancel** sets an event flag; the worker finishes its current run, records any partial results, and stops. The fire's status becomes PARTIAL. Re-starting picks up where it left off.
- The analyzer releases the GPU lock **between** runs (not per-fire), so a regular user can squeeze in a normal mapping while the analyzer is running.
- Only one analyzer session can be running at a time (enforced in the start endpoint).

### Output directory

The analyzer writes everything under `<out_dir>/analyzing_parameters/`. You can delete the whole directory to start fresh without touching user-accepted fires in `<out_dir>/<FIRE_NUMBER>/` or the user-facing `accepted_params.csv`.

```
<out_dir>/analyzing_parameters/
    analyzer_config.yaml              # admin's current config (N sets, M, jitter, fires)
    analyzer_accepted.csv             # master CSV, one row per accepted run
    <FIRE>/                           # only exists after at least one accept
        <FIRE>_crop_max.bin / .hdr    # biggest-padded accepted crop (overlay backdrop)
        <FIRE>_post_max.png           # post-fire preview at max padding
        <FIRE>_overlay.png            # composite overlay (rebuilt on demand)
        <FIRE>_overlay_legend.json    # color -> accept_id legend sidecar
        manifest.yaml                 # {saved_padding, saved_crop, accepts: [run_0001, ...]}
        run_0001/                     # one accepted run
            classified.bin / .hdr
            comparison.png
            brush_comparison.png
            thumb.png
            params.yaml               # full provenance (fire + grid + params + outcome)
        run_0002/
            ...
    .analyzer_cache/<FIRE>/           # working cache (all N*M runs, accepted or not)
        p_0.2000/                     # snapshot per padding
            <FIRE>_crop.bin / .hdr
            <FIRE>_perimeter.bin
            VIIRS_*.bin               # only if viirs mode
            snapshot.json             # crop metadata
            previews/                 # post.png etc., for thumbnail generation
            tsne_rf_<sig>.npz         # cached T-SNE + RF intermediate state
            set_00_run_00/            # per-run outputs
                classified.bin / .hdr
                comparison.png
                thumb.png
                params.yaml
                agreement.json        # sidecar (resume marker)
            set_00_run_01/
                ...
        runs.yaml                     # fast-reload cache of in-memory AnalyzerRun list
```

### Differences from the user-facing workflow

| Aspect | User workflow | Analyzer |
|---|---|---|
| Who | any authenticated user | admin only |
| Accept semantics | one best result replaces prior accept | multiple accepts per fire coexist |
| Output dir | `<FIRE>/` alongside `.web_cache/<FIRE>/` | `analyzing_parameters/<FIRE>/` + `.analyzer_cache/<FIRE>/` |
| Param log | `accepted_params.csv` (deduplicated on re-accept) | `analyzer_accepted.csv` (append-only, keyed by `accept_id`) |
| Parameter selection | one set per run, recommended or learned | N×M grid with automatic HDBSCAN jitter |
| Purpose | map fires for production output | build parameter-performance dataset for offline analysis |

The two workflows are strictly isolated -- nothing the analyzer does affects the user-accepted `<FIRE>/` directories or `accepted_params.csv`, and vice versa.

---

## Recommended settings

The file `recommended_settings.yaml` defines an ordered list of parameter presets plus K (HDBSCAN replicates per setting). The first setting is the **primary** used by one-click "Map Fire". "Map Fire with Settings" runs every setting × K replicates. Loaded on startup from the output directory first, falling back to the package directory for defaults.

```yaml
k_runs_per_setting: 3
k_jitter: 1          # HDBSCAN fan-out step across K replicates

settings:
  - label: all-bands-tight
    params:
      padding: 0.10
      hdbscan_min_samples: 21
      tsne_perplexity: 60
      embed_bands: '1,2,3,4,5,6,7,8,9,10,11,12'
      # ... rest of pipeline parameters

  - label: all-bands-loose
    params:
      padding: 0.05
      hdbscan_min_samples: 19
      # ...

  - label: change-only
    params:
      embed_bands: '7,8,9,10,11,12'
      # ...
```

Admins edit the global list in the Recommended Settings panel on the fire list page. Per-fire overrides live in `fire_state.yaml` and are edited via the **Edit** button in each fire row — clearing the override falls back to the global list. The old size-bucket schema is rejected on startup; regenerate the file if you see a legacy-format error.

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
    fire_state.yaml               # Per-fire state (survives restart)
    fire_status.yaml              # Status index for all fires
    accepted_params.csv           # Parameter log (one row per accepted fire)
    notes.yaml                    # Per-fire notes
    recommended_settings.yaml     # User-edited settings (overrides package defaults)
    access_control.yaml           # Approved/blocked/pending IP addresses
    sessions.yaml                 # Persistent login sessions (hashed tokens)
    .web_cache/                   # Working files (preserved across re-prepares)
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
    analyzing_parameters/         # Parameter Analyzer outputs (admin only)
        analyzer_config.yaml
        analyzer_accepted.csv
        <FIRE_NUMBER>/            # Only exists after at least one analyzer accept
            <FIRE>_crop_max.bin   # Biggest-padded accepted crop (overlay backdrop)
            <FIRE>_post_max.png
            <FIRE>_overlay.png    # Composite overlay (rebuilt on demand)
            manifest.yaml
            run_0001/             # One accepted run
            run_0002/
        .analyzer_cache/          # Working cache, delete-safe
            <FIRE_NUMBER>/
                p_<padding>/      # One snapshot per padding
                    tsne_rf_<sig>.npz    # Cached intermediate state
                    set_<S>_run_<R>/     # Per-run outputs + sidecar
```

Accepted fire directories are compatible with `batch_fire_mapping` CLI output. The `analyzing_parameters/` tree is self-contained and can be deleted independently -- see the Parameter Analyzer section for full details.

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

## Security hardening

The following security measures are built into the server:

- **Session tokens** are hashed (SHA-256) before storage on disk. Raw tokens exist only in browser cookies. Cookies are always set with `HttpOnly` and `SameSite=Lax`. The `Secure` flag is added only when the request came in over HTTPS (detected via `X-Forwarded-Proto: https` when `--trust_proxy` is enabled). Over plain HTTP, `Secure` is omitted so cookies survive on non-localhost addresses (e.g. a LAN IP reached over a VPN) -- browsers silently drop `Secure` cookies on plain-HTTP non-localhost hosts, which would otherwise cause an endless redirect back to `/login`.
- **CSRF protection** on all POST endpoints via Origin header validation and `X-Requested-With` header requirement.
- **Login rate limiting**: 5 failed attempts per IP per 5-minute window.
- **IP normalization**: IPv6-mapped IPv4 addresses (e.g. `::ffff:10.0.0.1`) are normalized to plain IPv4, preventing bypass via address format tricks.
- **Path traversal prevention**: static file serving, report generation, and fire name handling all validate paths with `os.path.realpath` containment checks.
- **Parameter validation**: all pipeline parameters are validated against typed bounds before being passed to subprocesses, preventing resource exhaustion via extreme values.
- **Atomic file writes**: YAML persistence files (sessions, IP access lists, fire state) are written atomically (write to temp file, then `os.replace`) with `0600` permissions to prevent data loss on crash and limit read access.
- **Request body limits**: POST bodies are capped at 1 MB; oversized requests receive a 413 response.
- **Blocked IP enforcement**: blocked IPs are checked before any role-based auto-approve logic, so a blocked IP cannot bypass the block by logging in as admin.
- **Batch mapping** requires admin role.
- **Thread safety**: mutable shared state (sessions, console logs, serial results, GPU queue, batch status, IP access lists) is protected by locks to prevent race conditions. Mutations and reads are performed under `state.lock`; persistence save-paths (`_save_sessions`, `_save_ip_list`, `_save_notes`, `_save_fire_state`) snapshot state under the lock before writing to disk.
- **Atomic fire-status updates**: status and `error_msg` are updated together via `_set_fire_status()` under the lock, so readers never observe an ERROR status paired with a stale error message.
- **Analyzer CSV serialization**: appends and removals to `analyzer_accepted.csv` are serialized via a dedicated module-level lock to prevent read-modify-write races.
- **Fixed CSV headers**: `accepted_params.csv` uses a fixed fieldname list, preventing header drift across appends. Re-accepting a fire replaces its previous entry rather than appending duplicates.
- **Structured error logging**: persistence failures (sessions, notes, fire state, settings) are logged to stderr with context rather than silently swallowed.
- **URL encoding**: all client-side API URLs encode fire names with `encodeURIComponent` to handle special characters safely.
- **Template escaping**: the client-side `esc()` helper escapes `<`, `>`, `&`, `"`, and `'`, making interpolated values safe in both text and attribute contexts.
- **Bounded memory**:
  - Per-fire console log buffers are capped at 2000 lines via `collections.deque(maxlen=...)`, preventing unbounded memory growth on long-running mappings.
  - The in-memory login-attempt rate-limit table is bounded: empty entries are popped on access and a sweep removes stale entries when the dict exceeds 1024 IPs.
- **Subprocess watchdog**: mapping and analyzer CLI subprocesses are killed after 30 minutes of silent stdout to prevent a hung run from wedging the GPU lock indefinitely.
- **SSE disconnect kills CLI**: when a client disconnects from a mapping stream, the broken-pipe exception propagates through the SSE writer to terminate the running subprocess rather than orphaning it. A guard in the mapping handler's `finally` also resets any fire stuck in `MAPPING` back to `READY`.
- **Session expiry sweep**: stale (expired) sessions are swept opportunistically on each successful login, in addition to the per-request lazy check.
- **Input hardening**: malformed or negative `Content-Length` headers are rejected with a 400 response instead of crashing or hanging the request. The same guard is applied to the login form parser.
- **Admin-only endpoints**: `/remove`, `/unhide`, `/fires/hidden`, and `/batch/cancel` require the admin role, in addition to the existing admin gates on batch mapping and the analyzer.

---

## Persistence and crash recovery

The server persists all important state to disk so that work survives restarts and crashes:

| File | What it stores | When it's written |
|---|---|---|
| `fire_state.yaml` | Per-fire status, cache paths, parameters, agreement, hidden flags | After every prepare, mapping, accept, and remove/restore |
| `notes.yaml` | Per-fire text annotations | On every note edit |
| `accepted_params.csv` | One row per accepted fire (deduplicated on re-accept) | On accept |
| `access_control.yaml` | Approved, blocked, and pending IPs | On every IP action |
| `sessions.yaml` | Hashed session tokens | On login/logout |
| `recommended_settings.yaml` | User-edited parameter presets (in output dir) | On admin save |
| `analyzing_parameters/analyzer_config.yaml` | Analyzer config (N sets, M, jitter, fires) | On analyzer save |
| `analyzing_parameters/analyzer_accepted.csv` | One row per accepted analyzer run (append-only) | On analyzer accept |
| `analyzing_parameters/<FIRE>/manifest.yaml` | Per-fire accept list + max-crop tracking | On analyzer accept/unaccept |
| `.analyzer_cache/<FIRE>/runs.yaml` | Fast-reload cache of the in-memory runs list | After every run completes |

On startup, the server:
1. Loads fires from the shapefile and detects accepted fires by checking for `_comparison.png` on disk.
2. Restores fire state from `fire_state.yaml` — mapped fires, cache paths, hidden flags, parameters, and agreement scores are all recovered.
3. Validates that cached files still exist before restoring status. If a fire was mid-mapping when the server crashed, it recovers to MAPPED (if results exist in cache) or READY (if not), rather than being stuck in MAPPING.
4. Initializes the Parameter Analyzer: loads `analyzer_config.yaml`, scans `analyzing_parameters/<FIRE>/` directories to reconstruct accepted-run state, and scans `.analyzer_cache/<FIRE>/` for un-accepted (pending or partial) run sidecars. Fires that had analyzer data on disk return to ANALYZED or PARTIAL status accordingly.

The `.web_cache/` directory is no longer wiped on every re-prepare. Cache is only cleared when padding actually changes (which invalidates the crop dimensions). This means un-accepted mapping results survive across page reloads and re-prepares with the same padding.

The `.analyzer_cache/` directory is never wiped automatically — un-accepted analyzer runs survive across restarts so the admin can resume or reaccept without re-running the grid. Delete the whole `analyzing_parameters/` tree to start fresh; nothing in the user workflow depends on it.

### Crash-safety details

- `accepted_params.csv` and `analyzer_accepted.csv` are written via tmp-file + `os.replace`. A crash or disk-full mid-write leaves the previous file intact rather than a truncated one. Re-accepting the same fire (user CSV) still dedupes — the read/filter/write happens inside one atomic replace.
- All YAML persistence goes through `_atomic_yaml_dump` (tmp + `os.replace`, `0o600` for sessions/IP lists).
- Analyzer accepts claim the target `accept_id` and set `run.accepted = True` under `astate.lock` before any file copies, so two concurrent admin clicks on the same run cannot both succeed. If the subsequent copy fails, the claim is rolled back.
- On startup, `accept_id` values loaded from `analyzing_parameters/<FIRE>/manifest.yaml` are validated against `run_NNNN` before being joined into paths. A corrupted manifest is logged and the malformed entries are dropped.
- The GPU queue's `waiting_jobs` append is wrapped in the same `try/finally` that releases it, so a client disconnect while the "queued" log line is being streamed cannot leak a phantom entry.
- `fire_mapping_cli.py` is launched with `start_new_session=True`. The silence watchdog (and the cleanup path) signal the whole process group (SIGTERM → SIGKILL), so helper grandchildren the CLI spawns via `subprocess.run` (e.g. `gdal_translate`, `qgis`) do not orphan to init when the watchdog fires.
- `handle_api_batch_map` wraps the body read + validation in `try/finally`: a client disconnect after headers but before the JSON body arrives no longer leaves `state.batch_status = {'running': True}` stuck, which would previously block every subsequent batch until a server restart.
- Accepting a serial run mid-batch (user clicks Accept on run 2 while runs 3–5 are still queued) sets `fire.serial_canceled = True`. The worker polls this between replicates, between settings, and inside the GPU-locked status write just before starting each subprocess, then skips its final "pick best + overwrite main cache files + set status=MAPPED" block on exit. Previously, the worker would race past the accept, overwrite the ACCEPTED status with MAPPING→MAPPED, overwrite the cache comparison with the best of the non-accepted runs, and persist the wrong state to `fire_state.yaml` — so the accepted run effectively vanished from the UI (and `previously_accepted` stayed `False` across restarts, so a subsequent re-map wouldn't show the accepted result as its Run 0 snapshot).

---

## Known limitations

- **HDBSCAN non-determinism**: cuML's GPU HDBSCAN does not support a random seed. Identical runs *may* produce slightly different results due to GPU parallelism, but for some inputs the GPU is effectively deterministic. All other pipeline stages are seeded and deterministic. The Parameter Analyzer works around this by jittering `hdbscan_min_samples` across replicates (see the jitter section above).
- **Single GPU queue**: one fire maps at a time. Additional requests queue automatically. The analyzer releases the GPU lock between runs so user mappings can interleave.
- **No TLS**: the server uses plaintext HTTP. Use an SSH tunnel or reverse proxy for encrypted access. Cookies omit the `Secure` flag on plain HTTP so that login works over VPN / LAN IPs (see the security section); `Secure` is added back automatically when running behind an HTTPS-terminating proxy with `--trust_proxy`.
- **Login rate limiting is in-memory**: the 5-attempt rate limit resets on restart. The table is size-bounded (see Security hardening) but not persisted. All other security state persists.
- **Analyzer disk usage grows with grid size**: every completed run stores a classification `.bin`, comparison PNG, and thumbnail (typically ~5-50 MB per run). A grid of 2500 runs can consume many GB. Delete the `.analyzer_cache/<FIRE>/` tree for a fire once you have accepted everything you want; the accepted results live in `analyzing_parameters/<FIRE>/run_XXXX/` and are unaffected.
- **Analyzer concurrency**: only one analyzer session runs at a time (enforced server-side). Only admins can start one.

---

## File overview

| File | Purpose |
|---|---|
| `__main__.py` | Entry point. Parses arguments, loads data, initializes state, starts server, registers analyzer routes. |
| `app.py` | Web server, all route handlers, mapping orchestration, template rendering. |
| `state.py` | Data classes for per-fire state (`FireInfo`) and global app state (`AppState`). |
| `preview.py` | ENVI header parsing, band detection, preview PNG generation. |
| `analyzer_state.py` | Analyzer data classes (`AnalyzerRun`, `AnalyzerFireInfo`, `AnalyzerConfig`, `AnalyzerState`), the 35-column CSV schema, and the list of parameter keys that invalidate the t-SNE+RF cache. |
| `analyzer_app.py` | Analyzer route handlers and admin gate. Registers its routes on `FireHandler` via monkey-patching so `app.py` stays untouched. Config read/write, fires list, status, per-fire gallery, run images, accept/unaccept, CSV preview/download, composite overlay. |
| `analyzer_worker.py` | Grid planner (padding-grouped → signature-grouped → run_idx), HDBSCAN jitter, snapshot preparer, per-run subprocess launcher, sidecar-based resume, cancel event, startup cache scan. |
| `analyzer_accept.py` | Accept/unaccept logic: promote a cached run to `analyzing_parameters/<FIRE>/run_XXXX/`, append / remove CSV row, maintain `manifest.yaml`, grow `<FIRE>_crop_max.bin` backdrop only when accepted padding exceeds saved padding. |
| `templates/fire_list.html` | Fire list page template. |
| `templates/fire_mapping.html` | Fire mapping page template (image viewer, parameters, console, results gallery). |
| `templates/login.html` | Login page template. |
| `templates/admin.html` | Admin dashboard template. Entry point to the analyzer. |
| `templates/pending.html` | IP approval waiting page template. |
| `templates/analyzer.html` | Analyzer main page: status, param-set editor, fire selector with rich filters, accepted-runs CSV preview table. |
| `templates/analyzer_fire.html` | Per-fire analyzer gallery: composite overlay viewer, worker console, runs grouped by set with accept/unaccept. |
| `static/style.css` | All CSS styles. |
| `recommended_settings.yaml` | Default parameter presets by fire size range. |
