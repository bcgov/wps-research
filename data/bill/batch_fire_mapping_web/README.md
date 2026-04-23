# batch_fire_mapping_web

*Last updated: April 23, 2026*

Interactive web interface for mapping wildfire burn areas from Sentinel-2 satellite imagery. Uses a machine learning pipeline (T-SNE dimensionality reduction, Random Forest classification, HDBSCAN clustering) accelerated on GPU to classify burned vs. unburned pixels, then lets users visually review and accept results through a browser.

This is the web companion to the `batch_fire_mapping` CLI. It wraps the same underlying pipeline but replaces the sequential batch workflow with an interactive one: users can inspect each fire, tune parameters, compare results side-by-side, and build up a parameter knowledge base over time.

**Multi-year aware**: one launch accepts many Sentinel-2 rasters (one per year). The server auto-detects each raster's year from its filename, picks a default "active year", and lets admins swap to another year from the fire-list filter panel without restarting the process.

---

## What this tool does

1. **Loads** a shapefile of historical fire perimeters and one Sentinel-2 raster **per year** (e.g. `pgfc_2022.bin`, `pgfc_2023.bin`).
2. **Optionally downloads** VIIRS active fire satellite detections per year to use as classification hints.
3. **Serves a web UI** where users can:
   - Pick the **active year** from a single-select radio in the filter panel (admin-only). The active year determines which raster and which per-year output directory are in effect.
   - Browse all fires whose `FIRE_YEAR` matches the active year **and** whose polygon intersects that year's raster footprint, sorted and filtered by number, size, status, etc.
   - Open any fire to see post-fire, pre-fire, and difference imagery at full pixel resolution.
   - Run the ML classification pipeline with adjustable parameters.
   - Run **serial mapping** (multiple parameter sets at once) and compare results in a gallery.
   - Accept the best result, which saves it to that year's per-year output directory and logs the parameters for future use.
   - Batch-map many fires at once using recommended settings.
4. **Learns over time**: accepted parameters are logged. Future serial mappings rank parameter sets by how well they performed on similar fires (same region, similar size), so results improve as more fires are processed.
5. **Parameter Analyzer (admin-only)**: a dedicated high-throughput tool for exploring N parameter sets × M HDBSCAN replicates across selected fires (scoped to the active year). Admins accept one or more runs per fire; accepted parameters and fire characteristics are logged to a master CSV for offline analysis. All outputs live in a separate `analyzing_parameters/` directory under the active year's outdir and never touch the user-facing accepted fires.

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
    --rasters  /path/to/pgfc_2022.bin \
               /path/to/pgfc_2023.bin \
               /path/to/pgfc_2024.bin \
    --out_root ./mapping_results \
    --skip_download
```

Then open `http://localhost:8765` in your browser. The server picks the newest year (2024 in this example) as the initial active year; an admin can switch to another year from the fire-list filter panel.

Each filename must contain a unique, plausible 4-digit year (1970 ≤ year ≤ current year + 1). Digit position is not hardcoded — `pgfc_2023.bin`, `2023_pgfc.bin`, and `S2C_MSIL1C_20251014T192401_20m.bin` are all valid. Two rasters with the same detected year are rejected at startup.

Per-year outputs live under `<out_root>/<raster_stem>_mapping_results/` (e.g. `./mapping_results/pgfc_2023_mapping_results/`). Shared state — sessions, IP access list, recommended settings, active-year pointer — lives at `<out_root>/` and is preserved across year switches.

Authentication is required by default. Set passwords via environment variables or CLI flags:

```bash
export FIRE_ADMIN_PASSWORD=<your-admin-password>
export FIRE_USER_PASSWORD=<your-user-password>

python -m batch_fire_mapping_web \
    fire_perimeters.shp \
    --rasters  pgfc_2022.bin  pgfc_2023.bin  pgfc_2024.bin \
    --out_root ./mapping_results --skip_download
```

To run without authentication (e.g. localhost-only testing), pass `--insecure_no_auth`:

```bash
python -m batch_fire_mapping_web \
    fire_perimeters.shp \
    --rasters  pgfc_2023.bin \
    --out_root ./mapping_results --skip_download --insecure_no_auth
```

> **Tip**: Use environment variables instead of CLI flags on shared systems, since CLI arguments are visible in process listings.

> **Bash tip**: pass the raster list with an unquoted expansion or a bash array so argparse sees multiple words, not one long string: `--rasters "${RASTERS[@]}"` is safe; `--rasters "$RASTERS"` is not.

---

## Command reference

```
python -m batch_fire_mapping_web  POLYGON_FILE  --rasters R1 [R2 …]  --out_root DIR  [options]
```

### Required arguments

| Argument | Description |
|---|---|
| `POLYGON_FILE` | Fire perimeters shapefile (`.shp`, positional). Must have columns: `FIRE_NUMBE`, `FIRE_DATE`, `FIRE_YEAR`, `FIRE_SIZE_`. One shapefile is shared across all years; per-year polygons are selected by `FIRE_YEAR`. |
| `--rasters R1 [R2 …]` | One or more Sentinel-2 ENVI rasters (`.bin` with companion `.hdr`). Each filename must contain a unique 4-digit year; the year is auto-detected. Duplicate years are rejected. |
| `--out_root DIR` | Mother directory. Per-year outputs go to `<out_root>/<raster_stem>_mapping_results/`. Shared state (sessions, IP list, recommended settings, active-year pointer) lives directly under `<out_root>/`. |

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--year N` | newest year, or value saved in `<out_root>/active_year.yaml` | Initial active year at startup. Must match one of the years detected from `--rasters`. |
| `--perimeter_mode {viirs,traditional}` | `viirs` | Hint source. `viirs` uses VIIRS active fire data when available, falls back to polygon perimeter. `traditional` always uses the polygon and skips VIIRS entirely. |
| `--skip_download` | off | Skip VIIRS download + shapify (use when data already exists from a previous run). |
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

## Multi-year support

A single server process manages **N rasters for N years** under one shared `<out_root>`. Only one year is "active" at a time; the active year determines which raster, which per-year output directory, and which set of fires the UI shows.

### Year detection

Each raster's year is extracted from its filename by scanning **all** 4-digit substrings and keeping those inside `[1970 … currentYear+1]`. The stem must yield **exactly one** plausible year:

| Filename | Detected year |
|---|---|
| `pgfc_2023.bin` | 2023 |
| `2024_pgfc.bin` | 2024 |
| `S2_2023_v2.bin` | 2023 |
| `S2C_MSIL1C_20251014T192401_20m.bin` | 2025 |
| `foo.bin` | ERROR: no plausible year |
| `a_1999_b_2001.bin` | ERROR: multiple plausible years |
| `pgfc_2023_2024.bin` | ERROR: multiple plausible years |

Duplicate years across `--rasters` also error at startup. This guarantees the `{year → raster}` map is injective.

### Active year

- The active year is set from (in order of precedence): `--year N` CLI arg → the value in `<out_root>/active_year.yaml` → the newest year in the registry.
- It is persisted to `<out_root>/active_year.yaml` on every successful switch, so restarts remember.
- Fire list scope: `state.fires` only contains polygons whose `FIRE_YEAR` column equals the active year **and** whose geometry intersects the active raster's extent. Fires from other years that happen to sit inside the raster footprint do not leak in.

### Switching years (admin-only)

The filter panel's "Year" section is a set of **radio buttons** (single-select) covering every year passed via `--rasters`. The active year is pre-checked.

- Pick a different year → click **Apply** → confirm the prompt → the server swaps state in place and the page reloads.
- Pick the same year → click **Apply** → plain client-side refilter, no backend call.
- Non-admin roles see the radios disabled with a `(year switch is admin-only)` note.

The backend endpoint is `POST /api/year/switch {year: N}`. It refuses if any long-running job is in flight — a one-shot mapping, a batch, or an analyzer session — returning `409` with a message like `A mapping job is running (fire XXX). Wait for it to finish.` Finish or cancel the job, then retry.

A successful switch runs synchronously under `state.lock`: re-reads raster info for the new year, re-projects the cached raw polygon GDF and spatial-/FIRE_YEAR-filters it, reloads VIIRS, rebuilds `state.fires` via `init_fires_from_gdf`, calls `_load_fire_state()` against the new outdir's `fire_state.yaml`, and re-initializes the Parameter Analyzer against the new `analyzing_parameters/` directory. Typical cost is well under a second with warm caches.

### Shared vs. per-year files

| Path | Scope | Contents |
|---|---|---|
| `<out_root>/active_year.yaml` | shared | The currently-active year. |
| `<out_root>/sessions.yaml` | shared | Hashed session tokens. Survives year switches so logins carry across. |
| `<out_root>/access_control.yaml` | shared | Approved / blocked / pending IP list. |
| `<out_root>/recommended_settings.yaml` | shared | User-edited settings (with fallback to the package default). |
| `<out_root>/<raster_stem>_mapping_results/` | per-year | Everything else: `fire_state.yaml`, canonical `<FIRE>/` directories, `.web_cache/`, `accepted_params.csv`, `notes.yaml`, `fire_status.yaml`, `analyzing_parameters/`. |

---

## Authentication and access control

Authentication is **required by default**. The server will refuse to start without passwords unless `--insecure_no_auth` is explicitly passed. When both an admin and user password are configured, the server enables a two-role system:

- **Admin**: full access. Can approve or block other users' IP addresses via the admin dashboard (`/admin`). Can edit recommended settings. Can trigger and cancel batch mapping. Can restore hidden fires.
- **User**: fire mapping access. Must have their IP approved by an admin before they can see any content.

New user IPs appear as "pending" on the admin dashboard. The admin approves or blocks them. Approved, blocked, and pending IP state all persist across server restarts (stored in `<out_root>/access_control.yaml`). Sessions are cookie-based and persist across restarts and year switches (stored in `<out_root>/sessions.yaml`).

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

A sortable, filterable table of every fire for the currently-active year.

- **Columns**: fire number, date, year, size (ha), agreement %, ML area, status.
- **Filters**: year (single-select radio, doubles as admin-only year switcher — see *Multi-year support*), fire number (regex), minimum size, status. Non-year filter state persists across page navigation, browser tabs, and browser restarts (stored in `localStorage`). The year selection itself is **not** persisted client-side — it's the server-side active year, kept in `<out_root>/active_year.yaml`.
- **Year filter preview**: when you pick a year different from the active one, the preview count panel shows `Apply will switch active year to YYYY (current year: N / M matches)` so you know Apply will trigger a raster swap rather than a pure client-side refilter.
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
- **View dropdown**: switch between post-fire, pre-fire, difference, ML classification, hint perimeter, comparison figure, **brush comparison** (side-by-side PNG showing raw HDBSCAN output vs. brushed output — regenerated on every rebrush, making it the quickest way to see whether a brush-parameter change actually did what you wanted).
- **Geospatial overlay alignment**: when re-cropping with different padding, previously accepted ML classification overlays are placed at the correct geographic position within the new crop using GDAL geotransforms, rather than being stretched to fit. The same GeoTransform-based alignment is now also used inside `_compute_agreement`: when a serial run's `classified.bin` and the current `fire.hint_bin` have different extents (e.g. a rebrush on a run whose padding differs from the current hint, or a recommended-settings sweep that spans multiple paddings), IoU is computed over the common overlap rectangle rather than collapsing to `-1`. Without this, every cross-padding rebrush dropped agreement to `-1` and the Accept button disappeared from the gallery card.

#### Parameters (right side)

Collapsible sections for every pipeline parameter:

- **Crop & Sampling**: padding, sample rate, min/max samples, seed.
- **T-SNE**: perplexity, learning rate, max iterations, init method, components, random state, embed bands.
- **Random Forest**: estimators, max depth, max features, random state.
- **HDBSCAN**: controlled ratio, min samples.
- **Display / brush**: `contour_width`, plus the brush post-processing knobs that control how `class_brush.exe` cleans the raw HDBSCAN mask:
  - `brush_size` (int, default 15) — radius, in pixels, of the morphological brush used to close gaps and smooth edges. Larger values produce fatter, more forgiving perimeters; smaller values hug the raw classification more tightly.
  - `point_threshold` (int, default 10) — minimum connected-component size, in pixels. Components smaller than this are dropped as speckle.
  - `brush_all_segments` (bool, default false) — when true, the brush is applied to every connected component independently; when false, the brush operates on the dominant component only. Useful for fires with multiple disjoint burn patches.
  These three parameters are also consumed by the standalone **Rebrush** button (see *Rebrushing without re-mapping* below), which re-runs only the brush stage against the cached pre-brush raster.
- **Notes**: free-text annotations per fire (e.g., "cloud contamination"). Persisted immediately on change with visual save confirmation (green border flash).

#### Mapping

- **Map Fire**: runs the pipeline using exactly the parameters currently in the right-hand panel. The panel is snapshotted into a single custom setting (labelled `Current panel` in the gallery) and sent to the serial worker, so whatever you see in the form is what runs — including any bespoke `embed_bands` string, custom `padding`, or tuned HDBSCAN values. If the padding value has changed since the last crop, the fire is automatically re-cropped first and all preview images update immediately before mapping begins. This is the default "single custom run" button; use it when you want full control over one parameter set.
  > Historical note: earlier versions of the UI silently fell back to the first entry of `recommended_settings.yaml` when you clicked Map Fire, which meant panel edits (like switching `embed_bands` to `4,…,12`) were ignored and the run was mislabelled. That bug was fixed on 2026-04-21 — the panel is now authoritative for this button.
- **Map Fire / with settings**: ignores the form and runs **every** entry in the recommended settings list × `k_runs_per_setting` HDBSCAN replicates (with `k_jitter` applied to `hdbscan_min_samples`). Use this for a broader sweep when you don't yet know which parameter family works for this fire. Padding is honored per-setting, and grouped cache reuse means parameter sets sharing a `(padding, tsne_rf_signature)` key only run the expensive stages once — see *Serial mapping* below for details.
- **Re-crop**: manually re-crops the fire with the current padding value and updates all preview images, without running the ML pipeline.
- **Runs (N)**: legacy control retained for compatibility; the recommended-settings sweep uses `k_runs_per_setting` from the YAML instead. Set N > 1 to run multiple mappings with varied HDBSCAN parameters. All runs appear as cards in a results gallery, ranked by agreement score.
- **Console**: streams pipeline output in real time. Persists across page navigation.

#### Rebrushing without re-mapping

The brush stage (morphological smoothing + speckle removal) is the cheapest part of the pipeline and also the one users most often want to tweak after seeing the classification. **Rebrush only** re-runs this stage against the pre-brush raster without touching T-SNE or Random Forest, so iterating on `brush_size` / `point_threshold` / `brush_all_segments` costs seconds instead of minutes.

- **Rebrush only** button (next to Map Fire): takes the three brush parameters from the panel and re-runs `class_brush.exe` against the fire's cached classification. The ML classification overlay, agreement %, ML area, and brush-comparison figure all refresh in place. No T-SNE or Random Forest cost is incurred.
- **Per-run rebrush**: each card in the results gallery has its own Rebrush button. Clicking it rebrushes that specific serial run's `classified.bin`, regenerates its thumbnail and brush-comparison PNG, and updates the card's agreement % / ML area stats in the gallery (and in `fire.serial_results` on the server). The main classification on the left pane is **not** affected by a per-run rebrush — only the card. Use this when you want to compare different brush settings across serial results side-by-side.
- **Cancel** button: appears next to the Rebrush button while a rebrush is running. Posts to `/api/fire/<FIRE_NUMBER>/rebrush/cancel`, which SIGTERMs the running `class_brush.exe`. A cancelled rebrush returns `status: cancelled` and leaves the canonical classification untouched.
- **`_raw.bin` backup**: the first time a classification is rebrushed, the server copies the pre-brush raster to `<classified>_raw.bin` (plus a matching `.hdr`) alongside the classified file. Every subsequent rebrush reads from `_raw.bin` as its input, so repeated rebrushes are always applied to the original raw HDBSCAN output, not to an already-smoothed mask. This means `brush_size = 3` followed by `brush_size = 30` does not compound; each run starts clean. Older fires without a `_raw.bin` sibling fall back to using the current `classified.bin` as input (documented in `handle_api_rebrush`) — this is a one-time edge case; the first rebrush creates the backup for future runs.
- **Concurrency**: only one rebrush per fire at a time. Overlapping requests return HTTP 409 with `{"error": "A rebrush is already running for this fire"}`. Rebrushing is also disabled while the fire is in `MAPPING` status (409 with `Mapping in progress; rebrush disabled`).
- **State recovery**: the `/api/fire/<FIRE>/console` endpoint returns a `rebrush_running` boolean so the UI can *adopt* a rebrush that started before a page reload. If you kick off a rebrush and then refresh the page, the UI polls `/console`, sees `rebrush_running: true`, and restores the "Brushing…" button state + Cancel button without the user having to know it was in flight. Once the rebrush ends, the polled state flips to false and the UI releases the adopt lock.
- **Backend**: `_run_class_brush_only` (in `app.py`) launches `class_brush.exe` via `subprocess.Popen` with `stdout=PIPE`, then uses `proc.communicate()` (not `proc.wait()`) to drain the pipe concurrently on an internal thread. `class_brush.exe` emits one status line per connected component — on complex fires the pipe buffer easily overflows 64 KB, which would deadlock a naive `wait()`. The running subprocess is registered in `_rebrush_procs[fire_numbe]` so the cancel endpoint can SIGTERM it; the registry is single-slot per fire and cleared in a `finally` to guarantee release even on exceptions.

#### Results gallery

Every mapping (even a single run) produces a result card showing:

- Thumbnail of the ML classification overlay (cache-busted with a `?t=<timestamp>` query param, so rebrushed cards update visually without a hard reload).
- Agreement score (IoU between ML result and hint perimeter).
- ML fire area estimate in hectares.
- Expandable parameter details.
- **Accept** button to save that result.
- **Rebrush** button — re-runs the brush stage on just this serial run using the current panel brush parameters. Per-card stats update in place when the rebrush finishes. See *Rebrushing without re-mapping* above.

When re-mapping a previously accepted fire, the old result appears as a "Previously accepted" card (gold border) so you can compare old vs. new before deciding.

Cards are re-rendered whenever their backing stats change. Each card writes a `statSig` (agreement %, ML area, error flag, is-best flag) into `card.dataset.statSig`; on every poll, `showSerialGallery` compares the new signature against the stored one and only rebuilds the DOM node when something actually changed. This means rebrushes that bump `ml_area_ha` or `agreement_pct` propagate to the gallery even when the card count is unchanged — a regression fixed on 2026-04-21 where a length-only short-circuit had been suppressing stat updates after rebrush.

#### Accepting

All mapping results live in the active year's `.web_cache/` until explicitly accepted. Nothing is written to the canonical output directory until you click Accept. Clicking Accept on a result card:
- Copies all outputs to the active year's canonical output directory (`<out_root>/<raster_stem>_mapping_results/<FIRE_NUMBER>/`).
- Writes a `_params.yaml` file with the full parameter record and ML area estimate.
- Logs parameters to that year's `accepted_params.csv` (feeds the learning system). Re-accepting a fire replaces its previous CSV entry rather than appending a duplicate.
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

**Cross-padding sweeps.** When the recommended-settings list contains entries with different `padding` values, the worker re-enters `_prepare_fire_sync` between settings to re-crop. The padding-change cleanup is intentionally selective: per-run serial artifacts from earlier settings are preserved, so their gallery cards stay live with working thumbnails, Accept buttons, and agreement scores — even though their `classified.bin` was cropped at a different extent than the current main hint. Agreement for those cards is recomputed via GeoTransform alignment on the overlap rectangle (see *Geospatial overlay alignment* above), not by shape-matching, so a sweep like `[pad=0.05, pad=0.1, pad=0.2]` produces a gallery with three comparable cards instead of two ghosts and one usable card.

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
- **Fires to analyze**: multi-select via a rich filter panel (year, region, zone, size bucket, user status, analyzer status, accepted-runs toggle, fire-number regex) with live count preview. Filter state is persisted in `localStorage`. The candidate pool is scoped to the **active year** — switching years in the main UI reinitializes the analyzer against the new year's outdir. If you need to analyze 2023 and 2024 separately, switch years between runs.
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

The analyzer is scoped to the active year: it writes everything under `<out_root>/<active_raster_stem>_mapping_results/analyzing_parameters/`. Switching years reinitializes the analyzer against the new year's directory. You can delete the whole `analyzing_parameters/` tree to start fresh without touching user-accepted fires in `<FIRE_NUMBER>/` or the year's `accepted_params.csv`.

```
<out_root>/<active_raster_stem>_mapping_results/analyzing_parameters/
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

The file `recommended_settings.yaml` defines an ordered list of parameter presets plus K (HDBSCAN replicates per setting). The first setting is the **primary** used by one-click "Map Fire". "Map Fire with Settings" runs every setting × K replicates. Loaded on startup from `<out_root>/recommended_settings.yaml` first, falling back to the package-shipped default if the shared copy is missing. The recommended settings list is **shared across years** (not per-year): a year switch preserves whatever the admin last saved.

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

1. **Pick the active year** in the filter panel (single-select radio, admin-only). This loads that year's raster and filters the fire list to fires whose `FIRE_YEAR` matches. Skip if you only launched with one raster — its year is already active.
2. **Apply the other filters** (minimum size, fire-number regex, status).
3. **Select all** and click **Map Selected (with settings)** for a batch run.
4. Wait. The fire list auto-refreshes as each fire completes.
5. **Filter by status = mapped** to see fires needing review.
6. **Open** the first fire. The split view shows post-fire imagery alongside the ML classification.
7. If the result looks good, click **Accept** on the result card. If not, tweak parameters and re-map.
8. Click **Next** to move to the next fire.
9. Accepted results are in `<out_root>/<raster_stem>_mapping_results/<FIRE_NUMBER>/` with full parameter records. When you're done with that year, switch the year in the filter panel and repeat from step 2 — the next year's `.web_cache/` and canonical dirs live in a sibling directory, so the two years cannot overwrite each other.

---

## Output structure

Multi-year: `<out_root>/` holds shared state at the top and **one sibling directory per year**, named `<raster_stem>_mapping_results/`, with everything per-year inside.

```
<out_root>/                                    # mother dir (--out_root)
    active_year.yaml                           # {active_year: NNNN}
    recommended_settings.yaml                  # Shared user-edited settings
    access_control.yaml                        # Shared approved/blocked/pending IPs
    sessions.yaml                              # Shared persistent login sessions

    pgfc_2023_mapping_results/                 # one sibling per --rasters entry
        fire_state.yaml                        # Per-fire state (survives restart)
        fire_status.yaml                       # Status index for this year's fires
        accepted_params.csv                    # Parameter log (one row per accepted fire)
        notes.yaml                             # Per-fire notes
        .web_cache/                            # Working files (preserved across re-prepares)
            <FIRE_NUMBER>/
                *_crop.bin / .hdr              # Cropped raster
                *_perimeter.bin                # Rasterized perimeter
                VIIRS_*.bin                    # Rasterized VIIRS hint
                *_classified.bin               # Classification output (post-brush)
                *_classified_raw.bin           # Pre-brush HDBSCAN mask + .hdr
                                               #   Created on first rebrush; every
                                               #   subsequent rebrush reads from here
                                               #   so brushes never compound.
                *_comparison.png               # Comparison figure
                *_brush_comparison.png         # Raw vs brushed side-by-side
                *_serial_<ID>_classified.bin   # Per serial run
                *_serial_<ID>_classified_raw.bin # Per-run raw backup
                *_serial_<ID>_brush.png        # Per-run brush preview
                previews/                      # Preview PNGs for web display
        <FIRE_NUMBER>/                         # Accepted results (one directory per fire)
            *_crop.bin / .hdr
            *_classified.bin
            *_comparison.png
            *_brush_comparison.png
            *_params.yaml                      # Full parameter record + ML area
        analyzing_parameters/                  # Parameter Analyzer outputs (admin only)
            analyzer_config.yaml
            analyzer_accepted.csv
            <FIRE_NUMBER>/                     # Only exists after at least one analyzer accept
                <FIRE>_crop_max.bin            # Biggest-padded accepted crop (overlay backdrop)
                <FIRE>_post_max.png
                <FIRE>_overlay.png             # Composite overlay (rebuilt on demand)
                manifest.yaml
                run_0001/                      # One accepted run
                run_0002/
            .analyzer_cache/                   # Working cache, delete-safe
                <FIRE_NUMBER>/
                    p_<padding>/               # One snapshot per padding
                        tsne_rf_<sig>.npz      # Cached intermediate state
                        set_<S>_run_<R>/       # Per-run outputs + sidecar

    pgfc_2024_mapping_results/                 # same shape as above, for 2024
        fire_state.yaml
        accepted_params.csv
        .web_cache/
        <FIRE_NUMBER>/
        analyzing_parameters/
        ...
```

Accepted fire directories (`<out_root>/<stem>_mapping_results/<FIRE_NUMBER>/`) are compatible with `batch_fire_mapping` CLI output. The `analyzing_parameters/` tree under each year's outdir is self-contained and can be deleted independently — see the Parameter Analyzer section for full details.

**Practical note on VIIRS downloads**: VIIRS shapefiles are stored next to each raster (`<raster_dir>/<stem>_VIIRS/VNP14IMG/`), not inside `<out_root>`. This is deliberate so repeated launches against the same raster reuse the same VIIRS cache. At startup, the server prepares VIIRS up-front for **every** year (honoring `--skip_download`), so later year switches never block on a download.

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

All paths below are relative to the active year's outdir (`<out_root>/<stem>_mapping_results/`) unless otherwise noted.

| File | Scope | What it stores | When it's written |
|---|---|---|---|
| `<out_root>/active_year.yaml` | shared | The currently-active year | On startup and on every successful `/api/year/switch` |
| `<out_root>/sessions.yaml` | shared | Hashed session tokens | On login/logout; survives year switches |
| `<out_root>/access_control.yaml` | shared | Approved, blocked, and pending IPs | On every IP action |
| `<out_root>/recommended_settings.yaml` | shared | User-edited parameter presets | On admin save |
| `fire_state.yaml` | per-year | Per-fire status, cache paths, parameters, agreement, hidden flags | After every prepare, mapping, accept, and remove/restore. Also flushed during `/api/year/switch` before the swap. |
| `notes.yaml` | per-year | Per-fire text annotations | On every note edit |
| `accepted_params.csv` | per-year | One row per accepted fire (deduplicated on re-accept) | On accept |
| `<FIRE>/<FIRE>_params.yaml` | per-year | Full per-fire parameter record: context (`fire`, `run`, `inputs`, `crop`, `sampling`, `accumulation`) plus CLI-stage sections (`tsne`, `hdbscan`, `random_forest`, `brush`, `bands`, `output`, `misc`) flattened back into `last_params` on startup | On accept (atomic: tmp + `os.replace`) |
| `analyzing_parameters/analyzer_config.yaml` | per-year | Analyzer config (N sets, M, jitter, fires) | On analyzer save |
| `analyzing_parameters/analyzer_accepted.csv` | per-year | One row per accepted analyzer run (append-only) | On analyzer accept |
| `analyzing_parameters/<FIRE>/manifest.yaml` | per-year | Per-fire accept list + max-crop tracking | On analyzer accept/unaccept |
| `.analyzer_cache/<FIRE>/runs.yaml` | per-year | Fast-reload cache of the in-memory runs list | After every run completes |

On startup, the server:
1. Builds the `{year → raster}` registry from `--rasters`, auto-detecting each year from the filename. Duplicate years error out.
2. Determines the initial active year from `--year` → `<out_root>/active_year.yaml` → newest year, and persists it back to `active_year.yaml`.
3. Prepares VIIRS (download + shapify) for **every** year's raster, unless `--skip_download` is passed. Per-year VIIRS directories live next to each raster as `<raster_stem>_VIIRS/`.
4. Loads the polygon shapefile into memory once (the raw cache used by `/api/year/switch` for fast re-filtering) and filters it for the active year: keeps only polygons whose `FIRE_YEAR` matches and whose geometry intersects the active raster.
5. Loads fires from the filtered GDF and detects accepted fires by checking for `_comparison.png` on disk in the active year's outdir.
6. Restores fire state from the active year's `fire_state.yaml` — mapped fires, cache paths, hidden flags, parameters, and agreement scores are all recovered.
7. Validates that cached files still exist before restoring status. If a fire was mid-mapping when the server crashed, it recovers to MAPPED (if results exist in cache) or READY (if not), rather than being stuck in MAPPING.
8. Initializes the Parameter Analyzer against the active year's `analyzing_parameters/` tree: loads `analyzer_config.yaml`, scans `analyzing_parameters/<FIRE>/` directories to reconstruct accepted-run state, and scans `.analyzer_cache/<FIRE>/` for un-accepted (pending or partial) run sidecars. Fires that had analyzer data on disk return to ANALYZED or PARTIAL status accordingly. Switching years reruns steps 4–8 against the new year's outdir.

The `.web_cache/` directory is no longer wiped on every re-prepare. On a same-padding re-prepare, nothing is cleared — un-accepted mapping results survive across page reloads. On a padding change (which invalidates the crop dimensions), the wipe is **selective**: per-run serial artifacts (`{FIRE}_serial_*` — classified bins and standalone comparison PNGs) are preserved, while the old `previews/` directory (tied to the old post.png extent) and the stale main-crop files are dropped. This matters during a recommended-settings sweep whose settings span multiple padding values: the gallery cards produced by earlier settings stay intact, and on-demand serial overlay PNGs are regenerated from the surviving `classified.bin` via `_overlay_mask_on_post`, which aligns across crop extents using GeoTransform. Previously, any padding change between settings wiped every prior setting's gallery files off disk, leaving in-memory `fire.serial_results` entries pointing at non-existent files (ghost cards with no thumbnails).

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
- `{FIRE}_params.yaml` is written via `_atomic_yaml_dump` (tmp + `os.replace`, `0o644`), matching the rest of the YAML persistence layer. A crash or disk-full mid-write leaves the previous accepted-params file intact rather than truncated. Before this, the file was written with a plain `open('w')` + `yaml.dump`, and a mid-write failure would zero out the entire accepted parameter record for a fire.
- Pipeline parameters round-trip correctly across restart. `fire.last_params` is a **flat CLI-style dict** in memory (`tsne_perplexity`, `rf_n_estimators`, `hdbscan_min_samples`, `embed_bands`, `brush_size`, …). On accept, `_accept_fire_sync` groups keys by prefix into nested YAML sections (`tsne`, `hdbscan`, `random_forest`, `brush`, `bands`, `output`, `misc`) so a reader can pull a whole stage without string parsing; unknown keys fall into `misc`. On startup, `init_fires_from_gdf` flattens those sections back into the flat dict. The context-only sections (`fire`, `run`, `inputs`, `crop`, `sampling`, `accumulation`) are skipped during flatten so they never pollute `last_params`. Previously the accept path expected nested sub-dicts already in `last_params` and silently wrote nothing, so every accepted YAML (and the PDF built from it) lost t-SNE, RF, HDBSCAN, brush, and band settings.
- `_prepare_fire_sync` refuses concurrent prepare by checking only `PREPARING`, not `MAPPING`. The serial worker flips `status=MAPPING` before calling `_prepare_fire_sync` between sweep settings whose paddings differ; treating `MAPPING` as "busy" (the old behavior) was the root cause of batch-mapping failures on never-opened fires — the guard no-op'd, `crop_bin`/`hint_bin` stayed empty, and the CLI subprocess crashed on empty positional args. Same-fire double-prepare from external callers is still rejected.
- GDAL dataset handles in the hot paths (`_compute_agreement`, `_overlay_mask_on_post`'s hint contour read, `preview.generate_preview_png`, and `preview.generate_all_previews`' band-count fallback) are released in `try/finally` so an exception in `ReadAsArray` or downstream numpy/scipy calls cannot leak file descriptors. Under a long-running session with many rebrushes, previews, and cross-extent agreement calls, the old code could exhaust GDAL's internal handle cache and wedge subsequent opens.

---

## Known limitations

- **HDBSCAN non-determinism**: cuML's GPU HDBSCAN does not support a random seed. Identical runs *may* produce slightly different results due to GPU parallelism, but for some inputs the GPU is effectively deterministic. All other pipeline stages are seeded and deterministic. The Parameter Analyzer works around this by jittering `hdbscan_min_samples` across replicates (see the jitter section above).
- **Single GPU queue**: one fire maps at a time. Additional requests queue automatically. The analyzer releases the GPU lock between runs so user mappings can interleave.
- **No TLS**: the server uses plaintext HTTP. Use an SSH tunnel or reverse proxy for encrypted access. Cookies omit the `Secure` flag on plain HTTP so that login works over VPN / LAN IPs (see the security section); `Secure` is added back automatically when running behind an HTTPS-terminating proxy with `--trust_proxy`.
- **Login rate limiting is in-memory**: the 5-attempt rate limit resets on restart. The table is size-bounded (see Security hardening) but not persisted. All other security state persists.
- **Analyzer disk usage grows with grid size**: every completed run stores a classification `.bin`, comparison PNG, and thumbnail (typically ~5-50 MB per run). A grid of 2500 runs can consume many GB. Delete the `.analyzer_cache/<FIRE>/` tree for a fire once you have accepted everything you want; the accepted results live in `analyzing_parameters/<FIRE>/run_XXXX/` and are unaffected.
- **Analyzer concurrency**: only one analyzer session runs at a time (enforced server-side). Only admins can start one.

---

## HTTP API reference (selected endpoints)

The server is a plain `http.server` dispatcher; routes are registered on `FireHandler` via regex tables in `app.py`. The endpoints most relevant to the mapping / rebrush workflow are documented below. All POST endpoints require an `X-Requested-With: XMLHttpRequest` header (CSRF guard) and a valid session cookie unless `--insecure_no_auth` is on.

### Mapping / state

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/fire/<FIRE>/console`  | Returns the fire's console log tail, current status, serial-results array, and a `rebrush_running` boolean. The frontend polls this to keep the UI in sync after a page reload — a `true` `rebrush_running` triggers the rebrush-adopt UI path. |
| `POST` | `/api/fire/<FIRE>/map`      | Kicks off a serial mapping job. Body: `{mode: 'primary' \| 'settings', settings: [{label, params}, …]}`. Map Fire sends the panel as a single-element `settings` array; Map Fire with settings omits `settings` and the server fans out the recommended YAML. |
| `POST` | `/api/fire/<FIRE>/accept`   | Accepts a specific serial run into the active year's canonical `<out_root>/<raster_stem>_mapping_results/<FIRE>/` directory. |

### Multi-year

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/years`        | Returns `{years: [NNNN, …], active: NNNN}` — the sorted list of years in the registry plus the active year. Any authenticated session. |
| `POST` | `/api/year/switch`  | Admin-only. Body: `{year: NNNN}`. Swaps the active raster, outdir, polygon filter, fires, fire-state, and analyzer in place under `state.lock`. Returns `{ok: true, year: NNNN}` on success; `409` with `{error: '…'}` if any mapping / batch / analyzer job is running (cancel or wait). The browser should call `location.reload()` on a `200`. |

### Rebrush (new, as of 2026-04-20)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/fire/<FIRE>/rebrush`         | Re-run `class_brush.exe` on an existing classification. See body schema below. |
| `POST` | `/api/fire/<FIRE>/rebrush/cancel`  | SIGTERM the currently-running rebrush for this fire. 200 even if nothing is running. |

**`/rebrush` request body:**

```json
{
  "brush_size":         15,
  "point_threshold":    10,
  "brush_all_segments": false,
  "run_id":             null
}
```

- `brush_size` (int, 1..10000) — morphological brush radius in pixels. Required.
- `point_threshold` (int, 1..10_000_000) — min connected-component size, in pixels. Required.
- `brush_all_segments` (bool) — apply brush per-segment when true; dominant-segment only when false. Default false.
- `run_id` (int or null) — when set, rebrushes that serial run's `classified.bin` only and regenerates its per-run `*_serial_<ID>_brush.png`. When null, rebrushes the main canonical `classified.bin` used by the left-pane ML overlay.

**`/rebrush` response (2xx):**

```json
{
  "status":             "ok"  | "cancelled",
  "run_id":             null | <int>,
  "ml_area_ha":         <float or null>,
  "agreement_pct":      <float or null>,
  "brush_size":         <int>,
  "point_threshold":    <int>,
  "brush_all_segments": <bool>
}
```

**`/rebrush` error responses:**

| Status | Condition |
|---|---|
| 400 | Parameter failed `_validate_param` (out of bounds, wrong type). |
| 404 | `Fire not found`, `No classification found to rebrush`, or `Post-fire preview missing; prepare the fire first`. |
| 409 | Fire is currently `MAPPING`, or another rebrush is already running for this fire (single-slot per `fire_numbe` in `_rebrush_procs`). |

On success, for a whole-fire rebrush, the server updates `fire.ml_area_ha`, `fire.agreement_pct`, regenerates `*_comparison.png` and `*_brush_comparison.png`, and persists fire state. For a per-run rebrush, the server updates the matching entry in `fire.serial_results` so the gallery card stats reflect the new values on the next poll.

---

## File overview

| File | Purpose |
|---|---|
| `__main__.py` | Entry point. Parses arguments, auto-detects each raster's year via `_year_from_filename`, builds the `{year → raster/outdir/viirs_dir}` registry, resolves the initial active year, prepares VIIRS per year, filters polygons for the active year, initializes state, starts server, registers analyzer routes. |
| `app.py` | Web server, all route handlers, mapping orchestration, template rendering. Owns `_switch_year` (in-place year swap), `_save_active_year`, and the `/api/year/switch` + `/api/years` endpoints. |
| `state.py` | Data classes for per-fire state (`FireInfo`) and global app state (`AppState`). `AppState` carries the multi-year registry (`active_year`, `shared_root`, `rasters_by_year`, `outdirs_by_year`, `viirs_shp_dirs_by_year`, `polygon_gdf_raw`) alongside the active-year views (`raster_path`, `output_root`, `viirs_shp_dir`, `gdf`, `viirs_gdf`). |
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
