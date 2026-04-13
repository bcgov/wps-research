# Batch Fire Mapping - Web Interface

Interactive web interface for Sentinel-2 fire mapping using T-SNE, Random Forest, and HDBSCAN.

Wraps the same GPU-accelerated pipeline as `batch_fire_mapping` (`fire_mapping_cli.py`) but replaces the sequential CLI batch with a browser-based workflow. Users can inspect fires, tune parameters per-fire or apply size-based recommended settings, map individually or in batch, review results, and accept or re-map.

Last updated: 2026-04-13

---

## Requirements

- Python 3.10+
- All packages required by the batch pipeline (numpy, GDAL, scipy, matplotlib, geopandas, shapely, cuml, pyyaml)
- A CUDA-capable GPU (required by cuML for T-SNE and HDBSCAN)
- No additional web framework packages -- the server uses Python's built-in `http.server`

---

## Quick Start

```bash
python -m batch_fire_mapping_web \
    /path/to/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    /path/to/stack.bin \
    --skip_download \
    --out_dir ./results \
    --admin_password myadminpass \
    --user_password userpass123
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
| `--out_dir DIR` | same directory as the raster | Root directory for all outputs (cache, accepted results, access control, sessions). |

#### Perimeter / VIIRS

| Flag | Default | Description |
|---|---|---|
| `--perimeter_mode {viirs,traditional}` | `viirs` | Classification hint source. `viirs`: use VIIRS active fire detections when available, fall back to the traditional polygon perimeter. `traditional`: always use the fire polygon perimeter. |
| `--skip_download` | off | Skip VIIRS download and shapify steps. Use this when VIIRS data was already downloaded by a previous run. |
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
| `--admin_password PASSWORD` | none | Administrator password. Grants full access including IP approval and admin dashboard. Can also be set via `FIRE_ADMIN_PASSWORD` environment variable. When omitted (along with `--user_password`), no authentication is required. |
| `--user_password PASSWORD` | none | Generic user password. Grants fire mapping access only after IP is approved by an admin. Can also be set via `FIRE_USER_PASSWORD` environment variable. Requires `--admin_password` to also be set. Must differ from the admin password. Prefer environment variables on shared systems (CLI args are visible in `ps`). |

---

## Access Control

The server implements a two-tier access control system: custom login page with cookie-based sessions, plus IP-based approval.

### Login

Users see a branded login page with the BCWS logo, username, and password fields. Sessions are stored as secure `HttpOnly` cookies (`SameSite=Strict`, 30-day expiry). Sessions persist across server restarts via `<out_dir>/sessions.yaml`. A **Logout** button is available on every page to switch users or roles.

### Roles

| Role | Password | Capabilities |
|---|---|---|
| **Admin** | `--admin_password` | Full access. IP auto-approved on first login. Can approve/block other IPs via the admin dashboard. Can edit recommended settings in the web UI. |
| **User** | `--user_password` | Fire mapping access only. Must have IP approved by an admin before accessing any content. Cannot access the admin dashboard or edit settings. |

### IP Approval Flow

1. A new user logs in with the user password.
2. The server registers their IP as **pending** and shows a "waiting for approval" page (auto-refreshes every 5 seconds).
3. An admin opens the **Admin Dashboard** (`/admin`) and sees the pending IP with the username entered at login.
4. The admin clicks **Approve** or **Block**.
5. If approved, the user's page automatically redirects to the fire list.
6. Approved and blocked IPs persist to `<out_dir>/access_control.yaml` and survive server restarts.
7. The admin can later **Revoke** an approved IP or **Unblock** a blocked IP.

### Admin Dashboard (`/admin`)

Available only to admin users. Shows:

- **Pending IPs** -- users waiting for approval, with username, first/last seen timestamps, Approve/Block buttons.
- **Approved IPs** -- active users, with username, role, who approved them, Revoke button.
- **Blocked IPs** -- denied users, with username, who blocked them, Unblock button.
- **GPU Queue** -- currently running mapping job and waiting queue, with fire number, client IP, and timestamps.

The dashboard auto-refreshes every 5 seconds.

---

## Examples

### First run (downloads VIIRS data)

```bash
python -m batch_fire_mapping_web \
    /data/GIS/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    /ram/T09UYU_prepost/stack.bin \
    --out_dir ./pgfc_results \
    --admin_password secretadmin \
    --user_password teampassword
```

### Skip VIIRS download (data already exists)

```bash
python -m batch_fire_mapping_web \
    /data/GIS/IN_HISTORICAL_FIRE_POLYGONS_SVW.shp \
    /ram/T09UYU_prepost/stack.bin \
    --skip_download \
    --out_dir ./pgfc_results \
    --admin_password secretadmin \
    --user_password teampassword
```

### Using environment variables (recommended for shared HPC)

```bash
export FIRE_ADMIN_PASSWORD=secretadmin
export FIRE_USER_PASSWORD=teampassword

python -m batch_fire_mapping_web \
    perimeters.shp stack.bin \
    --skip_download --out_dir ./results
```

### Access from another machine on the same network

The server binds to `0.0.0.0` by default. On startup it prints the network address:

```
============================================================
  Server ready!
  Local:   http://localhost:8765
  Network: http://142.36.19.133:8765
           http://heartofgold:8765
  Auth:    admin + user passwords configured
  IP ctrl: ./pgfc_results/access_control.yaml
  45 fire(s) available
============================================================
```

From another machine, open `http://142.36.19.133:8765` in a browser. The login page prompts for username and password. Admin enters the admin password; regular users enter the user password. New user IPs require admin approval before they can access the fire list.

Alternatively, use an SSH tunnel for encrypted access:

```bash
ssh -L 8765:localhost:8765 user@heartofgold
# Then open http://localhost:8765 on your local machine
```

---

## Recommended Settings

The file `recommended_settings.yaml` in the package directory defines parameter presets based on fire size ranges. If the file exists, it is loaded automatically on startup.

### File format

```yaml
- min_ha: 0
  max_ha: 500
  params:
    padding: 0.1
    sample_rate: 0.05
    embed_bands: '5,6,7,9,10,11,13,14,15'
    tsne_perplexity: 60
    tsne_learning_rate: 200
    tsne_max_iter: 2000
    tsne_init: pca
    tsne_n_components: 2
    tsne_random_state: 42
    rf_n_estimators: 100
    rf_max_depth: 15
    rf_max_features: sqrt
    rf_random_state: 42
    controlled_ratio: 0.5
    hdbscan_min_samples: 20
    contour_width: 1
    seed: 123
    min_samples: 500
    max_samples: 30000

- min_ha: 500
  max_ha: null    # null = infinity
  params:
    padding: 0.1
    embed_bands: '1,2,3,5,6,7,9,10,11,13,14,15'
    # ... same structure, different bands for larger fires
```

### How it works

- Each row defines a fire size range (`min_ha` to `max_ha`) and the parameters to use for fires in that range. `max_ha: null` means infinity.
- **Single fire**: on the fire mapping page, the **"Map Fire / with settings"** button applies the matching preset for the fire's size (including re-cropping if padding differs), then starts mapping.
- **Batch**: on the fire list page, selecting fires and clicking **"Map Selected (with settings)"** maps all selected fires using the appropriate preset for each fire's size.
- The settings panel on the fire list page allows admins to view and temporarily edit settings in-memory. **Edits in the web UI are not written back to the YAML file** -- the file is the source of truth on restart.
- To permanently change settings, edit `batch_fire_mapping_web/recommended_settings.yaml` directly.

---

## Web UI Workflow

### 1. Fire List (home page)

**Table**:
- Sortable table of all fires from the shapefile. Click any column header to sort.
- Each row has a **checkbox** for batch selection and an **Open** button for individual fire mapping.

**Checkbox selection and batch actions**:
- Select individual fires with checkboxes, or use the header checkbox to select/deselect all visible fires.
- When fires are selected, a sticky **batch action bar** appears:
  - **Map Selected (with settings)**: queues all selected fires for batch mapping using recommended settings based on each fire's size. Accepted fires are automatically skipped. Fires process one at a time through the GPU queue. Progress shows `3/12 done (C11659)` in real time.
  - **Remove Selected**: hide selected fires from the list (with confirmation). Does not delete any files on disk.

**Filters** (collapsible panel):
- **Year**: checkboxes auto-populated from the shapefile.
- **Fire Number**: regex text input (e.g., `C11.*`, `G8`).
- **Min Size (ha)**: number input.
- **Status**: checkboxes for pending / ready / mapping / mapped / accepted / error.
- Live match count updates as you change filter values. Click **Apply** to filter the table, **Reset** to clear all filters.
- Filter state (including sort column and direction) persists across page navigation via browser session storage.

**Recommended Settings** (collapsible panel):
- View and edit parameter presets by fire size range.
- Add thresholds to split ranges. Remove rows individually.
- Save updates in-memory for the current server session (not written to disk).

**Other features**:
- **Admin** button (admin only): opens the admin dashboard.
- **Logout** button: clears session, returns to login page.
- Status badges: green = accepted, blue = mapped, yellow = ready, gray = pending, red = error.
- Green checkmark badge next to status: fire was previously accepted but is now being re-worked with different parameters.
- Auto-refreshes every 5 seconds to show status changes from other users or batch progress.

### 2. Fire Mapping Page

**Navigation**:
- **Prev / Next** buttons in the header navigate to the previous/next fire based on the current filtered and sorted order from the fire list. This lets you review mapped fires sequentially without returning to the list each time.
- **Back to fire list** link returns to the main page with filters preserved.

**On page load**, the server automatically:
1. Crops the raster to the fire bounding box (with padding, default 0.1).
2. Runs VIIRS accumulation and rasterizes the hint perimeter.
3. Rasterizes the traditional polygon perimeter.
4. Generates preview images (post-fire, pre-fire, diff views).

**Image viewer** (left panel):
- Dropdown to switch between preview images (post, pre, diff) and result images (comparison, brush comparison).
- Scroll wheel to zoom in/out (zooms towards cursor position).
- Click and drag to pan.
- Toolbar buttons: **+** (zoom in), **-** (zoom out), **Fit** (reset to fit window).
- `image-rendering: pixelated` for crisp pixels at all zoom levels -- important for small fires.

**Parameters** (right panel, collapsible sections):
- **Crop & Sampling**: padding (with Re-crop button), sample rate, min/max samples, seed.
- **T-SNE**: perplexity, learning rate, max iterations, init method, components, random state, embed bands.
- **Random Forest**: number of estimators, max depth, max features, random state.
- **HDBSCAN**: controlled ratio, min samples.
- **Display**: contour width (integer pixels).

**Action buttons**:
- **Map Fire**: maps using the parameters currently shown in the form.
- **Map Fire / with settings**: looks up the recommended settings for this fire's size, applies them to the form (including re-cropping if padding differs), and starts mapping.
- **Accept**: saves the mapped result to the canonical output directory. Only appears after a successful mapping.

**Console** (bottom of right panel):
- Streams every line of `fire_mapping_cli.py` output in real time via Server-Sent Events.
- Resizable by dragging the handle between parameters and console.

**Mapping workflow**:
1. Adjust parameters manually, or let recommended settings handle it.
2. Click **Map Fire** or **Map Fire / with settings**.
3. Console shows real-time progress (loading image, T-SNE, HDBSCAN clusters, Random Forest, class_brush, figure generation).
4. When complete, the comparison figure appears in the image viewer.
5. Review the result. If satisfactory, click **Accept**. Otherwise, adjust parameters and map again.
6. Use **Next** to move to the next fire in your filtered list.

**GPU queue**: only one fire processes at a time (GPU serialization). If another mapping is already running, the console shows queue position and waits automatically.

**Re-mapping accepted fires**: changing padding and clicking Re-crop resets the fire to "ready" status with a green checkmark badge indicating it was previously accepted. The old accepted results on disk are not overwritten until you Accept the new mapping.

**Auto-refresh**: status polls every 5 seconds. If another user or the batch process maps or accepts the fire, the UI updates automatically (comparison appears, status badge changes).

---

## Typical Review Workflow

The intended workflow for processing a set of fires:

1. **Filter** the fire list (e.g., by year, minimum size, or fire number pattern).
2. **Select all** filtered fires using the header checkbox.
3. Click **Map Selected (with settings)** to batch-map them all with recommended settings.
4. Monitor progress in the batch bar and terminal. The fire list auto-refreshes as each fire is mapped.
5. **Filter by status = "mapped"** to see only fires that need review.
6. **Open** the first mapped fire.
7. Review the comparison image. If acceptable, click **Accept**. If not, adjust parameters and re-map.
8. Click **Next** to move to the next mapped fire. Repeat until all are reviewed.
9. Accepted results are in `<out_dir>/<FIRE_NUMBE>/` with full parameter records.

---

## Output Structure

All outputs are written under `--out_dir` (or the raster directory if not set).

```
<out_dir>/
    fire_status.yaml                        # Status index for all fires
    access_control.yaml                     # Persistent approved/blocked IP list
    sessions.yaml                           # Active login sessions (survives restarts)
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
| Working files | `.web_cache/<FIRE_NUMBE>/` | Fire is prepared (page load or batch). Overwritten on every re-map. |
| Preview images | `.web_cache/<FIRE_NUMBE>/previews/` | Prepare step. For web display only, not copied on Accept. |
| Accepted results | `<FIRE_NUMBE>/` | Accept button is clicked. Includes `_params.yaml` with ML area estimate. |
| Access control | `access_control.yaml` | First admin login. Updated on every IP approve/block/revoke. |
| Sessions | `sessions.yaml` | Each login. Cleaned up on startup (expired sessions removed). |
| VIIRS downloads | `<raster_dir>/<raster_name>_VIIRS/` | VIIRS download step. Stored next to the raster, shared with `batch_fire_mapping`. |

Accepted fire folders have the same structure and file names as the `batch_fire_mapping` CLI output and are fully compatible. The `fire_status.yaml` index is also compatible -- web entries include `source: web`.

---

## Model Parameters

All parameters are forwarded to `fire_mapping_cli.py` and match the batch pipeline CLI flags:

| Parameter | UI Location | CLI Flag | Default |
|---|---|---|---|
| Padding | Crop & Sampling | `--padding` | 0.1 |
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

## Security

- **Two-role authentication** via custom login page with cookie-based sessions (`HttpOnly`, `SameSite=Strict`). Passwords checked with constant-time comparison (`hmac.compare_digest`). Sessions persist across restarts (30-day expiry). Startup validation prevents misconfiguration (identical passwords, user-only without admin).
- **IP-based access control**: user IPs must be approved by an admin before accessing any content. Approved/blocked IPs persist to `access_control.yaml`. Admin IPs are auto-approved.
- **CSRF protection**: POST requests require either a matching `Origin` header or `X-Requested-With` header. HTML form attacks from external sites are blocked.
- **Path traversal protection**: fire numbers sanitized on load (reject `..`, `/`, `\`). Preview view names validated against `[A-Za-z0-9_-]+`.
- **Request body limit**: 1 MB maximum to prevent memory exhaustion.
- **Cache control**: all HTML responses include `Cache-Control: no-store` to prevent stale cached pages.
- **No TLS**: the server uses plaintext HTTP. On internal BC government networks this is acceptable. For untrusted networks, use an SSH tunnel (`ssh -L 8765:localhost:8765 user@host`) or place behind a TLS-terminating reverse proxy.

---

## Known Limitations

- **HDBSCAN non-determinism**: cuML's GPU HDBSCAN does not support a random seed parameter. Identical runs may produce slightly different cluster assignments due to GPU parallelism. All other pipeline stages (T-SNE, Random Forest, sampling) are seeded and deterministic.
- **Single GPU queue**: only one fire maps at a time. Additional requests (individual or batch) are queued and wait automatically. Queue state is visible in the admin dashboard.
- **IP granularity**: access control is per-IP address. If multiple users share an IP (e.g., behind a NAT), they are treated as one user. On internal government networks, IPs are typically unique per workstation.
- **Recommended settings file**: edits made in the web UI are in-memory only and lost on server restart. To make permanent changes, edit `batch_fire_mapping_web/recommended_settings.yaml` directly.
