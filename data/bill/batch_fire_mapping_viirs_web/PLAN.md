# batch_fire_mapping_viirs_web — Implementation Plan

*Author: planning conversation, 2026-04-27. Implementing agent: read top-to-bottom, then start at §11 file-map.*

This package is a sibling to `data/bill/batch_fire_mapping_web/`. It replaces the **polygon-driven fire list** with a **user-defined fire** workflow: an analyst draws a bounding box on the year's reference raster, names a fire, supplies a date range, and the server downloads VIIRS data for that bbox/range, accumulates+rasterizes it, derives a tight crop from the actual fire pixels, and seeds the standard ML mapping pipeline.

The downstream mapping/brush/accept/serial flow is **unchanged** from `batch_fire_mapping_web`. Only the *front of the pipeline* changes.

---

## 0. Origin and code-share strategy

- Initial source: full file-by-file copy of `batch_fire_mapping_web/`. Diff against that copy.
- Why duplicate instead of share: the `_web` package is stdlib-only with mixin-globals binding (`handlers/*.py:init`); making it parameterizable on "polygon vs. user-fire" mode would force it through both code paths and pollute the existing audit-clean module. A copy is cheaper and lets the existing package keep evolving.
- Implementing agent should `cp -r` first, then make the edits in §3-§9. Run the test suite (§10) at the end.

---

## 1. Top-level package layout

```
batch_fire_mapping_viirs_web/
├── __init__.py                      (empty, copy)
├── __main__.py                      (EDIT — drop polygon arg, add overview gen, token check)
├── app.py                           (EDIT — drop polygon refs in init_app, add new_fire route)
├── state.py                         (EDIT — drop gdf/viirs_gdf, add bbox/is_new fields, init_fires_from_disk)
├── auth.py                          (copy unchanged)
├── notifications.py                 (copy unchanged)
├── cache_retention.py               (copy unchanged)
├── progress.py                      (EDIT — add VIIRS sub-stage labels)
├── mapping.py                       (copy unchanged)
├── persistence.py                   (EDIT — drop polygon paths from _switch_year, fire_state)
├── brush.py                         (copy unchanged)
├── kml.py                           (copy unchanged)
├── templates.py                     (copy unchanged)
├── validation.py                    (EDIT — add bbox/date/name validators)
├── mapping_cmd.py                   (copy unchanged)
├── prepare.py                       (EDIT — replace polygon-driven crop with VIIRS-tight crop)
├── workers.py                       (copy unchanged)
├── preview.py                       (copy unchanged)
├── io_utils.py                      (copy unchanged)
├── overview.py                      (NEW — per-year raster overview PNG + sidecar JSON)
├── viirs_worker.py                  (NEW — download → shapify → accumulate → rasterize → tight crop)
├── recommended_settings.yaml        (copy unchanged)
├── handlers/
│   ├── __init__.py                  (copy)
│   ├── base.py                      (copy)
│   ├── auth.py                      (copy)
│   ├── fire_list.py                 (EDIT — add new_fire page route, /api/fire/create, /api/fire/<n>/cancel_create, /api/fire/<n>/clear_new, /api/year/<y>/overview*; drop gdf-based fire fields)
│   ├── fire.py                      (EDIT — handle_api_prepare reads from FireInfo bbox instead of polygon)
│   ├── mapping.py                   (copy unchanged)
│   ├── serial.py                    (copy unchanged)
│   ├── rebrush.py                   (copy unchanged)
│   ├── batch.py                     (copy unchanged)
│   ├── ops.py                       (copy unchanged)
│   └── static.py                    (copy unchanged)
├── templates/
│   ├── login.html                   (copy)
│   ├── pending.html                 (copy)
│   ├── admin.html                   (copy)
│   ├── fire_list.html               (EDIT — add "+ New Fire" button, new badge, sub-stage display)
│   ├── new_fire.html                (NEW — overview canvas + bbox drawer + right panel)
│   └── fire_mapping.html            (copy unchanged — downstream flow same)
├── static/
│   ├── style.css                    (EDIT — append new_fire.html styles + new badge styles)
│   ├── help.js                      (copy)
│   ├── new_fire.js                  (NEW — bbox drawer, coordinate readout, form validation)
│   └── BC-Wildfire-Service-logo.png (copy)
├── tests/
│   ├── conftest.py                  (NEW — synthetic raster fixture)
│   ├── test_overview.py             (NEW)
│   ├── test_bbox_validation.py      (NEW)
│   ├── test_date_defaults.py        (NEW)
│   ├── test_date_validation.py      (NEW)
│   ├── test_fire_name_validation.py (NEW)
│   ├── test_tight_crop.py           (NEW)
│   ├── test_fire_create_endpoint.py (NEW)
│   ├── test_viirs_worker_cancel.py  (NEW)
│   └── test_overview_caching.py     (NEW)
└── PLAN.md                          (this file)
```

---

## 2. Conceptual model — what changes

| Concept | `batch_fire_mapping_web` | `batch_fire_mapping_viirs_web` |
|---|---|---|
| Fire source | Polygon shapefile (FIRE_NUMBE rows) | User-drawn bbox + name |
| Fire identity | `FIRE_NUMBE` from polygon attribute | User-supplied `fire_name` |
| Crop bounds source | Polygon geometry intersection + padding | VIIRS hint pixel bounds + padding (computed *after* download) |
| User-drawn bbox | n/a | VIIRS download AOI (NOT the final crop) |
| VIIRS data location | Per-year shared `<raster>_VIIRS/` | Per-fire `.web_cache/<NAME>/_VIIRS/` |
| VIIRS download timing | Once at startup, all years | On-demand, per fire creation |
| `perimeter_mode` | `viirs` or `traditional` | `viirs` only (concept removed) |
| `perimeter_type` field | `'viirs'` or `'traditional'` per fire | always `'viirs'` |
| `agreement_pct` | ML vs. hint (still works) | ML vs. VIIRS hint — same code path |
| Multi-year | Yes | Yes (year selects active reference raster) |
| Polygon arg | Required positional | Removed |
| Source polygon shp | Required | Not used |

The user-drawn bbox is the **VIIRS download AOI** so the operator can avoid pulling fire pixels from across the entire raster footprint. The **final crop** of the Sentinel-2 raster is derived after accumulation by tightening to the actual fire pixels (mirrors `prepare.py:117-128` exactly, just sourcing bounds from the VIIRS bin instead of the polygon).

---

## 3. CLI changes — `__main__.py`

### Remove
- positional arg `polygon_file`
- `--perimeter_mode` (always viirs)
- `--skip_download` (no startup download)
- `--shapify_workers` (workers are per-fire now; expose as `--viirs_download_workers` and `--viirs_shapify_workers` defaulting to 16 / 8)
- imports: `load_and_filter_polygons`, `load_all_viirs`, `download_viirs`, `shapify_viirs`
- the `_prepare_year_for_viirs` helper

### Keep
- `--rasters <r1> <r2> ...` (one per year)
- `--out_root`
- `--year` (initial active year)
- `--host`, `--port`, `--admin_password`, `--user_password`, `--insecure_no_auth`, `--trust_proxy`
- year-from-filename detection (`_year_from_filename`)
- per-year output dir derivation `<out_root>/<raster_stem>_mapping_results`

### Add
- LAADS token load at startup. If `/data/.tokens/laads` missing or unreadable: `sys.exit('ERROR: ...')` with actionable message.
- Per-year overview generation (sync, blocking — one-time cost). Print progress per raster.
- After overview generation, store paths in `app_state.overview_png_by_year` and `app_state.overview_meta_by_year`.
- `app_state.laads_token` field.

### Startup sequence (replaces current Steps 1-3)
1. Validate `/data/.tokens/laads` exists, load into `app_state.laads_token`.
2. For each `(year, raster)` in `rasters_by_year`:
   - Compute overview cache path: `<shared_root>/.web_cache/_overviews/<raster_stem>.png` and `<...>.json`.
   - Cache key: `(raster_path, st_mtime_ns, st_size)` stored in JSON.
   - If JSON exists and key matches: skip regeneration.
   - Else: call `overview.generate_overview(raster, png_path, json_path, max_dim=2000)` (blocking).
   - Store paths in app_state maps.
3. Init `AppState` with no polygon, no `gdf`, no `viirs_gdf`. Call `state.init_fires_from_disk()` to rebuild fire registry from `<output_root>/` and `<.web_cache>/`.

---

## 4. AppState changes — `state.py`

### Remove fields
- `self.gdf`
- `self.viirs_gdf`
- `self.polygon_file`
- `self.polygon_gdf_raw`
- `self.viirs_shp_dir`
- `self.viirs_shp_dirs_by_year`

### Add fields
```python
# Overview PNG + sidecar JSON for the bbox-drawing page, per year.
self.overview_png_by_year: dict = {}   # {year: abs path to overview .png}
self.overview_meta_by_year: dict = {}  # {year: abs path to overview .json}

# LAADS DAAC token (read once at startup).
self.laads_token: str = ""

# Registry of running VIIRS prepare workers, keyed by fire_name.
# Mirrors _serial_procs / _rebrush_procs shape so cancel handlers work
# the same way. Each entry is the Popen handle.
self.viirs_jobs: dict = {}              # {fire_name: subprocess.Popen}
# Lock lives in app.py module globals (parallel to _serial_procs_lock).
```

### `FireInfo` additions / changes (`state.py`)
```python
# User-drawn bbox in the raster's native CRS — VIIRS download AOI.
bbox_native: Optional[tuple] = None     # (x_min, y_min, x_max, y_max) or None
# Same bbox in WGS84 — for LAADS DAAC URL.
bbox_wgs84: Optional[tuple] = None      # (W, S, E, N) or None
# User-entered date range (YYYY-MM-DD strings).
viirs_start_date: str = ""
viirs_end_date: str = ""
# Flips True the moment the fire reaches READY for the first time.
# Flips False when a logged-in user opens its detail page.
# Drives the "new" badge in the fire list.
is_new: bool = False
```

### `FireInfo` field reused or repurposed
- `fire_numbe` → reused as the user-supplied name (validated regex match — same as today's polygon-derived names: `^[A-Za-z0-9][A-Za-z0-9_. -]*$`, no `..`, no `/\\`, length ≤ 64).
- `fire_year` → set to the active year at create time (used only for filtering/display).
- `fire_size_ha` → computed from VIIRS hint area (after rasterize). Until then: `0`.
- `fire_date` → set to `viirs_end_date` for display compatibility.
- `perimeter_type` → always `'viirs'`.

### Replace `init_fires_from_gdf()` with `init_fires_from_disk()`
- Scan `<output_root>/<NAME>/<NAME>_params.yaml` → load each accepted fire's name + saved bbox/dates.
- Scan `<output_root>/.web_cache/<NAME>/` for in-flight or stale fires; reconstruct `FireInfo` skeleton with `status=PREPARING` (worker re-attaches if still running) or `ERROR` (if no live worker and stage file shows incomplete).
- Persist new fields in `fire_state.yaml` via `persistence._save_fire_state` (already extensible — just add the new keys).

### Status enum
Keep `FireStatus` exactly as today. The "downloading_viirs / shapifying / accumulating / rasterizing / cropping" sub-stages live in `fire.progress.stage` (already exposed by `progress.py:_progress_snapshot` and polled by the UI). Decision rationale: avoid pushing the UI status filter to seven new values; the existing `PREPARING` status with rich progress payload already gives the operator everything.

---

## 5. Overview module — `overview.py` (NEW)

### Purpose
Generate ONE downsampled PNG per year-raster, cached on disk, served as a static file. The bbox-drawing UI uses this PNG as its background; pixel→map→lat-lon math runs client-side off the sidecar JSON.

### API
```python
def generate_overview(
    raster_path: str,
    png_path: str,
    json_path: str,
    max_dim: int = 2000,
) -> None:
    """Generate overview PNG + sidecar JSON. Raises on failure.

    Reads the raster with GDAL ReadAsArray(buf_xsize=, buf_ysize=) so
    only ~max_dim*max_dim*4*4 bytes are allocated regardless of source
    size — a 100 GB raster reads at the same memory cost as a 100 MB
    one. Uses the same band-detection / percentile-stretch logic as
    preview.py:124 generate_preview_png. Writes a sidecar JSON:
    {
      "raster_path": str,
      "raster_stem": str,
      "raster_W": int,        # source pixel width
      "raster_H": int,        # source pixel height
      "geotransform": list,   # 6-tuple
      "crs_wkt": str,
      "overview_W": int,      # PNG pixel width
      "overview_H": int,      # PNG pixel height
      "year": int,
      "default_start": "YYYY-03-01",
      "default_end":   "YYYY-10-30",
      "extent_native": [x_min, y_min, x_max, y_max],
      "extent_wgs84":  [W, S, E, N],
      "cache_key": {"st_mtime_ns": int, "st_size": int}
    }
    """

def overview_is_fresh(raster_path: str, json_path: str) -> bool:
    """Return True iff json_path exists and its cache_key matches the
    current raster_path stat. Used to skip regeneration."""
```

### Implementation notes
- Band detection: reuse `preview.py:detect_band_groups` against header band names; prefer the `post` group (or `B12/B11/B9` fallback).
- Reproject bbox-corner polygon (raster CRS → EPSG:4326) using the same `_bbox_to_4326` code as `viirs/fp_gui/download_dialog.py:63`.
- Atomic write: write to `*.tmp`, fsync, rename. (Reuse `io_utils._atomic_yaml_dump` pattern.)
- Default dates: derive year from `_year_from_filename(raster_path)`; emit `f'{year}-03-01'` / `f'{year}-10-30'`.

---

## 6. New page — `templates/new_fire.html` + `static/new_fire.js`

### Layout (single-column on narrow screens, two-column on wide)
```
┌───────────────────────────────────────┬───────────────────────────────┐
│  YEAR: [2023▾]  (admin only on multi) │  Fire Name: [_____________]   │
│                                       │                               │
│  ┌─────────────────────────────────┐  │  Bounding box (native CRS)   │
│  │                                 │  │   x_min: [readonly]          │
│  │   <img id="overview"            │  │   y_min: [readonly]          │
│  │     src="/api/year/2023/        │  │   x_max: [readonly]          │
│  │          overview.png">         │  │   y_max: [readonly]          │
│  │   <canvas overlay/>             │  │                               │
│  │                                 │  │  Bounding box (WGS84 deg)    │
│  │   draw rectangle to define      │  │   W: [readonly]   E:[ro]     │
│  │   VIIRS download AOI            │  │   S: [readonly]   N:[ro]     │
│  │                                 │  │                               │
│  │                                 │  │  Start (YYYY-MM-DD)          │
│  │                                 │  │  [____-03-01]  (placeholder) │
│  └─────────────────────────────────┘  │                               │
│                                       │  End (YYYY-MM-DD)            │
│  Status: <hover for live coords>      │  [____-10-30]  (placeholder) │
│                                       │                               │
│                                       │  [Cancel]  [Confirm & Create]│
└───────────────────────────────────────┴───────────────────────────────┘
```

### Client-side behaviour (`new_fire.js`)
- On load: fetch `/api/year/<y>/overview_meta`. Use `geotransform`, `raster_W/H`, `overview_W/H` to build pixel↔map↔lat/lon converters.
- Mouse drag on canvas: draw a semi-transparent yellow rectangle. On `mouseup`, compute and write into the right panel: `x_min, y_min, x_max, y_max` (raster CRS, derived from the GeoTransform — `x_min = gt[0] + (px/ovr_W)*raster_W * gt[1]`, etc.) and `W, S, E, N` (via the EPSG:4326 corners — server-precomputed corners give the affine; client interpolates).
- **Drag interactions**: click-drag to draw new; click inside existing rect to drag-move it; click on edge handles to resize.
- Hover anywhere: show live cursor coords in the status bar (raster CRS + WGS84) so the user can sanity-check before confirming.
- Date placeholders: `<input type="text" placeholder="2023-03-01">` and `placeholder="2023-10-30"` populated from the meta JSON's `default_start` / `default_end`. Note: placeholders, not values — empty inputs use the default at submit time, but the user is shown what will happen.
- Form validation (client-side, defense-in-depth — server is authoritative):
  - Name: regex `/^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$/`, no `..` substring, not already used.
  - Dates: parseable `YYYY-MM-DD`, start ≤ end, start ≥ 2012-01-19, end ≤ today (server uses server-time).
  - Bbox: must be drawn (non-zero area), must intersect raster extent (always true since drawn on the overview), min size = 1 km on each side (warn but allow).
- "Confirm & Create" → POST `/api/fire/create`. On 202: redirect to `/` (fire list); the new fire shows `preparing` with sub-stage progress. On 4xx/5xx: render error in right panel, keep form state.

### Year selector
- Visible only when `len(rasters_by_year) > 1`.
- Admin-only: switching year here calls `/api/year/switch` and reloads `/new_fire` against the new active year.
- For non-admins on multi-year: the dropdown is disabled and shows "(admin only)".

---

## 7. New endpoints

All in `handlers/fire_list.py` (route registration in `handlers/base.py`'s router).

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/new_fire` | user+admin | Render `new_fire.html`. |
| GET | `/api/year/<y>/overview.png` | user+admin | Stream cached PNG. 404 if year invalid. |
| GET | `/api/year/<y>/overview_meta` | user+admin | Stream cached JSON. |
| POST | `/api/fire/create` | user+admin | Body: `{name, year, bbox_native:[x_min,y_min,x_max,y_max], start, end}`. Validates, creates `FireInfo`, enqueues VIIRS prepare worker. Returns 202 + `{name, status:'preparing'}`. |
| POST | `/api/fire/<name>/cancel_create` | creator+admin | Cancel an in-flight prepare. SIGTERM the worker subprocess group, rmtree cache, drop FireInfo. Returns 200 + `{status:'cancelled'}`. |
| POST | `/api/fire/<name>/clear_new` | user+admin | Flips `is_new=False`. Called by `/fire/<name>` page on first paint. |

### `/api/fire/create` server-side validation (in `validation.py`)
1. `name` — regex `^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$`, reject `..`, reject if already in `state.fires` (case-insensitive on existing names).
2. `year` — must be in `state.rasters_by_year`.
3. `bbox_native` — 4 finite floats, x_min < x_max, y_min < y_max, intersects the year's raster extent.
4. `start`, `end` — parseable `YYYY-MM-DD`. Empty → use the year's defaults from overview JSON. Then: start ≤ end, start ≥ `datetime.date(2012, 1, 19)`, end ≤ today. Reject otherwise with explicit message.
5. Return all validation errors in one response: `{errors: [{field, message}, ...]}` so the form can highlight every problem.

---

## 8. VIIRS prepare worker — `viirs_worker.py` (NEW)

### Threading model
- One module-level `ThreadingPool`-style queue, max parallelism = 1 (configurable via `--viirs_concurrent_jobs`, default 1). Rationale: each worker spawns its own 16 download workers and a shapify pool — running two prepare-jobs concurrently would saturate network and disk for no gain. The queue is fair (FIFO).
- Each job runs on a daemon thread spawned by the queue dispatcher. Thread lifecycle:

```python
def _viirs_worker(fire: FireInfo):
    cache_dir = os.path.join(state.output_root, '.web_cache', fire.fire_numbe)
    os.makedirs(cache_dir, exist_ok=True)
    fire.cache_dir = cache_dir
    fire.cancel_event = threading.Event()  # checked between stages

    try:
        _set_progress(fire, 'downloading_viirs', stage_idx=1, total=5)
        _laads_download(
            bbox_wgs84=fire.bbox_wgs84,
            start_dt=fire.viirs_start_date,
            end_dt=fire.viirs_end_date,
            save_dir=os.path.join(cache_dir, 'VNP14IMG'),
            token=state.laads_token,
            cancel_event=fire.cancel_event,
            workers=16,
        )
        if fire.cancel_event.is_set(): raise WorkerCancelled()

        _set_progress(fire, 'shapifying', stage_idx=2, total=5)
        from viirs.utils.shapify import process_file, find_nc_files
        # Use bbox filter so cross-AOI .nc fires don't bleed in.
        ref_raster = state.rasters_by_year[fire.fire_year]
        _shapify_dir(
            cache_dir, ref_raster=ref_raster,
            bbox=fire.bbox_wgs84, workers=8)
        if fire.cancel_event.is_set(): raise WorkerCancelled()

        _set_progress(fire, 'accumulating', stage_idx=3, total=5)
        from viirs.utils.accumulate import accumulate
        acc_paths = accumulate(
            shp_dir=cache_dir,
            start_str=fire.viirs_start_date.replace('-', ''),
            end_str=fire.viirs_end_date.replace('-', ''),
            reference_raster=ref_raster,
            output_dir=cache_dir,
            final_only=True,
            bbox=fire.bbox_native,    # source CRS — already matches
        )
        if not acc_paths:
            raise WorkerError('No VIIRS fire pixels found in bbox / date range.')
        if fire.cancel_event.is_set(): raise WorkerCancelled()

        _set_progress(fire, 'rasterizing', stage_idx=4, total=5)
        from viirs.utils.rasterize import rasterize_shapefile
        viirs_full = rasterize_shapefile(
            shp_path=acc_paths[-1],
            ref_image=ref_raster,
            output_dir=cache_dir,
            buffer_m=375.0,
        )
        # Sanity: bin must have at least one nonzero pixel.
        _verify_viirs_bin_nonzero(viirs_full)
        if fire.cancel_event.is_set(): raise WorkerCancelled()

        _set_progress(fire, 'cropping', stage_idx=5, total=5)
        # Tight crop bounds = bbox of fire pixels in viirs_full + padding.
        xmin, ymin, xmax, ymax = _tight_bounds_from_viirs_bin(
            viirs_full, padding=state.padding)
        crop_bin = os.path.join(cache_dir, f'{fire.fire_numbe}_crop.bin')
        from batch_fire_mapping.run_fire_mapping import crop_raster
        if not crop_raster(ref_raster, crop_bin, xmin, ymin, xmax, ymax):
            raise WorkerError('GDAL crop failed.')
        # Re-rasterize VIIRS to crop_bin extent so the hint is aligned.
        viirs_cropped = rasterize_shapefile(
            shp_path=acc_paths[-1],
            ref_image=crop_bin,
            output_dir=cache_dir,
            buffer_m=375.0,
        )
        # Generate previews from the crop.
        from .preview import generate_all_previews
        views = generate_all_previews(crop_bin, cache_dir, fire.fire_numbe)

        with state.lock:
            fire.crop_bin = crop_bin
            fire.viirs_bin = viirs_cropped
            fire.hint_bin = viirs_cropped
            fire.crop_w, fire.crop_h = _read_dims(crop_bin)
            fire.padding_used = state.padding
            fire.sample_size = max(state.min_samples, min(
                state.max_samples,
                int(round(fire.crop_w * fire.crop_h * state.sample_rate))))
            fire.acc_start = fire.viirs_start_date
            fire.acc_end = fire.viirs_end_date
            fire.perimeter_type = 'viirs'
            fire.available_views = views
            fire.fire_size_ha = _compute_viirs_area_ha(viirs_cropped)
            fire.status = FireStatus.READY
            fire.is_new = True
            fire.progress = {}
        _save_fire_state()
        _push_notification(
            kind='success',
            title='Fire prepared',
            body=f'{fire.fire_numbe} is ready to map.',
            fire=fire.fire_numbe,
        )

    except WorkerCancelled:
        with state.lock:
            fire.status = FireStatus.PENDING
            fire.progress = {}
        shutil.rmtree(cache_dir, ignore_errors=True)
        # FireInfo is dropped from state.fires by the cancel handler.

    except (WorkerError, Exception) as exc:
        with state.lock:
            fire.status = FireStatus.ERROR
            fire.error_msg = str(exc)
            fire.progress = {}
        _save_fire_state()
        _push_notification(
            kind='error', title='Prepare failed',
            body=f'{fire.fire_numbe}: {exc}', fire=fire.fire_numbe)

    finally:
        with state.viirs_jobs_lock:
            state.viirs_jobs.pop(fire.fire_numbe, None)
```

### Cancellation
- `fire.cancel_event` is checked between stages (already shown).
- The download stage uses `viirs/fp_gui/download_dialog.py:_download_worker` pattern: a `ThreadPoolExecutor(max_workers=16)` whose tasks check `cancel_event.is_set()` at the top. On cancel, `executor.shutdown(wait=False, cancel_futures=True)`.
- Shapify spawns subprocess. Track Popen in `state.viirs_jobs[fire_name]`. Cancel handler calls `os.killpg(proc.pid, SIGTERM)` (mirrors `app.py:_terminate_serial_proc`).

### Failure modes — explicit
- Token rejected by LAADS → `WorkerError('LAADS DAAC rejected the token; check /data/.tokens/laads.')`.
- Network failure mid-download → log per-day; only fail the worker if 0 .nc files made it.
- 0 fire pixels in AOI → `WorkerError('No VIIRS fire pixels in bbox during the chosen date range.')`. Status flips to ERROR; user can either delete the fire or re-create with a different bbox/date range.
- VIIRS bin all zeros after rasterize → same error message.
- Disk full → propagate OSError.
- Reference raster missing CRS → already raised by `shapify.get_crs_from_raster`.

### Persistence
- Worker state is reconstructable: `cache_dir` content tells you where you got. On server restart mid-prepare, `init_fires_from_disk` flips the in-flight fire to ERROR (no worker to re-attach to) — user re-creates or restarts the prepare via a new endpoint `POST /api/fire/<name>/retry_create`. (Out of scope for v1 — note as future work.)

---

## 9. Tight-crop derivation — in `viirs_worker.py`

```python
def _tight_bounds_from_viirs_bin(
    viirs_bin: str, padding: float
) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in raster CRS, tightened to the
    bbox of nonzero pixels in viirs_bin and expanded by
    padding * max_dim_in_pixels (mirrors prepare.py:122-128).

    Raises WorkerError if the bin has zero nonzero pixels.
    """
    ds = gdal.Open(viirs_bin, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    ds = None

    nz = np.nonzero(arr)
    if nz[0].size == 0:
        raise WorkerError('VIIRS hint has no fire pixels.')
    py_lo, py_hi = int(nz[0].min()), int(nz[0].max())
    px_lo, px_hi = int(nz[1].min()), int(nz[1].max())

    fire_max_dim = max(px_hi - px_lo, py_hi - py_lo)
    p = max(1, int(round(padding * fire_max_dim)))
    px_lo = max(0, px_lo - p)
    px_hi = min(W - 1, px_hi + p)
    py_lo = max(0, py_lo - p)
    py_hi = min(H - 1, py_hi + p)

    xmin = gt[0] + px_lo * gt[1]
    xmax = gt[0] + px_hi * gt[1]
    ymax = gt[3] + py_lo * gt[5]    # gt[5] is negative
    ymin = gt[3] + py_hi * gt[5]
    return xmin, ymin, xmax, ymax
```

This intentionally mirrors `prepare.py:117-128` so the resulting crop has identical shape/feel to a polygon-driven crop in `_web`.

---

## 10. Tests — `tests/`

All tests use `pytest`. Synthetic fixtures live in `conftest.py`.

### `conftest.py` fixtures
- `tmp_raster_3x3km`: a 100×100-pixel ENVI .bin/.hdr in EPSG:32610 (UTM 10N) covering a known area in BC. Three bands, each filled with deterministic gradients. `mtime`, `size` exposed.
- `tmp_viirs_bin`: same shape as crop, with 5 nonzero pixels at known coords (used for tight-crop testing).
- `mock_laads_token`: monkeypatch a fake `/data/.tokens/laads` to a tmp path.
- `mock_laads_sync`: monkeypatch `viirs.utils.laads_data_download_v2.sync` to write a known fixture .nc to the destination dir without hitting the network.

### `test_overview.py`
- `test_overview_dimensions_clip_to_max_dim` — overview PNG longest edge ≤ 2000 px.
- `test_overview_sidecar_json_round_trip` — load JSON, verify pixel→map round-trips through GeoTransform and inverse exactly for the corners.
- `test_overview_pre_post_band_detection` — synthetic raster with `pre_B12, pre_B11, pre_B9, post_B12, post_B11, post_B9` band names → overview uses post group.
- `test_overview_default_dates_match_year` — for `pgfc_2023.bin`, sidecar contains `default_start='2023-03-01'`, `default_end='2023-10-30'`.

### `test_overview_caching.py`
- `test_overview_skipped_when_fresh` — second call with same raster mtime/size does not regenerate (mock the GDAL read and assert call count = 0).
- `test_overview_regenerated_when_raster_changes` — touch raster's mtime, second call regenerates.

### `test_bbox_validation.py`
- `test_bbox_outside_raster_extent_rejected`
- `test_bbox_zero_area_rejected`
- `test_bbox_partially_outside_clipped` — bbox extending past raster bounds is clipped to extent and accepted.
- `test_bbox_non_finite_rejected` — NaN / inf in any coord rejected.

### `test_date_defaults.py`
- `test_default_start_is_march_1_of_raster_year` — for `pgfc_2023.bin`, default `start='2023-03-01'`.
- `test_default_end_is_october_30_of_raster_year` — for `pgfc_2023.bin`, default `end='2023-10-30'`.
- `test_defaults_match_active_year_when_multi_raster` — overview JSON for each year contains its own year's defaults.
- `test_empty_dates_in_create_request_use_defaults` — POST with `start=''` and `end=''` resolves to the year's defaults.

### `test_date_validation.py`
- `test_unparseable_date_rejected` — `'2023-13-01'`, `'not-a-date'`.
- `test_end_before_start_rejected`.
- `test_start_before_2012_01_19_rejected` — VNP14IMG availability lower bound.
- `test_end_in_future_rejected` — end > today (using `freezegun` or monkeypatching `datetime.date.today`).
- `test_start_equals_end_accepted` — single-day range valid.
- `test_iso_format_only` — reject `'2023/03/01'`, accept `'2023-03-01'`.

### `test_fire_name_validation.py`
- `test_valid_names` — `'C12345'`, `'My Fire 2023'`, `'fire.A'`.
- `test_path_traversal_rejected` — `'../foo'`, `'..foo'`, `'foo/bar'`, `'foo\\bar'`.
- `test_empty_name_rejected`.
- `test_too_long_name_rejected` — > 64 chars.
- `test_duplicate_name_rejected_case_insensitive` — `'fire1'` and `'FIRE1'` collide.
- `test_leading_punctuation_rejected` — `'-foo'`, `'.foo'` (regex starts with `[A-Za-z0-9]`).

### `test_tight_crop.py`
- `test_tight_crop_one_pixel_fire` — VIIRS bin with 1 pixel at known coords; crop bounds match expected (1×1 + padding).
- `test_tight_crop_padding_clamped_to_raster_extent` — fire near edge → crop clipped to raster.
- `test_tight_crop_zero_pixels_raises` — VIIRS bin all zero raises `WorkerError`.
- `test_tight_crop_padding_zero_returns_exact_bounds` — `padding=0` → exact pixel bbox.

### `test_fire_create_endpoint.py`
- `test_create_returns_202_and_appears_in_list` — happy path, mocking the LAADS sync.
- `test_create_with_invalid_name_returns_400` — error response body has structured `errors[]`.
- `test_create_with_year_not_in_registry_returns_400`.
- `test_create_dispatches_worker_to_queue` — assert worker thread started with the expected fire.
- `test_concurrent_create_with_same_name_second_returns_409`.
- `test_admin_and_user_can_both_create` — both roles allowed.

### `test_viirs_worker_cancel.py`
- `test_cancel_during_download_aborts_and_cleans_cache` — start a worker with mocked slow download, hit `/cancel_create`, assert: `state.fires` no longer contains the fire, `cache_dir` is removed, no zombie threads.
- `test_cancel_during_shapify_kills_subprocess` — mock shapify subprocess that sleeps; cancel sends SIGTERM; subprocess terminates.
- `test_cancel_after_ready_no_op` — once status is READY, `/cancel_create` returns 409 (use `/api/fire/<name>/remove` to delete an accepted fire instead).

### Integration test (deferred to manual QA)
- Real LAADS DAAC end-to-end against a small AOI + 2-day window. Requires real token; not in CI.

---

## 11. Implementation order (concrete steps for the next agent)

1. **Bootstrap** — `cp -r batch_fire_mapping_web batch_fire_mapping_viirs_web`. Update package name in any docstrings (don't rewrite history; just do find-replace on `batch_fire_mapping_web` → `batch_fire_mapping_viirs_web` where it appears in module docstrings or imports — being careful with the `batch_fire_mapping` parent package which is shared).
2. **`overview.py`** — implement + write `test_overview.py`, `test_overview_caching.py`. Run those two test files only.
3. **`state.py` edits** — drop polygon fields, add bbox/is_new/etc., replace `init_fires_from_gdf` with `init_fires_from_disk`. Keep `FireInfo` backward-compatible enough that `persistence.py` still serializes/loads.
4. **`__main__.py` edits** — drop polygon arg, add overview generation, add token check. Confirm package `python -m batch_fire_mapping_viirs_web --help` shows the new CLI.
5. **`validation.py`** — add `_validate_bbox`, `_validate_date`, `_validate_fire_name`. Write+run `test_bbox_validation.py`, `test_date_defaults.py`, `test_date_validation.py`, `test_fire_name_validation.py`.
6. **`viirs_worker.py`** — implement worker + `_tight_bounds_from_viirs_bin`. Write+run `test_tight_crop.py`, `test_viirs_worker_cancel.py`.
7. **`prepare.py` edits** — `_prepare_fire_sync` becomes `_reprepare_fire_sync` (called only when re-preparing an existing fire after padding change or cache eviction). Source crop bounds from `fire.viirs_bin`/`fire.bbox_native` instead of polygon.
8. **`persistence.py` edits** — add new fields to fire-state YAML, drop polygon refs from `_switch_year`. `_switch_year` now just swaps active year + reloads overview meta (no polygon reload).
9. **`handlers/fire_list.py`** + **`handlers/fire.py`** — add new endpoints (§7). Drop polygon-derived fields from `handle_api_fires`. Add `is_new` to the JSON. Write+run `test_fire_create_endpoint.py`.
10. **`templates/new_fire.html`** + **`static/new_fire.js`** + **`static/style.css` additions** — build the bbox-drawing UI. Manual smoke test (no automated UI test in v1).
11. **`templates/fire_list.html`** — add "+ New Fire" button (links to `/new_fire`), `new` badge in fire-number cell (clickable Open also POSTs `/clear_new`), sub-stage display in status cell when status==`preparing`.
12. **End-to-end manual QA** — start server, draw a bbox, create a fire with a small 1-day window over a known fire area, watch it progress through the stages, confirm it lands in READY, open the fire and run mapping.

---

## 12. Robustness considerations (the "extremely robust" bar)

| Concern | Mitigation |
|---|---|
| 100 GB raster overflows RAM | `ReadAsArray(buf_xsize=2000, buf_ysize=2000)` reads stride-decimated; ~50 MB peak. |
| Overview generation slow first time | Acceptable (one-time per raster). Print per-band progress to stdout. Cache result keyed by mtime+size — second startup is instant. |
| User browser drops mid-prepare | Worker is server-side, decoupled from session. State persists in `fire_state.yaml`. User reopens fire list, sees status `preparing` with sub-stage. |
| Server crash mid-prepare | On restart, `init_fires_from_disk` finds the fire with status `preparing` but no live worker → flips to `ERROR` with message "interrupted; retry create". |
| Two users name the same fire | `_validate_fire_name` checks under `state.lock`; second one gets 409. |
| Concurrent prepare jobs DOSing LAADS | Module-level dispatch queue with `max_concurrent_jobs=1` (configurable). FIFO. Status shows queue position. |
| Disk fill from runaway downloads | Each fire's cache lives under existing `cache_retention.py` sweeper's purview. Add a per-fire VIIRS quota (e.g. 5 GB) check before kicking off shapify; abort and warn if exceeded. |
| Bbox spans antimeridian | LAADS DAAC URL doesn't handle this cleanly. Reject in validator if `east < west` after WGS84 conversion. |
| Bbox crosses raster nodata regions only | Worker proceeds, accumulates 0 fires, raises `WorkerError`. Operator-actionable. |
| User picks date range with no satellite passes | Same as above — 0 fire pixels → ERROR with explicit message. |
| Token leaks via response body | Token is module-private; never echoed. URL with bbox+date is logged but not the token (already safe in `viirs/fp_gui/download_dialog.py`). |
| Path traversal in fire name | Regex + explicit `..` substring check + os.path.join via validated name only. |
| Race: cancel arrives just as worker flips to READY | Cancel handler reads status under `state.lock`; if READY, returns 409 with "fire already prepared; use /remove to delete". |
| Overview PNG sidecar JSON gets corrupted | If JSON parse fails, treat as not-fresh and regenerate. Write-temp + rename keeps the on-disk file atomically valid. |

---

## 13. Out of scope for v1 (note for future)

- Pan/zoom on overview (use Leaflet/MapLibre + tile server). v1 is single static PNG with rectangle drawer.
- Per-user LAADS tokens.
- Renaming a created fire.
- Bulk-create from a CSV of bboxes.
- Sharing downloaded `.nc` files between fires with overlapping bbox/date (cache-key by `(bbox, date)` and dedupe).
- "Retry create" endpoint to resume a failed prepare from the last completed stage.
- Animated VIIRS playback on the fire page (port from `viirs/fp_gui/fire_animation_controller.py`).
- Server-side validation of the LAADS token at startup (currently we trust the file).

---

## 14. Acceptance criteria

The implementing agent's work is "done" when:

1. `python -m batch_fire_mapping_viirs_web --rasters pgfc_2022.bin pgfc_2023.bin --out_root /tmp/test --insecure_no_auth` starts cleanly and shows an empty fire list.
2. `/new_fire` renders the overview PNG, supports rectangle drawing with live coord readout, accepts a name + dates, and POSTs `/api/fire/create` returning 202.
3. The new fire appears in the list with status `preparing`, advances through `downloading_viirs → shapifying → accumulating → rasterizing → cropping → ready`.
4. Once READY, fire shows `new` badge. Opening it clears the badge and shows the standard mapping page.
5. Mapping the fire (single-shot or with-settings) produces the same output structure as `_web`.
6. Accepting promotes outputs to `<out_root>/<fire_name>/` exactly like `_web`.
7. All tests in §10 pass: `pytest tests/`.
8. No references to `gdf`, `viirs_gdf`, `polygon_file`, `polygon_gdf_raw` survive in the new package's source tree (`grep -r` clean).
9. Manual QA: a 2-day download over a small known-fire bbox completes end-to-end against real LAADS DAAC.

---

*End of plan. The next agent should read §0-§14 in order and implement following §11.*
