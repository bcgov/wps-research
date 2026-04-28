This version of the web application is for operational use. It seeds fire mapping results from NRT (accumulated) VIIRS data. The other version is for historical research use and seeds fire mapping results from historical fire perimeters:

* [batch_fire_mapping_web](https://github.com/bcgov/wps-research/tree/master/data/bill/batch_fire_mapping_web)

# batch_fire_mapping_viirs_web

*Sibling of `batch_fire_mapping_web`. Same downstream ML pipeline; the
front of the pipeline is replaced with a user-defined bbox + VIIRS
download workflow instead of a polygon shapefile.*

Interactive web interface where analysts **draw a bounding box on the
year's overview, name a fire, pick a date range, and submit**. The
server downloads VIIRS active-fire data for that bbox + date window,
accumulates and rasterizes the hot pixels, derives a tight crop from
the actual fire pixels, and seeds the existing GPU mapping pipeline.
Once the prepare worker reaches `READY`, every downstream feature
(map / serial sweep / rebrush / accept / batch / multi-year) works
exactly as in the polygon-driven sibling.

---

## What changed vs. `batch_fire_mapping_web`

| | `_web` (polygon) | `_viirs_web` (this package) |
|---|---|---|
| Fire source | Pre-curated polygon shapefile | User draws bbox in `/new_fire` |
| Fire identity | `FIRE_NUMBE` from polygon attribute | User-supplied name (validated) |
| VIIRS download | All-years, at startup, blocking | **Per-year, at startup, blocking (idempotent)** — bootstrap also builds a per-year `year_index.gpkg` so fire creation runs a single bbox-pushdown read instead of walking the per-granule shp tree |
| Crop bounds | Polygon geometry intersection + padding | bbox of nonzero VIIRS pixels + padding |
| `--polygon_file` | Required positional | **Removed** |
| `--perimeter_mode` | `viirs` / `traditional` | **Removed** (always VIIRS) |
| LAADS token | Optional (`--skip_download`) | **Required** at `/data/.tokens/laads` |
| New page | n/a | `/new_fire` (canvas overlay + form) |
| New module | n/a | `overview.py`, `viirs_worker.py` |
| Multi-year | Yes | Yes |

The downstream mapping / brush / accept / serial / rebrush / batch
flows are **unchanged**. See the source package's
[README](../batch_fire_mapping_web/README.md) for that material; this
README only documents the new front-of-pipeline.

---

## Quick start

```bash
# from data/bill/
./run_fire_viirs_web.sh
```

…where `run_fire_viirs_web.sh` invokes:

```bash
python3 -m batch_fire_mapping_viirs_web \
    --rasters /ram/new_cloudfree/pgfc_2023.bin \
              /ram/new_cloudfree/2024_pgfc.bin \
              /ram/new_cloudfree/2025_pgfc.bin \
    --out_root ./fire_mapping_results_viirs \
    --laads_token_file /data/.tokens/laads \
    --user_password 888888 \
    --admin_password ashlin
```

Open `http://localhost:8765` and click **+ New Fire**.

The first launch generates one overview PNG per raster (memory-bounded
GDAL stride read; a 100 GB raster takes ~30 s). Subsequent launches
use the cached PNGs (cache key = raster mtime + size).

---

## Requirements

In addition to everything `batch_fire_mapping_web` needs:

- **LAADS DAAC token file** at `/data/.tokens/laads` (one line). Get
  one at <https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys>.
  Override the path with `--laads_token_file`.
- **Network egress** to `ladsweb.modaps.eosdis.nasa.gov` from the
  server (the per-fire download fetches VNP14IMG `.nc` granules).
- **`viirs.utils.shapify`**, **`viirs.utils.accumulate`**, and
  **`viirs.utils.rasterize`** importable from
  `data/bill/viirs/` (already present in this repo).
- **`netCDF4`** Python module (used by `shapify` to read VNP14IMG).

GDAL / NumPy / GeoPandas / PyYAML / matplotlib / scipy come from the
sibling package's existing requirements.

---

## The `/new_fire` flow

Two-pane layout:

**Left**: the year's overview PNG underneath a transparent canvas. The
analyst click-drags to draw a yellow rectangle. Clicking inside an
existing rectangle drag-moves it; **Clear bbox** resets. Hovering
shows live cursor coords (raster CRS + WGS84) in the toolbar.

**Right**: form fields:

- **Name** — must match `^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$`, no
  `..` substring, no `/` or `\`. Uniqueness is checked
  case-insensitively against existing fires.
- **Bounding Box (raster CRS / WGS84)** — read-only readouts that
  update as you drag.
- **Start / End** date inputs — placeholders are
  `<year>-03-01` and `<year>-10-30` (from the overview JSON's
  `default_start` / `default_end`). **Empty fields fall through to
  the placeholders** at submit time. Constraints:
  - parseable as `YYYY-MM-DD`
  - start ≤ end
  - start ≥ 2012-01-19 (VNP14IMG availability)
  - end ≤ today (server time)

**Confirm & Create** POSTs `/api/fire/create`. On 202, the page
redirects to `/`; the new fire shows up as `preparing` with live
sub-stage progress. Validation errors come back as
`{errors: [{field, message}, ...]}` and are rendered in the right
panel without losing form state.

The **year selector** is visible only on multi-year deployments and
disabled for non-admins (year-switch is admin-only).

---

## The VIIRS prepare worker

Download + shapify + **index build** happen **once per year at server boot**
in `year_viirs.bootstrap_all_years` (idempotent — pre-existing `.nc`,
`.shp`, and `year_index.gpkg` are reused). Each year's full raster
footprint is downloaded for the seasonal window `<year>-03-01` to
`<year>-10-30` (or to today if the upper bound is in the future) into
`<output_root_for_year>/_year_viirs/VNP14IMG/<year>/<jday>/`.

After shapify, `year_viirs.build_year_index` consolidates every
per-granule `*.shp` under that tree into a single GeoPackage at
`<output_root_for_year>/_year_viirs/year_index.gpkg` (layer `viirs`, GPKG
R-tree on geometry, text `det_dt` column in compact `YYYYMMDDHHMM` so date
filters reduce to lexicographic comparisons regardless of GDAL/SQLite type
coercion). A sidecar `year_index.gpkg.manifest` records the source `.shp`
count; the index is rebuilt only when the count or any source `.shp`'s
mtime advances. Atomic write via `.tmp.gpkg → rename`. Per-fire prepare
queries this single file with bbox pushdown instead of opening hundreds
of per-granule shapefiles.

Per-fire submitted via `/api/fire/create` are dispatched to a module-level
FIFO queue (`viirs_worker._dispatch_queue`) with `--viirs_concurrent_jobs`
parallel workers (default 1). Each fire walks **two** stages — no
download, no shapify (already done at boot):

| Stage | What happens | Cancellable |
|---|---|---|
| `accumulating` | **Fast path** (default): `viirs_worker._fast_accumulate_from_index` runs a single bbox-pushdown read against `year_index.gpkg`, date-filters in pandas, and writes the per-fire `VIIRS_VNP14IMG_<startdt>_<enddt>.shp` the rasterize step expects. **Slow fallback** (when the index is missing — older deployments, build failure): `viirs.utils.accumulate(...)` walks the per-granule shapefile tree with bbox filter at read time. Both paths are interchangeable from the rest of the pipeline's perspective. | Between stages. |
| `cropping` | `_tight_bounds_from_shapefile` reads `gdf.total_bounds` from the cumulative shapefile (no full-extent rasterize), expands by `_RASTERIZE_BUFFER_M` and `padding * max_dim`. `crop_raster` produces `<NAME>_crop.bin`; VIIRS is rasterized onto the cropped extent so the hint aligns to the crop's grid. Previews are generated, including a green-tinted hint overlay on `previews/post.png`. | n/a (last stage). |

On success the fire flips to `READY`, `is_new=True` (drives the "new"
badge), `fire.crop_bin` / `fire.viirs_bin` / `fire.hint_bin` /
`fire.acc_start` / `fire.acc_end` are populated, and a success toast
goes out. From here the rest of the pipeline (single-shot map, serial
sweep, accept, …) is unchanged from the sibling.

**Failure modes** — surfaced to `fire.status = ERROR` with an
actionable `fire.error_msg`:

- `LAADS DAAC rejected the token` (auth failure mid-download)
- `No VIIRS fire pixels in bbox during the chosen date range.`
- `shapify exited with code N`
- `GDAL crop failed.`
- Network failures inside one day are logged but only fail the worker
  when the entire shapify run produces nothing.

**Cancel mid-prepare**: `POST /api/fire/<NAME>/cancel_create` sets
`fire.cancel_event`, SIGTERMs any live subprocess, waits up to ~10 s
for the worker to tear down, then **drops the FireInfo from
`state.fires` and `rmtree`s the cache_dir**. Cancelling is destructive
on purpose — the operator chose to abandon the fire.

**Server crash mid-prepare**: on next boot, `init_fires_from_disk`
finds the orphan `.web_cache/<NAME>/` with no live worker and flips
the fire to `ERROR` with message `interrupted; retry create`. The
operator deletes via `/api/fire/<NAME>/remove` (or via the admin
dashboard) and re-creates.

---

## CLI reference

```
python -m batch_fire_mapping_viirs_web \
    --rasters R1 [R2 ...] --out_root DIR [options]
```

| Argument | Default | Purpose |
|---|---|---|
| `--rasters` | (required) | Sentinel-2 ENVI `.bin` rasters; each filename must contain a unique 4-digit year (1970..now+1). |
| `--out_root` | (required) | Mother dir; per-year results land in `<out_root>/<raster_stem>_mapping_results/`. |
| `--year` | newest | Initial active year. Falls back to `<out_root>/active_year.yaml` then to the newest year. |
| `--padding` | `0.1` | Tight-crop padding fraction (`p * max(width, height)` pixels). |
| `--sample_rate` | `0.05` | Default sampling rate for the ML pipeline. |
| `--min_samples` | `500` | Lower bound on per-fire sample size. |
| `--max_samples` | `30000` | Upper bound on per-fire sample size. |
| `--viirs_concurrent_jobs` | `1` | How many prepare jobs run in parallel; the rest queue FIFO. |
| `--viirs_download_workers` | `16` | Per-job parallel LAADS download workers. |
| `--viirs_shapify_workers` | `8` | Per-job parallel shapify workers. |
| `--host`, `--port` | `0.0.0.0:8765` | Server bind address. |
| `--admin_password`, `--user_password` | (required) | Set both; or use `--insecure_no_auth` for isolated environments. |
| `--insecure_no_auth` | off | Disable all auth + IP gating. **Do not use on a multi-user host.** |
| `--trust_proxy` | off | Honor `X-Forwarded-For` for client IP (only behind a trusted reverse proxy). |
| `--laads_token_file` | `/data/.tokens/laads` | Path to the LAADS DAAC token file (one line). |

Run with `--help` for the canonical list.

---

## HTTP API — what's new

The full API is the sibling package's API plus these endpoints:

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/new_fire` | user+admin | Bbox-drawing + form (HTML). |
| GET | `/api/year/<y>/overview.png` | user+admin | Per-year overview PNG (cached on disk). |
| GET | `/api/year/<y>/overview_meta` | user+admin | Sidecar JSON: `geotransform`, `crs_wkt`, `raster_W/H`, `overview_W/H`, `extent_native`, `extent_wgs84`, `default_start`, `default_end`, `cache_key`. |
| POST | `/api/fire/create` | user+admin | Body `{name, year, bbox_native:[xmin,ymin,xmax,ymax], start, end}`. Validates, creates `FireInfo`, enqueues prepare worker. **202** on success with `{name, status:'preparing'}`. **400** with `{errors: [{field, message}, ...]}` on validation failure. **409** if the name collides under a race. |
| POST | `/api/fire/preview_hint` | user+admin | Body `{year, bbox_native, start, end}`. Accumulates VIIRS for the bbox + dates from the year-wide shared shp dir, rasterises onto the user's bbox, generates pre-classification preview PNGs, returns `{preview_id, area_ha, views: {hint, post}}`. |
| GET | `/api/fire/preview_hint/<preview_id>/<view>.png` | user+admin | The PNG referenced by `preview_hint`. Reaped after ~30 min. |
| POST | `/api/fire/<name>/cancel_create` | user+admin | Cancel an in-flight prepare. SIGTERMs subprocesses, rmtrees the cache, drops the FireInfo. **409** if the fire is already past `PREPARING` (e.g. `READY` — use `/remove` instead). |
| POST | `/api/fire/<name>/clear_new` | user+admin | Flip `is_new=False`. Called fire-and-forget when the user clicks **Open** in the fire list. |

`/api/fires` exposes additional fields per fire:
`is_new`, `error_msg`, `sub_stage`, `sub_stage_idx`, `sub_stage_total`,
`sub_stage_detail`. The fire list renders these as a "new" badge, an
inline error line, and a sub-stage progress line under the status
pill.

---

## Output structure

```
<out_root>/
├── active_year.yaml
├── sessions.yaml
├── access_control.yaml
├── notes.yaml, notifications.yaml, stage_timings.yaml,
│   cache_retention.yaml, cancel_audit.yaml
├── .web_cache/
│   └── _overviews/
│       ├── pgfc_2023.png    # per-year overview PNG (cache_key = mtime + size)
│       ├── pgfc_2023.json   # sidecar metadata (consumed by /new_fire)
│       └── ...
├── pgfc_2023_mapping_results/
│   ├── fire_state.yaml
│   ├── accepted_params.csv
│   ├── _year_viirs/                # year-wide VIIRS data (built once at boot)
│   │   ├── VNP14IMG/<year>/<jday>/*.nc + *.shp  # per-granule raw + shapified
│   │   ├── year_index.gpkg                       # consolidated R-tree-indexed GPKG
│   │   └── year_index.gpkg.manifest              # shp-count freshness sidecar
│   ├── .web_cache/
│   │   └── <NAME>/                 # per-fire prepare cache
│   │       ├── VIIRS_VNP14IMG_<...>.shp/.dbf/.shx/.prj  # cumulative (from fast path or accumulate)
│   │       ├── _viirs_crop/VIIRS_VNP14IMG_<...>.bin   # crop-aligned rasterize
│   │       ├── <NAME>_crop.bin                    # ENVI cropped raster
│   │       ├── <NAME>_serial_<N>_classified.bin   # gallery entries
│   │       └── previews/{pre,post,hint,result}.png
│   └── <NAME>/                      # promoted on accept (canonical)
│       ├── <NAME>_crop.bin_classified.bin
│       ├── <NAME>.shp / .kml
│       ├── <NAME>_comparison.png
│       ├── <NAME>_brush_comparison.png
│       └── <NAME>_params.yaml       # has a new `bbox:` section + `accumulation:` dates
└── 2024_pgfc_mapping_results/
    └── ...
```

`<NAME>_params.yaml` carries an extra section so an accepted fire can
round-trip back through `init_fires_from_disk` on next boot:

```yaml
fire:
  fire_numbe: NAME
  fire_date: 2023-08-15
  ...
bbox:
  native: [501000.0, 5497000.0, 502500.0, 5499000.0]   # raster CRS
  wgs84:  [-123.92, 49.61, -123.88, 49.65]              # W, S, E, N
accumulation:
  start_date: 2023-07-01
  end_date:   2023-08-15
```

---

## File overview — what's new or changed

Files that exist only in this package:

| File | Purpose |
|---|---|
| `overview.py` | Per-year overview PNG + sidecar JSON generator. `generate_overview(raster, png_path, json_path, max_dim=2000)` does a memory-bounded GDAL `ReadAsArray(buf_xsize=, buf_ysize=)` so a 100 GB raster reads at ~50 MB peak. Atomic write (tmp + fsync + rename + parent-dir fsync). Reuses `preview.detect_band_groups` to prefer the post group. `overview_is_fresh` and `ensure_overview` implement the cache by `(st_mtime_ns, st_size)`. The sidecar JSON is consumed client-side in `new_fire.js` for pixel ↔ CRS ↔ WGS84 math. |
| `year_viirs.py` | Year-wide VIIRS bootstrap (download + shapify + index) run once at server boot. `bootstrap_all_years(state)` iterates every (year, raster) and calls `bootstrap_year` → `download_year` (per-day LAADS pulls into `_year_viirs/VNP14IMG/<year>/<jday>/`, parallel via `--viirs_download_workers`) → `shapify_year` (parallel via `--viirs_shapify_workers`, skips already-shapified granules) → `build_year_index`. `build_year_index` consolidates every `*.shp` under that tree into a single GeoPackage `year_index.gpkg` (layer `viirs`, GPKG R-tree on geometry, text `det_dt` column in compact `YYYYMMDDHHMM` for lexicographic compare). Idempotent via a `.manifest` recording shp count + per-source-shp mtime check. Atomic write (`.tmp.gpkg → rename`). Drops the `FID` column geopandas pulls off shapefiles so it doesn't collide on the GPKG primary key. |
| `viirs_worker.py` | The 2-stage prepare worker (`accumulating`, `cropping`). `submit_fire(fire)` enqueues; a daemon dispatcher pulls FIFO and calls `_viirs_worker(fire)` which walks `accumulate_for_fire → _tight_bounds_from_shapefile → crop_raster → rasterize_shapefile (onto crop) → previews`. `accumulate_for_fire` first checks for a matching seeded shapefile (`_seeded_shp_matches_fire`), then tries the **fast path** (`_fast_accumulate_from_index`) which bbox-pushdowns into `year_index.gpkg`, date-filters in pandas, and writes a per-fire cumulative `VIIRS_VNP14IMG_<startdt>_<enddt>.shp` matching the slow path's column contract; falls back to `viirs.utils.accumulate(...)` (per-granule walk with bbox-filtered reads) when the index is missing or unreadable. `cancel_fire(fire)` sets `fire.cancel_event` and SIGTERMs any live subprocess. `_tight_bounds_from_shapefile` reads `gdf.total_bounds` directly (no full-extent rasterize); `_tight_bounds_from_viirs_bin` is retained for tests / re-prepare. Cancellation, subprocess group kills, and progress snapshots all go through this module. |
| `templates/new_fire.html` | The bbox-drawing page. Static `<img>` overview underneath an HTML5 `<canvas>` overlay. Embeds a small JSON config block for `new_fire.js`. |
| `static/new_fire.js` | Canvas drag handler (create / move), pixel ↔ raster CRS ↔ WGS84 conversion off the overview JSON, live cursor readout, form validation, POST to `/api/fire/create`. No external libraries. |
| `tests/conftest.py` | Synthetic raster + VIIRS bin fixtures (UTM 10N, 30 m/pixel). |
| `tests/test_overview.py`, `test_overview_caching.py` | 12 tests for overview generation + cache-key freshness. |
| `tests/test_bbox_validation.py` | 9 tests for bbox geometry + non-finite + clipping rules. |
| `tests/test_date_defaults.py`, `test_date_validation.py` | 14 tests for placeholder fall-through, ISO parsing, VNP14IMG lower bound, future-date rejection. |
| `tests/test_fire_name_validation.py` | 16 tests for path-traversal / case-insensitive uniqueness / leading-punctuation rules. |
| `tests/test_tight_crop.py` | 5 tests for the bbox-of-nonzero-pixels + padding math. |
| `tests/test_viirs_worker_cancel.py` | 4 tests for cooperative cancel during download / subprocess kill mid-shapify / cache cleanup / idempotent no-op. |
| `tests/test_fire_create_endpoint.py` | 7 tests exercising the validation-and-enqueue path of `/api/fire/create`. |
| `tests/test_year_index_fast_path.py` | 6 tests for `build_year_index` (creation, idempotence, rebuild on new shp) and `_fast_accumulate_from_index` (bbox+date filter, empty-result raise, missing-index fallback). |

Files changed from the sibling:

| File | Change |
|---|---|
| `__main__.py` | Drops `polygon_file` positional, `--perimeter_mode`, `--skip_download`, `--shapify_workers`. Adds `--laads_token_file`, `--viirs_concurrent_jobs`, `--viirs_download_workers`, `--viirs_shapify_workers`. Pre-startup loads the LAADS token and generates per-year overviews (via `overview.ensure_overview`) before booting the server. Calls `app_state.init_fires_from_disk()` instead of `init_fires_from_gdf()`. |
| `state.py` | `FireInfo` adds `bbox_native`, `bbox_wgs84`, `viirs_start_date`, `viirs_end_date`, `is_new`, `cancel_event`. `AppState` drops `gdf`, `viirs_gdf`, `polygon_file`, `polygon_gdf_raw`, `viirs_shp_dir`, `viirs_shp_dirs_by_year`; adds `overview_png_by_year`, `overview_meta_by_year`, `laads_token`, `viirs_jobs`, `viirs_subprocs`, `viirs_concurrent_jobs`, `viirs_download_workers`, `viirs_shapify_workers`. New `init_fires_from_disk()` rebuilds the registry from `<output_root>/<NAME>/<NAME>_params.yaml` (accepted) and `<output_root>/.web_cache/<NAME>/` (in-flight; orphaned mid-prepare entries flip to ERROR). |
| `app.py` | Adds `from . import viirs_worker as _viirs_worker` and a `_viirs_worker.init(app_state, _save_fire_state, _push_notification)` call at the end of `init_app`. No other changes — every other sibling module's wiring is unchanged. |
| `prepare.py` | `_prepare_fire_sync` is the **re-prepare** path now: when the operator changes padding or wipes the cache, it locates the cached full-extent VIIRS bin, re-derives tight bounds via `viirs_worker._tight_bounds_from_viirs_bin`, re-crops the reference raster, and re-rasterizes the cumulative VIIRS shapefile onto the new crop frame. The polygon-perimeter rasterize and polygon-VIIRS-intersection blocks are deleted. Initial prepare lives entirely in `viirs_worker._viirs_worker`. `_accept_fire_sync` is unchanged except for the dropped polygon refs. |
| `persistence.py` | `_save_fire_state` adds `bbox_native`, `bbox_wgs84`, `viirs_start_date`, `viirs_end_date`, `is_new`, `error_msg`, `fire_year`, `fire_size_ha`, `fire_date` to the persisted YAML. `_load_fire_state` synthesizes a new `FireInfo` for fires that exist only in `fire_state.yaml` (e.g. hidden + cache wiped). `_switch_year` is simplified to just swap the active-year handles + reload from disk — no polygon re-projection / spatial-filter pass. |
| `validation.py` | Adds `_validate_fire_name` (regex + traversal + case-insensitive uniqueness), `_validate_date` (strict ISO YYYY-MM-DD), `_validate_date_range` (empty-string default fallthrough, `start ≤ end`, `start ≥ 2012-01-19`, `end ≤ today`), `_validate_bbox` (4 finite floats, x/y ordering, raster-extent overlap, clip-to-extent return). Existing `_validate_param` / `_validate_embed_bands` are unchanged. |
| `handlers/base.py` | Registers six new routes (3 GET + 3 POST) for the bbox-drawing flow. |
| `handlers/fire_list.py` | Adds `handle_new_fire_page`, `handle_api_year_overview_png`, `handle_api_year_overview_meta`, `handle_api_fire_create`, `handle_api_fire_cancel_create`, `handle_api_fire_clear_new`. Extends `handle_api_fires` with `is_new` / `error_msg` / `sub_stage*` keys. Drops the `state.polygon_file` reference in the home-page render. |
| `templates/fire_list.html` | "+ New Fire" button in the header (links to `/new_fire`). "new" badge in the fire-number cell. Sub-stage display in the status cell when status==`preparing` (e.g. *downloading_viirs (1/5) — 3 / 5 days*). Inline **Cancel** button next to **Open** for in-flight fires. Two new helper functions: `cancelCreate` and `markFireOpened` (clears `is_new` server-side). |
| `static/style.css` | New `.newfire-*`, `.nf-*`, and `.status-new` selectors for the new page + badge. |

Files unchanged from the sibling (every other module — `auth`,
`notifications`, `cache_retention`, `progress`, `mapping`, `brush`,
`kml`, `templates`, `mapping_cmd`, `workers`, `preview`, `io_utils`,
`recommended_settings.yaml`, the rest of `handlers/`, the rest of
`templates/`, the rest of `static/`).

---

## Persistence and crash recovery

Same atomic-write pattern as the sibling (`io_utils._atomic_yaml_dump`
→ tmp + fsync + rename + parent-dir fsync). Two extras:

- **Overview cache**: PNG + sidecar JSON are atomically written. A
  partial PNG cannot survive a crash; if the JSON is corrupt,
  `overview_is_fresh` returns False and the next launch regenerates.
- **Orphan in-flight fires**: a server crash mid-prepare leaves the
  `.web_cache/<NAME>/` dir with no live worker. On boot,
  `init_fires_from_disk` flips those to `ERROR` with message
  `interrupted; retry create` so the operator can decide whether to
  delete and recreate. (No automatic resume — that's noted as future
  work in `PLAN.md` §13.)

---

## Tests

```bash
cd /home/bill/GitHub/wps-research/data/bill
python3 -m pytest batch_fire_mapping_viirs_web/tests/ \
    --ignore=batch_fire_mapping_viirs_web/tests/audit -v
```

Baseline: **111 pass** across `test_overview*`, `test_*validation*`,
`test_tight_crop`, `test_tight_bounds_from_shapefile`,
`test_viirs_worker_cancel`, `test_viirs_worker_progress`,
`test_cancel_create_nonblocking`, `test_detective`,
`test_fire_create_endpoint`, `test_year_index_fast_path`. The
audit-suite tests under `tests/audit/` are the legacy `bash run_all.sh`
PASS/FAIL framework imported wholesale from the sibling — they reference
the polygon package and are skipped here.

A real-LAADS end-to-end test (small AOI + 2-day window against a
working token) is left to manual QA — see `PLAN.md` §14 for the
checklist.

---

## Cross-references

For everything downstream of `READY` — fire-list filtering, single-shot
mapping, serial-mapping (N×K sweep), rebrush, accept, batch, cache
retention, multi-year switching, queue / toasts, KML
export, persistence + crash recovery, the full HTTP API, and the file
overview for unchanged modules — see the sibling package's
[`README.md`](../batch_fire_mapping_web/README.md). The architecture,
mixin-globals binding pattern, GPU-lock model, and `_wire_handlers`
helper-dict design are all identical.

For the implementation plan that drove this package, see
[`PLAN.md`](./PLAN.md) (§§0–14).
