# Agent Prompt: Build `batch_fire_mapping_viirs_web`

You are implementing a new Python package: a web app that lets analysts create wildfire-mapping projects by drawing a bounding box on a satellite-imagery raster, then automatically downloads VIIRS active-fire data for that bbox+date range, prepares the inputs, and seeds the existing ML mapping pipeline. It is a sibling to an already-working web package; the *front* of the pipeline changes, the back is unchanged.

## Read this first, in order

1. **`/home/bill/GitHub/wps-research/data/bill/batch_fire_mapping_viirs_web/PLAN.md`** — the full implementation plan. Read top to bottom. Sections §0-§14. **Do not skip.** All the design decisions, file-by-file diff against the source package, endpoint specs, worker state machine, and acceptance criteria are there. Your job is to execute §11 (implementation order) and verify §14 (acceptance criteria).
2. **`/home/bill/GitHub/wps-research/data/bill/batch_fire_mapping_web/`** — the source package. You will `cp -r` this into `batch_fire_mapping_viirs_web/` (keeping `PLAN.md` and this prompt) and then diff. Get familiar with its layout: `app.py` is the composition root, `state.py` has `AppState` and `FireInfo`, `prepare.py` has the polygon-driven crop logic you are replacing, `handlers/*.py` are mixin route classes, `templates/*.html` are simple variable-substitution templates (no Jinja). Stdlib-only — `http.server.ThreadingHTTPServer`, no FastAPI, no Flask.
3. **`/home/bill/GitHub/wps-research/data/bill/viirs/`** — the VIIRS toolkit you will reuse. The pieces you need:
   - `viirs/utils/laads_data_download_v2.py:111` — `sync(url, dest, token)` HTTP downloader.
   - `viirs/utils/shapify.py:202` — `process_file(...)` and `find_nc_files(...)`.
   - `viirs/utils/accumulate.py:135` — `accumulate(shp_dir, start_str, end_str, reference_raster=, output_dir=, final_only=, bbox=)`.
   - `viirs/utils/rasterize.py:62` — `rasterize_shapefile(shp_path, ref_image, output_dir, buffer_m)`.
   - `viirs/fp_gui/download_dialog.py:289` — proven end-to-end pipeline pattern (16 parallel ThreadPoolExecutor workers, cancel-on-close, automatic shapify after download). Mine this for the worker.
4. **`/home/bill/GitHub/wps-research/data/bill/batch_fire_mapping/run_fire_mapping.py`** — the underlying CLI helpers your worker will call: `crop_raster`, `raster_native_extent`, `get_raster_info`. Don't modify it.

## What you are building

A standalone web package at `/home/bill/GitHub/wps-research/data/bill/batch_fire_mapping_viirs_web/`. The PLAN.md spells out every file. High-level: copy `batch_fire_mapping_web` then make these changes:

- **CLI** (`__main__.py`): drop the polygon-shapefile positional argument; add per-year overview-PNG generation; require a LAADS token at `/data/.tokens/laads`.
- **State** (`state.py`): drop `gdf`/`viirs_gdf`/`polygon_file`; add `bbox_native`, `bbox_wgs84`, `viirs_start_date`, `viirs_end_date`, `is_new` to `FireInfo`; replace `init_fires_from_gdf()` with `init_fires_from_disk()`.
- **New page** (`templates/new_fire.html` + `static/new_fire.js`): a static overview PNG with an HTML5 canvas overlay; user drags a rectangle to define the VIIRS download AOI, types a fire name, optionally edits Start/End date placeholders (defaults `<year>-03-01` / `<year>-10-30`), confirms.
- **New endpoints** (`handlers/fire_list.py`): `GET /new_fire`, `GET /api/year/<y>/overview.png`, `GET /api/year/<y>/overview_meta`, `POST /api/fire/create`, `POST /api/fire/<n>/cancel_create`, `POST /api/fire/<n>/clear_new`.
- **New worker** (`viirs_worker.py`): background thread that runs download → shapify → accumulate → rasterize → tight-crop derivation → final crop, with cancellation between stages. Modeled on `viirs/fp_gui/download_dialog.py`.
- **New module** (`overview.py`): generates one downsampled PNG per year-raster (≤2000 px on the longest edge) plus a sidecar JSON with the GeoTransform, CRS, default dates, and bbox extents. Cached on disk keyed by `(raster mtime, raster size)`.
- **Tests** (`tests/`): see PLAN §10 for the full list. Use `pytest`. Synthetic fixtures for raster + VIIRS bin live in `conftest.py`.

The downstream mapping/brush/accept/serial flow is unchanged — once a fire reaches `READY`, every existing route works.

## Critical constraints

- **Stdlib HTTP only.** No FastAPI, uvicorn, Flask, Jinja2. Templates use the existing `_html_escape` + variable-substitution helpers in `templates.py`. Don't add new HTTP frameworks.
- **No `Pillow`.** Use `matplotlib` + `numpy` + `gdal` like the source package does (`preview.py:178`).
- **GDAL via `from osgeo import gdal`** with `gdal.UseExceptions()`. Same pattern as the source package.
- **Memory bound for big rasters.** Use `gdal.Band.ReadAsArray(buf_xsize=, buf_ysize=)` for stride-decimated reads. Never `ReadAsArray()` on a 100 GB raster. The overview module is the only place that touches the full raster, and it must use buffered reads.
- **Atomic writes.** Use the existing `io_utils._atomic_yaml_dump` pattern (write to `*.tmp.<pid>.<tid>`, fsync, rename, fsync parent dir) for any persistence file.
- **Locks.** `state.lock` (RLock) protects `AppState` mutations. New: `state.viirs_jobs_lock` (regular Lock) protects the worker registry. Test-and-set on `fire.status` happens under `state.lock` (see `prepare.py:72-77` for the pattern).
- **Path traversal.** Validate fire names with the regex `^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$`, reject `..` substring, reject `/` and `\`. The source package already does this for polygon-derived names (`state.py:276-278`); reuse the same regex.
- **No `mkdir -p`-then-write without `exist_ok=True`.** Workers race on cache_dir creation.
- **Cancellation.** Workers must check `fire.cancel_event` between stages and return promptly. Subprocess group kills use `os.killpg(proc.pid, SIGTERM)` (mirror `app.py:_terminate_serial_proc`); start subprocesses with `start_new_session=True`.

## Implementation order (from PLAN §11 — follow exactly)

1. `cp -r batch_fire_mapping_web batch_fire_mapping_viirs_web/` (preserve `PLAN.md` and `AGENT_PROMPT.md` — they're already in the destination).
2. Build `overview.py` + write+pass `test_overview.py`, `test_overview_caching.py`.
3. Edit `state.py` — drop polygon fields, add new ones, replace fire-init function.
4. Edit `__main__.py` — drop polygon arg, add overview generation + token load. Verify `python -m batch_fire_mapping_viirs_web --help` works.
5. Add validators in `validation.py` + write+pass `test_bbox_validation.py`, `test_date_defaults.py`, `test_date_validation.py`, `test_fire_name_validation.py`.
6. Build `viirs_worker.py` + write+pass `test_tight_crop.py`, `test_viirs_worker_cancel.py`.
7. Edit `prepare.py` — re-prepare path uses bbox/viirs_bin instead of polygon.
8. Edit `persistence.py` — fire-state YAML schema + simplified `_switch_year`.
9. Edit `handlers/fire_list.py` and `handlers/fire.py` — new endpoints + drop polygon-derived fields. Write+pass `test_fire_create_endpoint.py`.
10. Build `templates/new_fire.html` + `static/new_fire.js` + style additions.
11. Edit `templates/fire_list.html` — "+ New Fire" button, `new` badge, sub-stage display.
12. End-to-end manual QA against the `--insecure_no_auth` flag (PLAN §14).

Run only the tests for the layer you just built — don't run the full suite until step 12. This keeps the feedback loop tight.

## What you do NOT need to do

- Don't modify `viirs/`, `batch_fire_mapping/`, `py/fire_mapping/`, or any code outside the new package.
- Don't touch `batch_fire_mapping_web/` (the source package). It must keep working unchanged.
- Don't add pan/zoom, tile servers, Leaflet, or per-user LAADS tokens — explicitly out of scope (PLAN §13).
- Don't refactor the source package's mixin-globals binding pattern. Copy it as-is; it works.

## Things to watch out for (gotchas surfaced during planning)

- The user-drawn bbox is the **VIIRS download AOI**, NOT the final Sentinel-2 crop. Final crop bounds are derived in the worker by `_tight_bounds_from_viirs_bin` (PLAN §9) — bounding box of nonzero pixels in the rasterized VIIRS bin, expanded by `padding * max_dim` (mirrors `prepare.py:117-128` exactly). The PNG previews and the mapping pipeline see the tight crop, never the raw user bbox.
- After the tight crop, **VIIRS is rasterized AGAIN** with the cropped raster as the reference, so the hint aligns to the crop's grid. Don't skip this second rasterize.
- **Date placeholders, not values.** The HTML inputs render `placeholder="2023-03-01"` etc. in the user's date fields. If the user submits empty inputs, the server fills in the defaults. Test both the placeholder render and the empty-submit fallback (`test_empty_dates_in_create_request_use_defaults`).
- **VNP14IMG availability lower bound is 2012-01-19**, not 2012-01-01. Use that exact date in the validator.
- **`fire_numbe` field is reused** as the user-supplied fire name. The downstream code already treats it as an opaque string identifier — don't rename the field.
- **Fire-name uniqueness is case-insensitive.** `'fire1'` and `'FIRE1'` collide. Test for this.
- **Worker queue is FIFO with `max_concurrent_jobs=1`** by default, so two users creating fires back-to-back will see queue position. Add this as a CLI flag `--viirs_concurrent_jobs` defaulting to 1.
- **Overview generation is synchronous at startup** — printing per-raster progress to stdout. A 100 GB raster might take a couple of minutes the first time; subsequent starts hit the cache.
- **Server crash mid-prepare** leaves `fire_state.yaml` with a `preparing` status but no live worker. `init_fires_from_disk` must flip orphans to `ERROR` with message `'interrupted; retry create'`.
- **`agreement_pct`** still works without changes — the existing comparison is "ML mask vs. hint mask" and the hint is still the VIIRS rasterization. Don't gut that code.
- The user has set `--insecure_no_auth` for development. Don't gate fire-creation behind admin (both roles can create fires).

## How to validate your work

Tests pass:
```bash
cd /home/bill/GitHub/wps-research/data/bill/batch_fire_mapping_viirs_web
pytest tests/ -v
```

Server starts and the new-fire flow renders:
```bash
cd /home/bill/GitHub/wps-research/data/bill
python -m batch_fire_mapping_viirs_web \
    --rasters /path/to/pgfc_2023.bin \
    --out_root /tmp/viirs_web_test \
    --insecure_no_auth
# Then open http://localhost:8765/new_fire
```

End-to-end (manual, requires real LAADS token at `/data/.tokens/laads`): create a fire over a small known-fire AOI with a 2-day window, watch progress through `downloading_viirs → shapifying → accumulating → rasterizing → cropping → ready`, open the fire, run "Map Fire", confirm it produces the same output structure as the source package.

Then run PLAN §14 acceptance checklist top to bottom.

## Style and quality bar

- Match the existing package's style: short modules with `init(app_state, helpers)` for binding shared state, `state: AppState = None` module-level stubs, `from .module import _func` imports, no abstract base classes, no dependency injection frameworks.
- Comments only when *why* is non-obvious. Don't narrate *what* the code does. The source package is a good model — read a few of its modules to calibrate.
- No backwards-compatibility shims for the removed polygon path. This is a fresh package; just delete the dead code.
- Run `grep -r 'gdf\|viirs_gdf\|polygon_file\|polygon_gdf_raw' batch_fire_mapping_viirs_web/` and confirm zero hits before declaring done.

## When you finish

Print a short summary: which acceptance criteria you verified, which you couldn't (e.g. real-LAADS E2E if no token), and any deviations from PLAN.md with a one-line reason each. If you hit something the plan didn't anticipate, document it in a new `NOTES.md` next to PLAN.md and proceed with your best judgement.

Begin.
