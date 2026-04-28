This version of the web application is for historical research purposes. It seeds fire mapping results from historical polygons. The other version is for operational use and seeds fire mapping results from VIIRS data: 

* [batch_fire_mapping_viirs_web](https://github.com/bcgov/wps-research/tree/master/data/bill/batch_fire_mapping_viirs_web)

# batch_fire_mapping_web

*Last updated: April 27, 2026 — correctness audit pass. Critical and
High items from `AUDIT_REPORT.md` (parent-dir fsync, TOCTOU on
`fire.status` test-and-set, re-entrant accept guard, NaN leak into
`fire_size_ha`, single-pass template substitution, startup sweep of
stale tmp files, save/notify outside `_gpu_lock`) are fixed. Accept
now also emits `<FIRE>.shp` (source CRS) and `<FIRE>.kml` (EPSG:4326)
via the new `kml` module. The package layout: `app.py` remains the
thin composition root that wires `auth`, `notifications`,
`cache_retention`, `progress`, `mapping`, `persistence`, `brush`,
`kml`, `io_utils`, `templates`, `validation`, `mapping_cmd`,
`prepare`, `workers`, and the `handlers/` subpackage. `FireHandler`
is pure mixin composition — no inline route methods.*

Interactive web interface for mapping wildfire burn areas from
Sentinel‑2 satellite imagery. The system runs a GPU‑accelerated machine
learning pipeline (t‑SNE dimensionality reduction → Random Forest
classification → HDBSCAN clustering) to label burned vs. unburned
pixels per fire, then provides a browser UI for analysts to inspect
results, tune parameters, and accept the best run for each fire.

This package is the web companion to the `batch_fire_mapping` CLI.
The CLI runs everything from start to finish in one batch; this
server wraps the same underlying pipeline but turns it into an
interactive workflow with stage‑aware progress, side‑by‑side run
comparison, parameter presets, and a rebrush flow for refining the
post‑classification edge segmentation without re‑running the whole
pipeline.

**Multi-year aware**: a single launch accepts many Sentinel‑2 rasters
(one per year). The server auto‑detects each raster's year from its
filename, picks an initial active year, and lets administrators swap
to another year from the fire‑list filter panel without restarting
the process.

---

## Table of contents

- [What this tool does](#what-this-tool-does)
- [Architecture at a glance](#architecture-at-a-glance)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [Multi-year support](#multi-year-support)
- [Authentication and access control](#authentication-and-access-control)
- [How the web UI works](#how-the-web-ui-works)
- [Operational features](#operational-features)
- [Serial mapping and parameter learning](#serial-mapping-and-parameter-learning)
- [Recommended settings](#recommended-settings)
- [Typical workflow](#typical-workflow)
- [Output structure](#output-structure)
- [Network access](#network-access)
- [Persistence and crash recovery](#persistence-and-crash-recovery)
- [Known limitations](#known-limitations)
- [HTTP API reference](#http-api-reference)
- [File overview (every module)](#file-overview-every-module)

---

## What this tool does

Given (1) a Sentinel‑2 raster covering a fire season, (2) a polygon
shapefile of fire perimeters, and (3) a directory of VIIRS active‑fire
detections, this server lets analysts:

1. **Browse** every fire from the polygon set as a paginated list,
   filter by status / year / size, and open any one to a detail page.
2. **Prepare** a fire — crop the raster + VIIRS to the fire's
   bounding box (with configurable padding), accumulate VIIRS hot
   pixels into a hint mask, and pre‑render preview images.
3. **Map** the fire — run the GPU pipeline to produce a classified
   raster (burned / unburned), with live stage‑aware progress.
4. **Tune** parameters and re‑map — single‑shot for a single
   parameter set, or "Map with Settings" to sweep N recommended
   parameter sets × K HDBSCAN replicates each.
5. **Compare** runs side‑by‑side in a results gallery (overlay,
   classification, brush comparison) and pick a best run.
6. **Accept** a run — promote that run's outputs into the canonical
   per‑fire output directory and append a row to a CSV that is the
   ground truth for the parameter‑learning loop.
7. **Rebrush** an accepted (or candidate) classification — re‑run only
   the morphological brush step with new brush parameters, without
   redoing t‑SNE or RF.

Everything is exposed over HTTP from a single Python process. The
server is stdlib‑only (no FastAPI / uvicorn / Flask / Jinja); HTTP is
`http.server.ThreadingHTTPServer`, templates are simple variable
substitution, real‑time updates use Server‑Sent Events and short‑poll
JSON endpoints. This is intentional — the deployment surface is HPC
nodes and the dependency graph is kept minimal.

---

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────┐
│  __main__.py — argparse, year detection, VIIRS prep, AppState    │
│              construction, server boot.                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │ init_app(app_state)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  app.py — composition root (~450 lines)                          │
│   • module-level locks/registries (gpu_lock, batch_thread, etc)  │
│   • subprocess plumbing (_stream_subprocess, watchdog, sigterm)  │
│   • init_app() wires every sibling module in order               │
│   • FireHandler class (pure mixin composition, no method bodies) │
│   • create_server()                                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
  Sibling modules     handlers/ subpackage    Worker subprocesses
  (init(state, …))    (10 mixin classes)     (fire_mapping_cli.py,
                                              class_brush.exe)
```

**Sibling modules** (each holds focused logic + persistence; bound to
shared state via an `init(app_state, …)` call):
`state`, `auth`, `notifications`, `cache_retention`, `progress`,
`mapping`, `persistence`, `brush`, `templates`, `validation`,
`mapping_cmd`, `prepare`, `workers`, `preview`, `io_utils`.

**Handler mixin modules** (each declares stub globals and an
`init(app_state, helpers)` that copies the helpers dict into its
namespace; all method bodies see shared state through these
module-level names):
`base`, `auth`, `fire_list`, `fire`, `mapping`, `serial`, `rebrush`,
`batch`, `ops`, `static`. They compose into `FireHandler` via MRO.

**No circular imports**: every sibling module imports from
`state`/`io_utils` and from sibling modules whose `init()` runs
earlier. Handlers import from siblings but not vice‑versa.

---

## Requirements

- **Python 3.11+** (uses union types `X | Y`, `dict[k, v]`, etc.)
- **GDAL** with Python bindings (`from osgeo import gdal`)
- **NumPy**, **GeoPandas**, **Shapely**, **PyYAML**
- **CUDA runtime + a CUDA-capable GPU** for the underlying CLI
  pipeline (the web layer itself is CPU-only)
- **`fire_mapping_cli.py`** on the project tree (the GPU subprocess
  this server drives)
- **`class_brush.exe`** built from `cpp/class_brush.cpp` (the C++
  morphological-brush helper used by the rebrush flow)
- **`ogr2ogr`** on PATH and **`py/binary_polygonize.py`** on disk
  for the KML/shapefile export at accept time. The polygonize path
  defaults to `/home/bill/GitHub/wps-research/py/binary_polygonize.py`
  and can be overridden with the `WPS_BINARY_POLYGONIZE` env var.
  KML failures warn but do not abort an accept.

The web layer never imports the GPU pipeline directly — it spawns the
CLI as a subprocess and parses its stdout for stage markers.

---

## Quick start

```bash
# From the project root, after the CLI side is built and on PATH:
python -m batch_fire_mapping_web \
    /path/to/fire_perimeters.shp \
    --rasters /path/to/S2_2023_*.bin /path/to/S2_2024_*.bin \
    --out_root /path/to/outputs \
    --admin_password $ADMIN_PW \
    --user_password  $USER_PW \
    --port 8765
```

Then open `http://<host>:8765` in a browser.

The server prints a banner on startup listing the detected years, the
chosen active year, the raster path it resolved to, and the polygons
file. After that it idles waiting for HTTP requests; per‑fire prepare /
map / accept work happens on demand from the UI.

---

## Command reference

| Argument | Purpose |
|---|---|
| `polygon_file` | Path to fire perimeter shapefile (positional). |
| `--rasters` | One or more Sentinel‑2 rasters; year is auto-detected from filename. |
| `--out_root` | Output root directory. Per-year subdirs and shared files live here. |
| `--year` | Force initial active year. If omitted, the server reads `active_year.yaml` from `out_root`, or falls back to the newest year. |
| `--perimeter_mode` | `viirs` (use VIIRS hot pixels as hint) or `polygon` (use the perimeter shape). |
| `--skip_download` | Skip VIIRS download step (use already-downloaded shapefiles). |
| `--shapify_workers` | Parallel workers for VIIRS shapification (default 8). |
| `--sample_rate`, `--min_samples`, `--max_samples` | Default sampling parameters, also editable per-fire from the UI. |
| `--host`, `--port` | Listen address. Defaults to `0.0.0.0:8765`. |
| `--admin_password`, `--user_password` | Admin / user role passwords. **Required unless `--insecure_no_auth`.** Set them at launch from a secret store; do not commit them. |
| `--insecure_no_auth` | Skip authentication entirely. **Only for isolated, single-user environments** — the server has no other defenses against an unauthenticated client. |
| `--trust_proxy` | Honor `X-Forwarded-For` for client IP. Only enable when running behind a reverse proxy you control. |

Run with `--help` for the canonical list.

---

## Multi-year support

A fire season spans one Sentinel‑2 raster per year. Operators almost
always want to flip between years (e.g. compare 2023 vs. 2024 fires)
without restarting the server. This is what multi-year support
provides.

### Year detection

`__main__.py` calls `_year_from_filename` on each raster path. The
detector looks for a 4-digit year token bracketed by non-digits in the
filename (range `[1970, now_year + 1]`). Multiple year-like tokens in
a single filename, no year token at all, or two rasters that map to
the same year are all fatal errors at launch.

### Active year

At any moment one year is "active". `state.raster_path`,
`state.output_root`, `state.viirs_shp_dir`, `state.gdf`,
`state.viirs_gdf` all point at the active year. The full
`{year → resource}` map is in `state.rasters_by_year`,
`state.outdirs_by_year`, etc.

The initial active year is chosen in this priority:
1. `--year` argument if given.
2. `active_year.yaml` saved under `out_root` from a prior session.
3. The newest year in `rasters_by_year`.

### Switching years (admin-only)

`/api/year/switch` (admin role) re-projects and spatially-filters the
cached raw polygon GeoDataFrame against the new year's raster, replaces
the active-year handles, and broadcasts a toast notification. Refused
if any rebrush is currently running (rebrush state is per-fire and the
canonical output path moves between years).

### Shared vs. per-year files

Anything tied to a specific raster — fire state, accepted CSV,
`.web_cache` — is **per-year**, scoped under the per-year output
directory. Anything global to the deployment — sessions, IP access
list, settings overrides, notes, broadcast notifications, presets,
parameter recommendations — is **shared**, scoped under
`state.shared_root` (which equals `out_root`). This separation lets
year-switches preserve operator state (you don't lose your login or
your saved settings) while flipping the data layer.

---

## Authentication and access control

The server uses a two-role model: **admin** and **user**.

- **Login** is via a single password per role, set at launch through
  command-line args (do not commit; pull from a secret store).
- **Session tokens** are issued on successful login and persisted on
  disk so logins survive server restarts (and TTL-expire after a
  bounded period).
- **Per-IP rate limiting** on failed login attempts blocks brute force.
- **CSRF protection** is in place on every state-changing route via an
  origin/CSRF check.
- **IP access control** is layered on top: every client IP is in one
  of three states — *approved*, *blocked*, or *pending*. New IPs land
  in pending and admins approve or block them from the admin
  dashboard. The pending page shows a friendly "waiting for approval"
  banner with no information disclosure.
- **Role gates** are method-level: every admin-only handler checks
  the session role before doing anything that mutates shared state.
- **Audit trail** of cancels and admin actions is written to a YAML
  file under `out_root`.

`--insecure_no_auth` exists for isolated environments (single user, no
network exposure). Enabling it disables every check above. Do not use
it on a multi-user host.

---

## How the web UI works

### Fire list (home page)

The home page renders the polygon set as a sortable, filterable list:

- **Filter pills** for status (Pending / Ready / Mapped / Accepted /
  Error / Hidden), year, and size buckets.
- **Search** by fire number.
- **Hide** a fire to drop it from the default list (admins can unhide
  from the "hidden" filter pill).
- **Notes** field on each row for analyst comments (autosaved, shared
  across the deployment).
- **Per-fire status badge** — color-coded, with the live status
  retrieved from `state.fires`.
- **Batch operations bar** (admin only): select a subset of fires,
  hit *Map selected* to sweep them serially (one fire at a time, full
  N×K parameter sweep per fire).

### Fire mapping page

The fire detail page is split into three areas:

#### Image viewer (left side)

Tabs for each available view: pre, post, hint (VIIRS overlay),
classified, comparison, brush comparison, result. Tabs only show up
if their backing file exists on disk — `_load_fire_state` filters
`available_views` against the actual file system on every load, so a
manual `.web_cache` wipe can never leave the UI asking for a missing
PNG. Click‑to‑zoom and pan; toggle the polygon overlay.

#### Parameters (right side)

Editable form for sample rate, padding, t-SNE (perplexity, learning
rate, max iter, init, components, random state), Random Forest (n
estimators, max depth, max features, random state), HDBSCAN
(min samples), embedding bands (1-indexed), brush controls (size,
threshold, all-segments toggle), and seed. Every field is validated
client-side and again server-side via `validation._validate_param`
before being passed to the CLI.

**Preset bundles** are buttons above the parameters form. Clicking one
seeds the form with a known-good configuration from
`recommended_settings.yaml`. The user can override any field after
seeding.

#### Mapping

Three buttons:
- **Map** — single-shot run with the current parameters. Streams CLI
  stdout as Server-Sent Events into the live console below.
- **Map with settings** — runs the full N recommended settings × K
  HDBSCAN replicates sweep (see [Serial mapping](#serial-mapping-and-parameter-learning)).
  Each replicate produces a card in the results gallery.
- **Cancel** — sends SIGTERM to the running CLI subprocess and
  unwinds the worker safely.

Live progress bar above the console shows current stage (load → hint
→ sample → t-SNE → RF → HDBSCAN → classify → brush → figure), with an
ETA derived from a running median of past run durations for this
deployment.

#### Rebrushing without re-mapping

The morphological brush step (the cpp/class_brush.exe program) is the
edge-cleanup pass that runs after HDBSCAN. Re-running it doesn't need
t-SNE or RF, so the rebrush flow:

- Restores `*_classified_raw.bin` (the pre-brush snapshot saved by
  `prepare`) → runs class_brush.exe with new brush parameters →
  emits a new `*_classified.bin` and a new `*_brush_comparison.png`.
- Holds `_gpu_lock` for the duration so it serialises against
  mapping work.
- Registers its subprocess in `_rebrush_procs` so the cache sweeper
  knows the cache_dir is hot and won't evict it, and so the cancel
  endpoint can SIGTERM the right child.

#### Results gallery

Below the console is the gallery of serial-map runs (visible after
"Map with settings" or after a single-shot map). Each card has:

- **Run #** and which setting/replicate produced it.
- **Agreement %** (IoU between the ML mask and the VIIRS hint, with
  geotransform-aware alignment so cross-extent overlap is correct).
- **ML burned area** (ha).
- **Image strip**: classification, comparison, brush comparison.
- **Accept** button (admin-only): promote this run to the canonical
  output and append a CSV row.
- **Rebrush** button: open the rebrush controls keyed off this run's
  `_raw.bin` snapshot.

#### Accepting

"Accept" is the commit operation. It:

1. Acquires `_accept_file_lock` (serialises CSV writes across
   concurrent accepts on different fires).
2. Adds the fire to `_accept_in_progress` (so the cache sweeper does
   not evict the cache_dir mid-copy).
3. Copies the run's classified.bin / hdr / overlay / comparison PNG /
   raw.bin from the `.web_cache` dir into `<out_root>/<FIRE>/`.
3a. Generates `<FIRE>.shp` (source CRS) and `<FIRE>.kml` (EPSG:4326)
    by polygonizing the classified raster. KML failures warn-and-
    continue — they do not abort the accept.
4. Sets `serial_accept_promoted = True` and `serial_canceled = True`
   on the fire so the worker's cancel path will tear down the
   gallery cleanly when it picks up the signal.
5. Sets `serial_accept_event` so the worker waits for the file copy
   to finish before deleting any serial_* files.
6. Appends a row to `accepted_params.csv` under `out_root` with the
   parameters and metrics that produced this run.
7. Saves `fire_state.yaml` and removes the fire from
   `_accept_in_progress`.

The CSV is the single source of truth for the parameter knowledge
base.

### Navigation

Top bar has a year selector (admin only), the queue indicator (number
of jobs currently waiting on the GPU), the toast container (slide-in
notifications for batch / sweep / year-switch events, polled every
few seconds), and login/logout.

### GPU queue

A single `_gpu_lock` serialises all heavy operations. `_gpu_queue` is
a counter of waiting + running jobs. When a request hits a busy GPU
it shows "Queued — N jobs ahead" in the live console and waits. The
queue position is also shown in the queue banner at the top of the
page.

---

## Operational features

### Stage-aware progress bar + ETA

`progress.py` parses `fire_mapping_cli.py` stdout for stage marker
substrings (defined in `_STAGE_MARKERS`) and advances a progress pill
bar across the canonical stage sequence (load → hint → sample → tsne
→ rf → hdbscan → classify → brush → figure). Stage durations are
recorded in a sliding window per stage; ETA is computed from the
running medians of past durations on this deployment. Persisted
between restarts.

### Job queue visibility

`/api/queue` returns `{current_job, waiting_jobs, queue_depth}` so the
queue banner stays current. Both the user mapping flow and the
serial-mapping flow register entries here so a single-shot map shows
up alongside in-flight serial sweeps.

### Toast notifications

Per-session queues plus a broadcast channel. Used for:
- *info* — year-switch confirmations (broadcast).
- *success* / *warning* / *error* — mapping complete / failed / batch
  done (per-session, with a "View" action linking back to the fire).

Persisted to `notifications.yaml` so a server restart doesn't lose
in-flight messages. TTL-expired and over-limit-pruned on every push.

### Preset bundles

`recommended_settings.yaml` has two top-level keys:
- `presets:` — named bundles surfaced as buttons in the UI. Each has
  a label, description, and full parameter dict. Clicking one seeds
  the form. Mutating an override later does not touch the bundle.
- `settings:` — the ordered list consumed by "Map with Settings".
  First entry is the *primary* used by single-shot Map.

### Unified abort + cancel audit log

Every cancel goes through a single helper that: (1) flips the
appropriate flag (`serial_canceled`, `_batch_cancel`, etc.), (2)
signals the running subprocess via the per-fire process registry,
(3) appends a row to the cancel audit YAML with timestamp, user, fire,
and the user-supplied reason if any.

### Cache retention

`cache_retention.py` runs a periodic sweep of `<out_root>/.web_cache/`:
- `_cache_scan` reports per-cache-dir size, mtime, and pin reasons.
- `_cache_sweep` evicts oldest-first until under the configured size
  cap, respecting hard pins (rebrush in progress) and soft pins
  (accept in progress) so we never wipe a directory that's actively
  being copied out of.
- Configurable from the admin dashboard: max GB, max age days, sweep
  interval hours, enabled flag.

### Hardened year-switch guard

`_switch_year` refuses to swap years while any rebrush is running,
because a rebrush's canonical output path is per-year and changing
mid-flight would land the result in the wrong directory.

---

## Serial mapping and parameter learning

"Map Fire with Settings" runs N parameter configurations × K HDBSCAN
replicates per configuration on a single fire, populating a results
gallery the analyst chooses from. This is what powers the
parameter-learning loop.

### How it works

The orchestrator (`workers._serial_map_worker`) decomposes into five
phase helpers:

1. **`_serial_setup`** — wipes leftover `serial_*` artifacts from a
   previously cancelled sweep, resets the `fire.serial_*` fields, and
   builds a `_ProgressTracker`.
2. **`_serial_snapshot_run0`** — if the fire has a previously
   accepted result, copies it into the gallery as Run 0 so the analyst
   can compare new sweeps against the prior best.
3. **`_serial_run_replicate`** — executes one replicate inside the
   per-setting loop. Holds `_gpu_lock` across the replicate. Replicate
   0 of each setting saves t-SNE+RF state to a `.npz`; replicates 1..K
   load it and only re-run HDBSCAN with a jittered min-samples value
   (fan-out pattern: base, +step, −step, +2·step, −2·step, ...).
4. **`_serial_handle_cancel`** — runs cancel cleanup. Two flavors:
   accept-initiated (full gallery wipe, status set to ACCEPTED) and
   user-initiated (preserve gallery, promote best run to MAPPED if
   any succeeded, otherwise revert to pre-sweep status).
5. **`_serial_finalize`** — for the no-cancel path: pick the best
   successful run by agreement %, promote it into the main slot,
   free per-setting `.npz` caches, persist state, and notify the
   initiating session.

### Size buckets

Every fire has a size in hectares. The recommended-settings list is
ordered globally; in practice analysts use the same set for all
sizes, but each fire's `recommended_override` (set by an admin) can
substitute a per-fire list. The override flows through
`_get_recommended_settings`.

### The learning loop

1. Analyst runs "Map with Settings" on a fire.
2. Reviews the gallery, picks the best run, hits Accept.
3. Accept appends a row to `accepted_params.csv` with the parameters
   and metrics.
4. Over time `accepted_params.csv` becomes the parameter knowledge
   base.
5. The recommended-settings list is updated periodically based on
   what's been winning across the accepted set.

---

## Recommended settings

`recommended_settings.yaml` is the operator-tunable file that drives
preset buttons and the "Map with Settings" sweep. Top-level keys:

- `k_runs_per_setting` — replicates per setting (clamped 1..10).
- `k_jitter` — fan-out step for `hdbscan_min_samples` across
  replicates. 0 disables jitter.
- `presets:` — `{name → {label, description, params}}` bundles
  surfaced in the UI.
- `settings:` — ordered list of `{label, params}` consumed by
  "Map with Settings". First entry is also the default used by
  single-shot Map.

Loaded once at boot into `state.recommended_settings` (and
`state.presets`); `_get_recommended_settings(fire)` returns deep copies
so callers can mutate without polluting the canonical list.

---

## Typical workflow

1. **Boot** the server with the year's raster + the polygon shapefile.
2. **Approve** new IPs from the admin dashboard.
3. **Open** the fire list, filter to "Pending".
4. **Click** a fire → review the polygon overlay on the pre image →
   "Prepare" if not already prepared.
5. **Map with Settings** → wait for the sweep to finish → review the
   gallery.
6. **Accept** the best run.
7. Repeat across the day's fires. The accepted CSV grows; the
   parameter learning loop catches up over time.
8. At end of season, switch active year if needed and continue.

---

## Output structure

```
<out_root>/
├── active_year.yaml              # which year is currently active
├── sessions.yaml                 # session tokens (hashed)
├── ip_list.yaml                  # IP access control list
├── settings_overrides.yaml       # per-deployment settings
├── notes.yaml                    # per-fire analyst notes
├── notifications.yaml            # toast queue persistence
├── stage_timings.yaml            # running medians for ETA
├── cache_retention.yaml          # sweep config
├── cancel_audit.yaml             # audit trail of cancels
├── <raster_stem_year_A>_mapping_results/
│   ├── fire_state.yaml           # per-fire status snapshot
│   ├── accepted_params.csv       # accepted-run parameter rows
│   ├── .web_cache/
│   │   └── <FIRE_NUMBER>/        # per-fire cache (crop, hint, runs)
│   │       ├── <FIRE>_crop.bin
│   │       ├── <FIRE>_hint.bin
│   │       ├── <FIRE>_serial_<N>_classified.bin
│   │       ├── <FIRE>_serial_<N>.png
│   │       ├── <FIRE>_serial_state_s<idx>.npz
│   │       └── previews/
│   │           ├── pre.png, post.png, hint.png, ...
│   │           └── result.png
│   └── <FIRE_NUMBER>/            # promoted-on-accept canonical dir
│       ├── <FIRE>_crop.bin_classified.bin   # final raster (source CRS)
│       ├── <FIRE>.shp / .shx / .dbf / .prj  # polygons in source CRS
│       ├── <FIRE>.kml                       # polygons in WGS84
│       ├── <FIRE>_comparison.png
│       ├── <FIRE>_brush_comparison.png
│       └── <FIRE>_params.yaml
└── <raster_stem_year_B>_mapping_results/
    └── …                         # same shape, different year
```

`.web_cache/<FIRE>/` is the working dir; promoting on accept copies
the run's outputs out to `<FIRE>/`. The cache sweeper only evicts
from `.web_cache/`; the canonical per-fire dirs are never touched.

---

## Network access

The server binds to `0.0.0.0:8765` by default. On HPC nodes that's
typically only reachable through SSH port forwarding:

```bash
ssh -L 8765:localhost:8765 user@hpc
# Then open http://localhost:8765 locally
```

Behind a reverse proxy, set `--trust_proxy` so `X-Forwarded-For` is
honored for the IP access control layer.

---

## Persistence and crash recovery

Every state-changing endpoint persists immediately through
`io_utils._atomic_yaml_dump`, which writes to `<file>.<pid>.<tid>.tmp`,
fsyncs the file, `os.replace`s into place, and **then fsyncs the
parent directory**. The directory fsync is what makes the rename
durable — POSIX guarantees `rename` is atomic, not durable, so
without it a power loss right after `os.replace` can lose the new
file even though `os.fsync` on the file content succeeded. The same
pattern is applied to the open-coded CSV writer for
`accepted_params.csv` in `prepare.py`.

On restart, `init_app` first sweeps any stale `*.<pid>.<tid>.tmp`
files older than 24h from `output_root` and `shared_root` so a
previous SIGKILL'd process doesn't leak orphan tmp files. Then:

- `_load_fire_state` rebuilds `state.fires` from `fire_state.yaml`,
  filters every `available_views` entry against the actual on-disk
  PNGs (a manual `.web_cache` wipe cannot leave the UI asking for a
  vanished view), and restores in-flight statuses to safe terminal
  ones (`MAPPING` → `READY`, `PREPARING` → `PENDING`). If
  `fire_state.yaml` itself fails to parse, the file is copied aside
  to `fire_state.yaml.corrupt-<ts>` (the copy is fsynced, parent dir
  too, so the only forensic evidence survives a subsequent crash)
  and `state.fire_state_load_failed` is set, blocking subsequent
  saves until an operator clears it.
- `_load_notifications` restores in-flight toasts and per-session
  cursors.
- Sessions, IP access list, settings overrides, presets, stage
  timings, and cache-retention config all reload from their YAMLs.

Concurrent-write safety: `_atomic_yaml_dump` is itself atomic at the
filesystem level (each writer uses a unique `<pid>.<tid>` tmp suffix
so concurrent writers do not clobber each other's tmp file), and the
`_accept_file_lock` (for `accepted_params.csv`) serialises
accept-time CSV mutations across all routes.

---

## Known limitations

- **Single GPU**: `_gpu_lock` serialises all heavy work. The queue
  banner shows users where they are; nothing parallelises across
  multiple GPUs.
- **Per-deployment global state**: settings, presets, IP list, and
  sessions are global. Multi-tenant separation is out of scope.
- **VIIRS-based hint only**: when `--perimeter_mode polygon`, the
  polygon outline is used directly as the hint; otherwise VIIRS hot
  pixels accumulated for a date window around the fire date.
- **Year detection requires a 4-digit year in the raster filename.**
- **No real-time browser push** — toasts are short-polled every few
  seconds. Server-Sent Events are used only for the live mapping
  console.
- **Re-prepare on padding change** — changing padding inside
  "Map with Settings" forces a fresh crop+hint+previews and wipes
  the per-setting `.npz` caches. This is intentional but not
  obvious; the console logs it.

---

## HTTP API reference

Endpoints are grouped by their owning mixin module under `handlers/`.
All POST routes require a session cookie and a CSRF/origin header; all
admin-only routes additionally require the session to have role=admin.

### Auth (`handlers/auth.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/login` | Login page (HTML). |
| POST | `/login` | Login endpoint. |
| POST | `/logout` | Logout — invalidates session. |
| GET | `/admin` | Admin dashboard (HTML). |
| GET | `/api/access/status` | Returns current session role + IP status. |
| GET | `/api/admin/ips` | Lists pending / approved / blocked IPs. |
| POST | `/api/admin/ip/<action>` | `approve` / `block` / `revoke` / `unblock` an IP. |
| GET | `/api/admin/queue` | Admin view of in-flight jobs. |

### Fire list (`handlers/fire_list.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Fire list page (HTML). |
| GET | `/fire/<FIRE>` | Per-fire mapping page (HTML). |
| GET | `/api/fires` | Paginated fire list as JSON. |
| GET | `/api/fires/hidden` | Hidden fires (admin). |
| POST | `/api/fire/<FIRE>/notes` | Save analyst notes. |
| POST | `/api/fire/<FIRE>/remove` | Hide a fire. |
| POST | `/api/fire/<FIRE>/unhide` | Unhide a fire (admin). |

### Per-fire state and views (`handlers/fire.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/api/fire/<FIRE>/status` | Current status + metrics. |
| GET | `/api/fire/<FIRE>/console` | Live console log buffer. |
| GET | `/api/fire/<FIRE>/progress` | Stage / ETA snapshot. |
| GET | `/api/fire/<FIRE>/comparison` | Comparison PNG. |
| GET | `/api/fire/<FIRE>/brush_comparison` | Brush comparison PNG. |
| GET | `/api/fire/<FIRE>/preview/<view>` | Per-view preview PNG. |
| POST | `/api/fire/<FIRE>/prepare` | Crop + hint + previews. |
| POST | `/api/fire/<FIRE>/abort` | SIGTERM the running CLI. |

### Mapping (single-shot + accept) (`handlers/mapping.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/fire/<FIRE>/map` | Single-shot map. SSE stream. |
| POST | `/api/fire/<FIRE>/accept` | Promote single-shot result. |
| GET | `/api/fire/<FIRE>/recommended` | Per-fire override list. |
| POST | `/api/fire/<FIRE>/recommended` | Set per-fire override (admin). |
| GET | `/api/settings` | Default settings. |
| POST | `/api/settings` | Update default settings (admin). |
| GET | `/api/presets` | Available preset bundles. |
| POST | `/api/fire/<FIRE>/preset` | Apply preset to fire. |

### Serial mapping (`handlers/serial.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/fire/<FIRE>/serial_map` | Start N×K sweep. |
| POST | `/api/fire/<FIRE>/serial/cancel` | Cancel running sweep. |
| GET | `/api/fire/<FIRE>/serial_results` | Gallery contents. |
| GET | `/api/fire/<FIRE>/serial/<run_id>/image` | Per-run image. |
| POST | `/api/fire/<FIRE>/serial/<run_id>/accept` | Accept a specific run. |

### Rebrush (`handlers/rebrush.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/fire/<FIRE>/rebrush` | Re-run brush only. |
| POST | `/api/fire/<FIRE>/rebrush/cancel` | Cancel rebrush. |

### Batch (`handlers/batch.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/batch/map` | Start batch over selected fires (admin). |
| GET | `/api/batch/status` | Batch progress. |
| POST | `/api/batch/cancel` | Cancel batch (admin). |

### Operations (`handlers/ops.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/api/queue` | Current job + queue depth. |
| GET | `/api/notifications` | Drain pending toasts. |
| POST | `/api/notifications/ack` | Mark broadcast cursor. |
| GET | `/api/cache/status` | Cache sizes per year. |
| POST | `/api/cache/sweep` | Trigger sweep (admin). |
| GET | `/api/years` | Year registry. |
| POST | `/api/year/switch` | Switch active year (admin). |
| GET | `/api/report` | Diagnostic report. |

### Static (`handlers/static.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/static/*` | Serve files from `static/`. |

---

## File overview (every module)

This section documents every Python file in the package. Files are
grouped by role.

### Entry points

| File | Purpose |
|---|---|
| `__init__.py` | Empty package marker. |
| `__main__.py` | The `python -m batch_fire_mapping_web` entry point. Parses arguments, validates polygon and raster paths, builds the `{year → raster/outdir/viirs_dir}` registry via `_year_from_filename`, resolves the initial active year, prepares VIIRS for every detected year (so subsequent year-switches are O(1)), filters polygons against the active year's raster extent, constructs an `AppState`, calls `app.init_app`, starts `cache_retention._cache_sweep_loop` in a daemon thread, then runs `app.create_server(...).serve_forever()`. |
| `app.py` | Composition root (~450 lines). Holds `init_app` (which calls every sibling module's `init` in dependency order, plus a startup sweep of stale `*.<pid>.<tid>.tmp` files via `_sweep_stale_tmp_files`), the shared module-level locks/registries (`_gpu_lock`, `_gpu_queue`, `_batch_thread`, `_serial_procs`, `_rebrush_procs`, `_accept_in_progress`, `_accept_file_lock`), the subprocess plumbing (`_stream_subprocess` with idle-watchdog timer, `_terminate_serial_proc`), the small in-app helpers `_set_fire_status`, `_clone_setting`, `_get_recommended_settings`, the canonical CSV fieldnames, `_wire_handlers` (builds the helpers dict and calls each handler mixin's `init`), the `FireHandler` class (pure mixin composition — every route method body lives in `handlers/`), and `create_server`. Re-exports every name extracted into a sibling module so external callers and `__main__.py` need no changes. |

### Shared data and helpers

| File | Purpose |
|---|---|
| `state.py` | Data classes for app-wide state. `FireStatus` is the status enum. `FireInfo` is the per-fire dataclass — `fire_numbe`, `fire_date`, `fire_year`, `fire_size_ha` (required); status, error_msg, the per-fire cache paths (`cache_dir`, `crop_bin`, `hint_bin`, `perim_bin`, `viirs_bin`), crop dims and `padding_used`, accumulation date window, `available_views`, results (`last_comparison`, `last_params`, `ml_area_ha`, `agreement_pct`), tracking flags (`previously_accepted`, `hidden`, `notes`), the `console_log` deque, the serial-sweep fields (`serial_results`, `serial_settings`, `serial_canceled`, `serial_prev_status`, `serial_accept_promoted`, `serial_accept_event`), the live `progress` dict, plus `last_preset` and `last_cancel_reason`. `AppState` is the global container — fires dict, raster/polygon GeoDataFrames, the multi-year registry (`active_year`, `shared_root`, `rasters_by_year`, `outdirs_by_year`, `viirs_shp_dirs_by_year`, `polygon_gdf_raw`), per-deployment defaults, the auth/session/IP state, the operational state (`stage_timings`, `notifications`, `broadcast_cursor`, `cache_retention`, `presets`, etc.), and an `RLock` used for read-modify-write of any shared mutable field. |
| `io_utils.py` | `_atomic_yaml_dump(path, payload, mode=0o600)` — the durable atomic-write pattern used by every persistence module: write to `<path>.<pid>.<tid>.tmp`, fsync the file, `os.replace` into place, then fsync the parent directory so the rename itself reaches disk. `_sweep_stale_tmp_files(roots, max_age_seconds=86400)` — startup helper that removes `*.<pid>.<tid>.tmp` files older than the threshold from each root, called once from `init_app`. Stateless. |
| `templates.py` | `_html_escape` (the five-character XSS-safe escape covering `&`, `<`, `>`, `"`, `'`) and `render_template` (load a `.html` from the templates directory and resolve placeholders in a single regex pass: `{{ key }}` substitutes the HTML-escaped context value, `{{{ key }}}` substitutes the raw value for known-safe insertion. The single-pass walk means a context value containing placeholder syntax is treated as literal output — placeholders cannot recursively reference other context keys). |
| `validation.py` | Pure-function parameter validation. `_PARAM_SPEC` is a dict mapping every CLI parameter to its kind (`int` / `float` / `choice` / `bool`) and bounds. `_validate_param(key, raw)` casts and bounds-checks one parameter; raises `ValueError` on any failure; returns the raw value for unknown keys. `_validate_embed_bands(eb)` validates a comma-separated band string (1-indexed, 1..999), tolerates whitespace and trailing commas, returns the cleaned string or `None` for empty input. |
| `preview.py` | ENVI header parsing, band detection, and preview PNG generation. |
| `recommended_settings.yaml` | Operator-tunable presets and the ordered settings list for "Map with Settings". See [Recommended settings](#recommended-settings). |

### Routes / sibling modules

| File | Purpose |
|---|---|
| `auth.py` | Session token hashing, IP normalization, per-IP login rate limiting, expired-session sweeper, and the auth-related constants. Wired by `auth.init(app_state)`. Internals are deliberately undocumented here (open-source visibility); read the source if you're maintaining it. |
| `notifications.py` | Per-session toast queues plus a broadcast channel. `_push_notification(target_session_or_None, kind, title, body, …)` enqueues; `_pop_notifications(session_hash)` drains personal queue plus any new broadcast entries past the session's cursor. Persisted to `notifications.yaml` so restarts don't drop in-flight messages. TTL-pruned and over-limit-pruned on every push. Wired by `notifications.init(app_state)`. |
| `cache_retention.py` | `.web_cache/` retention sweeper: `_cache_scan` (size + mtime + pin reasons per cache_dir), `_cache_sweep` (age-then-size eviction with hard pins for in-flight rebrushes and soft pins for in-flight accepts), `_cache_sweep_loop` (background daemon thread), config persistence (`_save_cache_retention` / `_load_cache_retention`). Captures `_save_fire_state` plus the in-flight registries through `cache_retention.init(...)`. |
| `progress.py` | Stage-aware progress tracking: `_STAGE_MARKERS` (substring → stage table), `_detect_stage`, the canonical stage-order constants, the `_ProgressTracker` class consumed by `_stream_subprocess`'s `on_line` hook, `_progress_snapshot` (run-duration-median ETA), and stage-timing persistence (`_save_stage_timings` / `_load_stage_timings`). Wired by `progress.init(app_state)`. |
| `mapping.py` | Result helpers used after a mapping run completes: `_compute_ml_area` (burned hectares from a classified raster), `_compute_agreement` (IoU% between ML mask and hint, with cross-extent geotransform alignment), `_overlay_mask_on_post` (tinted preview PNG with geotransform alignment), `_generate_result_preview` (calls `_overlay_mask_on_post` for the result + hint). Wired by `mapping.init(app_state)`. |
| `persistence.py` | All on-disk YAML persistence: `_save_sessions`, `_save_settings`, `_save_notes`, `_save_ip_list`, `_save_fire_state`, `_load_fire_state` (filters `available_views` against on-disk PNGs — see [Persistence and crash recovery](#persistence-and-crash-recovery)), `_save_active_year`, `_switch_year`. Wired by `persistence.init(app_state, _rebrush_procs, _rebrush_procs_lock, _compute_agreement, _compute_ml_area, _push_notification)`. **Init order matters**: `persistence.init` must run before `cache_retention.init`. |
| `kml.py` | KML/shapefile export at accept time. `_export_kml(fire_numbe, fire_dir)` runs `binary_polygonize.py` on `<FIRE>_crop.bin_classified.bin` (in `fire_dir`), renames the verbose `<FIRE>_crop.bin_classified.bin.{shp,shx,dbf,prj,cpg}` sidecars to `<FIRE>.{shp,shx,dbf,prj,cpg}`, deletes the wrong-CRS leftover KML that polygonize emits as a side effect, then runs `ogr2ogr -f KML -t_srs EPSG:4326 -overwrite` to produce `<FIRE>.kml` reprojected to WGS84 for Google Earth. Polygonize exits with status 1 even on success — success is verified by checking the `.shp` exists on disk; on failure the captured stderr's last 3 lines are surfaced in the warning. Path to `binary_polygonize.py` is overridable via the `WPS_BINARY_POLYGONIZE` environment variable. Failures warn-and-return-`None`; the function never raises so KML problems cannot abort an accept. Stateless (no `init`). |
| `brush.py` | C++ `class_brush.exe` wrapper + figure renderers used by the rebrush flow: `_class_brush_exe` (locator), `_read_envi_mask` / `_write_envi_mask_like` (ENVI mask IO), `_run_class_brush_only` (subprocess runner with intermediate-file cleanup and registry-based cancel), `_align_mask_to_crop_frame` (geotransform-aware resampler), `_render_comparison_png`, `_render_ml_classification_png`, `_render_brush_comparison_png`. Wired by `brush.init(app_state, _rebrush_procs, _rebrush_procs_lock)` so the cancel handler in `app.py` and the cache pin logic in `cache_retention.py` share the same registry. |
| `mapping_cmd.py` | `_build_mapping_cmd(fire, params, save_state=None, load_state=None) -> list[str]`: assembles the `fire_mapping_cli.py` argv from a fire + a params dict, applies `validation._validate_param` to every entry, computes the bounds-checked sample size from `crop_w*crop_h*sample_rate`, and injects the `-u` flag for line-buffered child stdout. Wired by `mapping_cmd.init(app_state)`. |
| `prepare.py` | Synchronous prepare + accept flow. `_prepare_fire_sync(fire_numbe, padding=None)` — crops the raster + VIIRS to the fire bounding box (with padding), accumulates VIIRS hot pixels into a hint mask, generates preview PNGs, and writes the per-fire `.web_cache/<FIRE>/` layout. The `PREPARING` test-and-flip is atomic under `state.lock` so two concurrent prepares cannot both pass the guard and race on cache files; MAPPING is allowed through because the serial worker calls back into prepare to handle mid-sweep padding changes. Rasterize-polygon failures are logged to stderr instead of silently dropping the perimeter. `_ensure_brush_comparison_in_cache` makes sure the brush comparison PNG exists for the gallery view. `_accept_fire_sync` is the canonical accept implementation — refuses re-entry for the same fire (raises `RuntimeError` if `fire_numbe` is already in `_accept_in_progress`), copies the chosen run from `.web_cache/` into `<output_root>/<FIRE>/`, calls `kml._export_kml` to emit `<FIRE>.shp` + `<FIRE>.kml` (warn-and-continue on failure), appends a row to `accepted_params.csv` (open-coded atomic write with file fsync + parent-dir fsync) under `_accept_file_lock`, and registers/deregisters from `_accept_in_progress` so the cache sweeper holds off mid-copy. Failures to update `fire_status.yaml` log a WARNING to stderr instead of being silently swallowed. Wired by `prepare.init(app_state, _set_fire_status, _accept_in_progress, _accept_in_progress_lock, _accept_file_lock, _CSV_FIELDNAMES)`. |
| `workers.py` | Mapping worker family. **Top-level entry points**: `_batch_map_worker(fire_numbes, session_hash)` drives a list of fires sequentially, delegating each to `_serial_map_worker`; `_serial_map_worker(fire_numbe, settings, k_runs, k_jitter, session_hash)` runs the N×K sweep for one fire by walking the five phase helpers (`_serial_setup` → `_serial_snapshot_run0` → `_serial_run_replicate` looped → `_serial_handle_cancel` → `_serial_finalize`). `_serial_handle_cancel` does its file operations and state mutations under `_gpu_lock`, captures the values it needs into locals, and only then drops the lock to call `_save_fire_state` and `_push_notification` — disk I/O for one fire's cancel cleanup never blocks mapping/rebrush requests on a different fire. `_jitter_hdbscan(base, run_idx, step)` returns a fan-out-jittered min-samples value. Wired by `workers.init(app_state, helpers)` — the helpers dict supplies `_gpu_lock`, `_batch_cancel`, `_SUBPROCESS_SILENCE_TIMEOUT`, plus the small in-app helpers (`_set_fire_status`, `_get_recommended_settings`, `_clone_setting`, `_stream_subprocess`) that each reach back into `app.py`'s module-level state. Sibling-module functions like `_save_fire_state`, `_compute_agreement`, `_compute_ml_area`, `_overlay_mask_on_post`, `_generate_result_preview`, `_build_mapping_cmd`, `_prepare_fire_sync`, `_save_stage_timings`, `_push_notification` are imported directly. |

### Handler subpackage

The `handlers/` directory contains 10 mixin modules that compose into
`FireHandler` via MRO. Each mixin file declares stub globals (`state`,
`_gpu_lock`, `_gpu_queue`, `_batch_thread`, every shared lock, every
shared registry, every helper callable) at the top, and an
`init(app_state, helpers)` that copies the helpers dict into its
namespace. `_wire_handlers` in `app.py` builds the helpers dict and
calls each mixin's `init`. The unmodified method bodies look up
shared state through these module-level names at call time.

| File | Purpose |
|---|---|
| `handlers/__init__.py` | Re-exports the mixin classes and submodule references used by `_wire_handlers`. |
| `handlers/base.py` | HTTP plumbing shared across every other mixin — `_send_json`, body parsing with size limits, header helpers, session/IP helpers (`_session_hash`, `_client_ip`, role check), CSRF/origin enforcement, and the routing dispatch in `do_GET` / `do_POST`. |
| `handlers/auth.py` | Login page, login POST, logout, admin dashboard page, IP-list admin endpoints, session/IP status JSON for the UI top bar. |
| `handlers/fire_list.py` | Home page (fire list HTML), per-fire mapping page HTML, paginated fires JSON, hidden-fires JSON, per-fire notes / hide / unhide. |
| `handlers/fire.py` | Per-fire status / console / progress JSON, per-view preview PNG GETs, comparison and brush-comparison PNG GETs, prepare POST, abort POST. |
| `handlers/mapping.py` | Single-shot SSE mapping (`handle_api_map` — relocated here from `app.py` after the container refactor), accept POST, recommended-settings GET/POST, default-settings GET/POST, presets GET, per-fire preset POST. The single-shot map flips `fire.status` to `MAPPING` atomically under `state.lock` before joining the GPU queue, so two simultaneous POSTs cannot both enqueue duplicate runs. |
| `handlers/serial.py` | Serial-map start, cancel, results gallery JSON, per-run image GET, accept-best POST. Spawns the worker thread that calls `workers._serial_map_worker`. The MAPPING test-and-set (and the snapshotting of `serial_prev_status`, `serial_results`, `serial_settings`, `console_log`, `progress`) runs under `state.lock` before the worker thread is spawned. |
| `handlers/rebrush.py` | Rebrush start (atomically claims the `_rebrush_procs[fire_numbe]` slot under `_rebrush_procs_lock` with a `None` sentinel, then `class_brush.exe` replaces the sentinel with its real `Popen`; sentinel is freed by a `finally` if no Popen ever ran), rebrush cancel, brush-state IO. |
| `handlers/batch.py` | Batch start (`handle_api_batch_map` — relocated here from `app.py`; admin-only), status JSON, batch cancel POST. |
| `handlers/ops.py` | Queue / notifications / cache / years / year-switch / diagnostic-report / cancel-audit endpoints. |
| `handlers/static.py` | Serves files from `static/` with content-type detection. Path traversal is rejected. |

### Templates and static assets

| Path | Purpose |
|---|---|
| `templates/login.html` | Login page. |
| `templates/pending.html` | "Waiting for IP approval" page shown to clients whose IP is in pending state. |
| `templates/fire_list.html` | Home page. Embeds the toast container + `pollNotifications()` so cross-page events render even when the user is not on a fire detail page. |
| `templates/fire_mapping.html` | Per-fire detail page — image viewer, parameters form, console, progress bar, results gallery, preset buttons, cancel-with-reason flow. Polls `/api/fire/<FIRE>/progress`, `/api/queue`, and `/api/notifications`. |
| `templates/admin.html` | Admin dashboard. IP access list, batch operations, cache retention card (live size + per-year breakdown, config inputs, dry-run + run-now buttons), audit log view. |
| `static/style.css` | All CSS — fire list, mapping page, queue banner, progress pills with pulse animation, toast container with kind-coded coloring and slide animations, preset buttons. |
| `static/help.js` | Client-side help / contextual hints. |
| `static/BC-Wildfire-Service-logo.png` | Header logo. |

---

## Maintenance notes (for future contributors)

- **Adding a new route**: pick the appropriate mixin file in
  `handlers/`, add the `handle_*` method, and add the route mapping
  in `handlers/base.py`'s dispatch. If the method needs a new shared
  helper from `app.py`, add it to the helpers dict in
  `_wire_handlers` and add a stub `_helper = None` at the top of
  every mixin file (the same pattern used today). The init wiring
  will pick it up automatically.
- **Adding a new sibling module**: write an `init(app_state, …)` at
  the top, declare its module-level state stubs (`state: AppState =
  None`), import sibling helpers directly when those helpers
  read their own state (e.g. `_save_fire_state`). Wire `init` from
  `app.init_app` in dependency order — anything that captures
  `_save_fire_state` must run after `persistence.init`; anything
  that uses the `helpers` dict must run before `_wire_handlers`.
- **Re-exporting**: every name extracted into a sibling module is
  also re-imported into `app.py` so external callers and the
  handler mixins (which were generated with `from app import …` in
  mind) keep working unchanged. When you extract a new name, add
  the re-export.
- **Tests**: an audit-driven test suite lives under `tests/audit/`
  (run `bash tests/audit/run_all.sh`). It covers atomic-YAML write
  invariants, auth helpers, cache retention pin/evict logic, the
  CSV race, fire-numbe regex, fire-state round-trip, jitter fan-out,
  notifications, progress / stage detection, template substitution,
  parameter validation, year detection, and handler concurrency
  (the test-and-set patterns from `AUDIT_REPORT.md` items C3/C4/C5
  are exercised in `test_handler_concurrency.py`). A baseline run
  should report 284 pass / 0 fail. The `_wire_handlers` helpers dict
  is the primary point of failure for refactors — every mixin must
  end up `is`-identical to `app` for every shared name.

---
