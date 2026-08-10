# Backlog — batch_fire_mapping_viirs_web

Running list of requested work. Newest requests appended; move items to
**Done** with the date when shipped.

---

## Open

### 1. Add KGC algorithm to fire mapping ML methods
Add KGC as a selectable classifier alongside the current
t-SNE + Random Forest + HDBSCAN pipeline.

Notes / open questions:
- Needs to slot into the existing CLI `--method` style selection so the
  serial sweep can compare it against the current method on the same
  AOI stack and hint.
- Should report the same result contract as today (classified `.bin` on
  the AOI grid, agreement %, ML area) so the results gallery, agreement
  scoring and overlay rendering work unchanged.
- Worth benchmarking against the existing method on a set of fires with
  known-good perimeters before offering it as a default.

### 2. Line / rectangle based image annotation
Interactive annotation in the fire view, beyond the current brush.

Notes:
- Lines for cutting a classification (e.g. severing a spurious
  connection between two burn patches).
- Rectangles for include/exclude regions — e.g. forcing an area to
  burned/unburned before or after classification.
- Must render into the same crop pixel space as the previews so it
  survives the AOI-grid re-render, and should be exportable with the
  result.

### 3. Improve coverage prediction using historical coverage polygons
Calibrate the "Expected next coverage" forecast against what actually
arrived.

Notes:
- ESA's published KML footprints are simplified (corners joined), so
  predicted edges are approximate. We now record real per-acquisition
  coverage in `<stack>_dates.json`.
- Compare predicted footprint vs realised coverage per relative orbit
  and build an empirical correction (or replace the plan geometry with
  historical polygons keyed by relative orbit).
- Would also let us report a realistic probability of usable imagery
  (cloud included) rather than "planned = covered".

### 4. Apply BCWS output file naming conventions
Bring exported products in line with BCWS conventions.

Notes:
- Applies to the download bundle: classified raster, KML/shapefile
  perimeter, comparison figures.
- Need the authoritative convention (fire number, date, product type,
  version/iteration) before implementing.
- Should be applied at export time, not to internal cache filenames,
  so internal paths stay stable.

---

## Done

- Remove AOI padding (all previews share one grid; fixes split-view
  misalignment at the source) — 2026-08-08
- Sentinel-2 acquisition plans: download, cache, per-AOI "expected next
  coverage" visualisation — 2026-08-08
- Auto-open a newly created fire when ready ("Open when ready"
  checkbox) + concurrent AOI preparation — 2026-08-10
- JPEG previews for continuous-tone views + gzip JSON responses —
  2026-08-10
