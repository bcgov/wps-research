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

### 5. Core Sentinel-2 downloading / compositing: calculate actual data coverage from new frames on receipt
Right now coverage is only known once an AOI stack is built.

Notes:
- Compute and store per-frame valid-data footprints when a SAFE zip
  lands, not when a fire needs it.
- Enables: accurate "what do we actually have" per tile/date without
  opening zips; faster AOI builds (skip frames known not to cover);
  and feeds item 3 (historical coverage calibration).
- Store beside the zip or in a small index keyed by (tile, acquisition).

### 6. Catalogue data types, formats and locations
No single description exists of what the app stores, where, in what
format, and for how long.

Notes:
- Cover: MRAP mosaics, L2A SAFE zips, AOI stacks on `/ram`, previews
  and proxies, hint masks, classified rasters, serial run state,
  `fire_state.yaml`, VIIRS `.nc` + shapefiles, acquisition-plan cache,
  exports.
- For each: format, path, producer, consumer, lifetime (session /
  daily / permanent), and approximate size.
- Prerequisite for retention policy, disk-space guards, and backup.

### 7. Make VIIRS search, downloading and processing reliable
Currently disabled by default (`VIIRS_DOWNLOAD_ENABLED = False`) and
~735 cached `.nc` files fail to parse on every startup.

Notes:
- Quarantine unreadable `.nc` files instead of retrying forever.
- Verify downloads (size/checksum) before accepting.
- Make per-AOI download the normal path; re-enable by default once
  reliable.
- Clear reporting when VIIRS is unavailable so the red-wins fallback is
  an explicit choice rather than a silent one.

### 8. Auto-regenerate VIIRS (LAADS) keys
Tokens expire and renewal is manual.

Notes:
- May need a headless browser to complete the Earthdata login flow.
- Alternative worth checking first: whether LAADS offers a
  programmatic token refresh / long-lived app credential, which would
  avoid browser automation entirely.
- Detect expiry from the 401/403 response and renew automatically,
  logging clearly if renewal fails.

---

## Done

Reverse chronological. Dates are when the change was handed over.

### 2026-08-10

- **Auto-open a newly created fire when ready.** "Open when ready"
  checkbox beside Logout (default on, remembered). Cancelled if the
  user navigates away from the fire list, so a fire finishing later
  cannot yank them out of what they are doing. Also seeded from the
  page transition, because `new_fire.js` is a cached static file and
  the sessionStorage handoff alone was unreliable.
- **Concurrent AOI preparation.** `viirs_concurrent_jobs` 1 → 2. The
  work was already queued and backgrounded, but a single dispatch
  thread made concurrent creation serial in practice.
- **Fix `class_brush` flag mismatch.** Brush post-processing had not
  run at all against a binary older than `--no-intermediates`: the C++
  parser stops at an unknown flag and treats it as the filename, so
  `brush_size` parsed from an absolute path as 0. The CLI now probes
  the binary's usage text and passes only flags it supports. *Rebuild
  `cpp/class_brush.exe` to regain `--no-intermediates`.*
- **Progressive preview loading.** ~400 px JPEG proxy served via
  `?lowres=1`, painted immediately and replaced by the full image.
  Scaled to the same geographic framing and overlaid with the same
  vectors, so the swap shows only a sharpness change. ~5.06 s → ~35 ms
  to first paint at the measured link rate.
- **JPEG previews + gzip JSON.** `pre`/`post`/`diff*` gain a JPEG twin
  (~11x smaller); masks stay PNG. JSON responses gzip when the client
  supports it. Both report format and savings to the browser console.
- **Faster startup.** The daily province-overview regeneration moved
  off the startup path; only missing overviews block. Startup went
  from minutes to seconds on regeneration days.
- **Plan completeness diagnostics.** Per-satellite datatakes/day,
  window coverage and download size, with an explicit
  complete/truncated verdict.

### 2026-08-09 — Sentinel-2 acquisition plans

- **New `acq_plans.py`**: fetch, parse and cache ESA's S2A/S2B/S2C
  plans on the ramdisk; refresh at startup and daily, 15 min while
  incomplete.
- **"Expected next coverage" panel**: every planned pass over the AOI
  within the horizon, soonest painted on top, with per-pass share of
  the AOI and how much is new ground.
- **Resilience**: parallel downloads; multi-transport fetch (system CA
  → certifi → `$SSL_CERT_FILE` → curl → unverified as a last resort,
  because the network intercepts HTTPS with an expired certificate);
  short-read detection; salvage of truncated KMLs; per-satellite merge
  so one failure cannot drop a satellite; local-KML offline fallback.
- **Per-record validation.** The original all-or-nothing check
  discarded two complete plans over one legal antimeridian coordinate
  each (longitudes just outside ±180). Bad records are now dropped and
  counted instead.

### 2026-08-08 — Geometry and display correctness

- **Removed AOI padding entirely** (pinned to 0 in prepare, settings,
  CLI and persistence; control removed from the UI). Padding was the
  only thing that moved the AOI window after creation, and every move
  put previews on a different grid — the root cause of repeated
  split-view misalignment. All views now share one grid by
  construction.
- **Run overlays re-rendered onto the current AOI grid** after any
  prepare, source switch or sweep, with a self-healing size check when
  a preview is served.
- **Georeferencing shipped with the image** (`X-Geo-*` headers) so a
  pane can never be paired with another raster's extent.
- **Split-view sync**: identical grids copy the transform verbatim;
  differing grids reconcile through the native CRS.
- **Press-and-hold flicker** comparison, instant (direct `src` swap)
  and drift-free (exact transform restore).
- **Per-pane view selectors** in split view, replacing the ambiguous
  single dropdown.
- **Empty ML results no longer create a view** (an empty mask produced
  a PNG identical to post-fire, mislabelled as "ML classification").

### 2026-08-07 — Results pipeline

- **Fixed serial results never appearing.** The mapping CLI writes its
  outputs beside its input image, which moved to the ramdisk; the
  post-run code still looked in the fire cache. Classified masks,
  comparison figures and `serial_N.png` are now found where they
  actually land — this was the "all runs failed" symptom despite
  F1 85–89%.
- **Classified-mask naming** derived from the stack rather than the old
  `<fire>_crop.bin` convention, with legacy names as fallbacks.
- **Exhaustive clustering diagnostics** on every run: mask/hint pixel
  counts, geotransforms, intersection/union/IoU, and a named reason
  for every non-computable agreement.
- **Hint view registered whenever a hint mask exists**, rather than
  only when the generic overlay rendered.

### 2026-08-06 — Performance and data plumbing

- **Per-AOI on-demand stacks** replacing the province-wide stack;
  L2-recent compositing with parallel per-tile extraction and
  per-acquisition date attribution.
- **"L2 coverage by acquisition" plot**, later gaining satellite
  prefixes (`S2A+S2B · 2026-08-04`), backfilled for existing fires
  from SAFE filenames.
- **Post-source switching** (L2 recent ↔ MRAP composite) with per-source
  preview stashes and background prebuild of the other source.
- **Client-side preview caching and warming**, optimistic first paint,
  and `Cache-Control` on previews.
- **Timing instrumentation** throughout (TTFB vs transfer vs decode vs
  bake), which is what made the later diagnosis possible.
- **Atomic preview writes** (tmp + rename), fixing images served
  mid-rewrite as correct-width but truncated-height.
- **Background prebuild no longer mutates the visible source**, fixing
  new fires opening on MRAP instead of L2.
- **Honest status labels**: `cropping` → "Build AOI stack", `Crop:` →
  `AOI:`, and detail strings describing what actually happens.

---

## Notes

- The C++ tools live in `wps-research/cpp` and are **not** rebuilt by
  this app. `class_brush.exe` in particular must be recompiled after
  source changes or the flag probe will keep it in degraded mode.
- Diagnostics added during debugging are intended to be **permanent**;
  they are what made several of the above findable at all.
