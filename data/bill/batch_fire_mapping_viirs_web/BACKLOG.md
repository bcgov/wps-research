# Backlog — batch_fire_mapping_viirs_web

Single list for this project: requested work, decisions waiting on
input, improvement ideas, and what has shipped. Supersedes the separate
IMPROVEMENTS.md, whose contents are merged in below unchanged.

Rough effort: **S** hours · **M** a day or two · **L** a week+ ·
**XL** a project.

Sources: this codebase (~21k lines across 28 modules + handlers), the
wider `wps-research/py` toolset (~287 scripts), and issues observed
while debugging this session.

Rough effort: **S** hours · **M** a day or two · **L** a week+ ·
**XL** a project.

---

Sections:

1. **Requested work** — asked for, not yet built.
2. **Decisions awaiting input** — designed, needs a choice before
   building.
3. **Improvement ideas** — unprompted, grouped by impact on the job of
   producing a defensible perimeter quickly.
4. **Quick wins** — cheapest items with disproportionate value.
5. **Done** — shipped, newest first, with the reason where it matters.

---

## 1. Requested work

### 1. Line / rectangle based image annotation
Interactive annotation in the fire view, beyond the current brush.

Notes:
- Lines for cutting a classification (e.g. severing a spurious
  connection between two burn patches).
- Rectangles for include/exclude regions — e.g. forcing an area to
  burned/unburned before or after classification.
- Must render into the same crop pixel space as the previews so it
  survives the AOI-grid re-render, and should be exportable with the
  result.

PARTLY DONE (2026-08-12): the manual eraser delivers square-box
REMOVAL — sized in image pixels, live preview, Revert, optional
"Outside BCWS only", E to toggle — and the coordinate conversion,
in-place mask editing and re-scoring it needed are reusable. Still
outstanding from this item:
- a LINE tool (cut a thin gap through a classification, rather than
  erase a blob);
- RECTANGLE regions, including the ADD direction — forcing an area to
  burned, which the eraser cannot do since it only clears;
- persisting the annotations themselves as vectors alongside the
  result, so an edit can be reviewed or undone individually rather than
  only reverted as a session.

### 2. Improve coverage prediction using historical coverage polygons
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

### 3. Apply BCWS output file naming conventions
Bring exported products in line with BCWS conventions.

Notes:
- Applies to the download bundle: classified raster, KML/shapefile
  perimeter, comparison figures.
- Need the authoritative convention (fire number, date, product type,
  version/iteration) before implementing.
- Should be applied at export time, not to internal cache filenames,
  so internal paths stay stable.

### 4. Core Sentinel-2 downloading / compositing: calculate actual data coverage from new frames on receipt
Right now coverage is only known once an AOI stack is built.

Notes:
- Compute and store per-frame valid-data footprints when a SAFE zip
  lands, not when a fire needs it.
- Enables: accurate "what do we actually have" per tile/date without
  opening zips; faster AOI builds (skip frames known not to cover);
  and feeds item 3 (historical coverage calibration).
- Store beside the zip or in a small index keyed by (tile, acquisition).

### 5. Catalogue data types, formats and locations
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

### 6. Make VIIRS search, downloading and processing reliable
Currently disabled by default (`VIIRS_DOWNLOAD_ENABLED = False`) and
~735 cached `.nc` files fail to parse on every startup.

Notes:
- Quarantine unreadable `.nc` files instead of retrying forever.
- Verify downloads (size/checksum) before accepting.
- Make per-AOI download the normal path; re-enable by default once
  reliable.
- Clear reporting when VIIRS is unavailable so the red-wins fallback is
  an explicit choice rather than a silent one.

### 7. Auto-regenerate VIIRS (LAADS) keys
Tokens expire and renewal is manual.

Notes:
- May need a headless browser to complete the Earthdata login flow.
- Alternative worth checking first: whether LAADS offers a
  programmatic token refresh / long-lived app credential, which would
  avoid browser automation entirely.
- Detect expiry from the 401/403 response and renew automatically,
  logging clearly if renewal fails.

### 8. Distinguish active fire / hotspots from burned area
The system currently detects "fire/burn" as one class.

Notes:
- Active fire (thermal anomaly, currently burning) and burned area
  (already-consumed fuel) are different products with different users
  and different validation.
- VIIRS hotspots are inherently active-fire; the red-wins and BCWS
  perimeter hints are closer to burned area. Today they all feed the
  same binary classifier.
- Would likely need a third class in the classifier output, or two
  passes, plus a decision about how the two are exported.
- Relevant to the new BCWS perimeter hint: an official perimeter
  includes ground already burned AND ground still burning.

---

## 2. Decisions awaiting input

### D1. How to visualise scaled imagery
The Custom scaling options (per-band 0-1, percentile trims, global and
per-pixel min-max, per-pixel L2, z-score, robust z, log/dB) change what
"similar" means to the clustering, but scaled data is not a picture:
after z-score or per-pixel L2 there is no natural black or white point,
so ANY rendering needs a second, display-only stretch. If that stretch
is invisible, the operator is looking at an image whose appearance comes
from a transform they did not choose.

Three options, none implemented pending a decision:

- **Option A — "Scaled" entry in the view dropdown.** (M) Render the
  scaled stack as false colour with a fixed display stretch (e.g. 2-98%
  per band), labelling the pane `Scaled: robust_z (display 2-98%)`.
  Honest because the display stretch is named and separate from the
  analysis scaling. Risk: comparing two scalings visually really
  compares their INTERACTION with the display stretch, which can invert
  the apparent result.
- **Option B — before/after histogram panel.** (M) No image at all:
  per-band histograms of raw vs scaled with min/median/max. Answers
  "what did this do to my data?", which is the actual question, and
  cannot mislead the way A can.
- **Option C — both**, B as the default and A for spot-checking, with
  the display stretch always named on screen.

Recommendation: **B first.** Cheaper, and it tells you whether a
scaling is doing anything useful before spending a clustering run
finding out.

### D2. Should scaling also live in `cpp/kgc2/kgc.cpp`?
Scaling is currently implemented in Python at `reduced_stack()`, which
covers KGC, the t-SNE/RF/HDBSCAN pipeline and Download imagery from one
place. A C++ implementation inside `kgc.cpp` was requested; doing it
there alone would leave the other two paths unscaled. Options: keep the
single Python implementation, duplicate it in C++ (two places to keep
in step), or move it to C++ and have the other paths call the binary.

### D3. Sentinel-2 tile labels
The fire number labels are now white on opaque black. The tile labels
(e.g. `10UCD`) are still magenta on 55%-transparent black. Match them or
leave them visually distinct?

---

## 3. Improvement ideas

### Tier 0 — Correctness and trust

*If these are wrong, everything downstream is wrong, and users will
stop believing the tool.*

1. **Regression tests for the AOI-grid invariant** (M)
   Split-view misalignment recurred ~6 times this session because
   nothing enforced "all previews share one geotransform". Add a test
   that builds a fire, runs a sweep, and asserts every preview and
   every `serial_N` shares the crop's grid. This one test would have
   caught every recurrence.

2. **Golden-fire regression suite** (L)
   2–3 fires with hand-checked perimeters, run end-to-end nightly,
   asserting agreement % within tolerance. Detects silent quality
   regressions from parameter or dependency changes — currently
   invisible until someone eyeballs a result.

3. **Fix `class_brush.exe: brush_size must be > 0`** (S)
   Fails on *every* run today, so brush post-processing never happens
   and results are rougher at the edges than intended. Pure argument
   mismatch; noticed in logs but never chased.

4. **Quarantine corrupt VIIRS `.nc` files** (S)
   ~735 files fail `NetCDF: Unknown file format` on every startup and
   are retried forever. Sanity-check size/header once, move to
   `_quarantine/`, log a summary. Cuts startup time and log noise.

5. **Provenance record per exported perimeter** (M)
   Stamp each export with: source imagery dates + platforms, hint mode,
   algorithm + parameters, agreement %, operator, timestamp, app
   version. Required for anything that ends up in an official record,
   and cheap now that most of it is already tracked.

6. **AOI readiness check before using a new MRAP mosaic** (S)
   The active date switches the moment `<date>_mrap.bin` appears.
   If the refresh writes in place, a build can start against a
   half-written mosaic. Check header presence, size-vs-header, and
   mtime stability before switching.

---

### Tier 1 — Speed of the core loop

*The loop is: draw AOI → prepare → map → inspect → adjust → export.
Every second here is multiplied by every fire, every day of a season.*

7. **Persistent AOI stack cache keyed by bbox hash** (M)
   Re-drawing a similar AOI re-extracts the same L2 zips (~13 s/tile).
   Cache by rounded bbox + source + date; reuse when the new AOI is
   contained by a cached one. Biggest single win for iterative work.

8. **Tile-level L2 extraction cache** (M)
   Cache the decoded 20 m band stack per (tile, acquisition) rather
   than per AOI. Adjacent fires in the same tile then cost almost
   nothing. Bounded LRU on disk.

9. **Progressive preview loading** (M)
   Serve a small preview (~400 px) immediately, then the full one.
   First paint drops to well under a second on a slow link. The JPEG
   change helped; this removes the wait almost entirely.

10. **Warm only the current source** (S)
    12 preview combinations ≈ 80 MB of background traffic per fire
    open, competing with the image being waited on. Warm 6; fetch the
    other source on demand.

11. **Reuse t-SNE embedding across settings** (M)
    Already cached per setting; the embedding often doesn't depend on
    the varied parameter (e.g. `hdbscan_min_samples`). Detect that and
    skip re-embedding — could cut sweep time by half.

12. **Cancel in-flight work when a fire is closed or re-prepared** (M)
    Abandoned L2 extractions and sweeps keep running, competing for
    disk and GPU with what the user is actually looking at.

13. **Batch "prepare all" for a set of fires** (M)
    Select N fires from the list and queue them overnight. Combined
    with concurrency (now 2), a morning's work is ready on arrival.

---

### Tier 2 — Interaction and editing

*Where an analyst actually earns the result: fixing what the model got
wrong, fast.*

14. **Line / rectangle annotation** (M) — *in backlog*
    Cut spurious connections; force include/exclude regions. Must live
    in crop pixel space so it survives re-render, and be exportable.

15. **Undo/redo stack for all edits** (M)
    Brush and annotation edits are currently one-way. An undo stack
    turns cautious editing into confident editing.

16. **Polygon-level accept/reject** (M)
    After classification, let the user click connected components to
    keep/drop them. Most correction effort is "that patch isn't part of
    this fire" — one click each rather than brushing.

17. **Side-by-side result comparison across runs** (M)
    Currently one result at a time. A 2×2 or slider comparison of runs
    would make choosing between sweep outputs much faster.

18. **Keyboard shortcuts** (S)
    View switching, flicker, accept, next fire. Power users live in
    this app; mousing to a dropdown for every comparison is slow.

19. **Persist zoom/pan per fire** (S)
    Reopening a fire returns to the fitted view, losing the area of
    interest. Remember the last viewport per fire.

20. **Edit the AOI after creation** (M)
    Today a wrong AOI means recreating the fire and losing its history.
    Allow resize/move with re-prepare.

---

### Tier 3 — Mapping quality

*Better perimeters with less manual correction.*

21. **KGC algorithm** (L) — *in backlog*
    Slot into the existing `--method` selection so sweeps can compare
    it head-to-head on the same stack and hint.

22. **Multi-date compositing for cloud gaps** (L)
    Cloud is the main reason a pass yields nothing usable. Compose the
    best cloud-free pixel across recent acquisitions per pixel, rather
    than newest-wins. Would materially improve results in smoky or
    cloudy periods — the exact conditions during active fire.

23. **Explicit cloud/smoke masking** (L)
    Use the L2A scene classification layer (already in the SAFE) to
    exclude cloud/shadow from sampling and from the anomaly. Cheap
    relative to its effect on agreement.

24. **dNBR / BARC severity products** (M)
    `barc.py` already exists in the repo. Severity classes alongside a
    binary perimeter would answer questions the current output can't.

25. **Sentinel-1 SAR fallback** (XL)
    Cloud-independent. For a fire that stays clouded for a week, SAR is
    the difference between a perimeter and none.

26. **Auto-suggest the best run** (M)
    A sweep produces 12 results and the operator picks by eye. Rank by
    agreement + area plausibility + edge smoothness, and pre-select.

27. **Active learning from operator corrections** (XL)
    Every brush stroke is a labelled example. Accumulate them per
    region/fuel type and fine-tune. Over a season this could
    meaningfully cut correction effort.

28. **Uncertainty visualisation** (M)
    Show where the classifier was marginal (cluster score near the
    threshold) so attention goes where it matters instead of scanning
    the whole perimeter.

---

### Tier 4 — Situational awareness

*Answering "what should I work on next, and when will I have data?"*

29. **Province-wide planned-coverage overlay** (M) — *was item 5 of the
    acquisition-plan work, deliberately deferred*
    Tomorrow's swaths on the new-fire map, so an operator can see which
    active fires get fresh imagery next.

30. **Historical coverage calibration** (M) — *in backlog*
    Replace/adjust simplified plan footprints with realised coverage
    polygons per relative orbit, and report a realistic probability of
    usable imagery rather than "planned = covered".

31. **Notify when new imagery lands for a watched fire** (M)
    "Tell me when this fire has a new cloud-free image" is the actual
    question. Combine acquisition plans with product arrival.

32. **Fire list sorted by actionability** (S)
    Rank by "new imagery available since last mapped", not creation
    date. Turns the list into a work queue.

33. **Season/day summary view** (M)
    Fires mapped, area burned, agreement distribution, imagery
    availability. Useful for reporting and for spotting a systematic
    problem early.

---

### Tier 5 — Robustness and operations

*Reduce the chance that a bad day for infrastructure becomes a bad day
for the user.*

34. **Fix the TLS interception properly** (S, but external)
    An expired self-signed root is intercepting HTTPS. We work around
    it for acquisition plans; it will keep breaking other outbound
    HTTPS. Worth escalating rather than accreting workarounds.

35. **Remember per-host that Range requests are refused** (S)
    ESA answers 403 to `Range:`; each truncation wastes a round trip
    and adds log noise.

36. **Disk-space guards on `/ram`** (S)
    Every AOI stack lives on the ramdisk. Concurrency now 2, and a
    large AOI is GBs. Check free space before building and fail with a
    clear message rather than a confusing GDAL error.

37. **Structured logging with levels** (M)
    Logs are prose `sys.stderr.write` calls. Levels + a request id
    would make the kind of debugging done this session much faster.

38. **Health endpoint** (S)
    One JSON with: raster date, plan cache age and per-satellite
    counts, queue depth, ramdisk free, last error per subsystem.
    Screenshot-able, scriptable, and would have short-circuited several
    rounds of back-and-forth here.

39. **Config file instead of CLI flags** (S)
    Flag count is growing (`--acq_plans_insecure`, `--padding`,
    `--viirs_concurrent_jobs`, …). A YAML config with the current
    values echoed at startup reduces launch-script drift.

40. **Graceful degradation when GDAL/CUDA is missing** (M)
    Currently a missing dependency surfaces as a stack trace mid-run.
    Detect at startup, disable the affected feature, say so plainly.

---

### Tier 6 — Bigger bets

*High ceiling, high cost. Worth discussing before committing.*

41. **Multi-user awareness** (L)
    Show who else has a fire open and whether a result was accepted by
    someone else. Prevents duplicated or conflicting work as the team
    grows.

42. **Result versioning and audit trail** (L)
    Keep every accepted perimeter with its provenance; allow diffing
    across time. Fire perimeters are revised repeatedly — the history
    matters.

43. **Server-rendered tiles for very large AOIs** (L)
    Current previews cap at 2000 px. A 3000×3200 AOI is already
    downsampled ~40%. Real tiling would allow full-resolution
    inspection of large fires.

44. **Web-based hint drawing** (M)
    Draw or edit the hint perimeter directly instead of relying on
    VIIRS/red-wins. Would help small or early fires where the automatic
    hint is poor (`Hint Size: 0.0 ha` appears often in this session).

45. **Time-series animation of fire growth** (M)
    Sequence acquisitions to animate progression. Compelling for
    briefings; largely reuses the existing date-coverage machinery.

46. **Auto-detect new fires from VIIRS clusters** (XL)
    Propose AOIs automatically from VIIRS hotspot clusters, so
    operators confirm rather than draw. The largest possible reduction
    in per-fire effort.

47. **Export to BCWS systems directly** (M)
    Beyond file naming — push perimeters into the downstream system so
    the export step disappears entirely.

48. **Offline/degraded mode** (L)
    A field deployment with intermittent connectivity should still map
    from cached imagery. Much of the groundwork exists (ramdisk cache,
    local KML fallback).

---

---

## 4. Quick wins

Cheapest things with disproportionate value, extracted from above:

| # | Item | Effort |
|---|------|--------|
| 3 | Fix `class_brush` argument bug | S |
| 4 | Quarantine corrupt VIIRS files | S |
| 38 | Health endpoint | S |
| 18 | Keyboard shortcuts | S |
| 10 | Warm only the current source | S |
| 32 | Sort fire list by actionability | S |
| 19 | Persist zoom/pan per fire | S |
| 1 | AOI-grid invariant test | M |

---

## 5. Done

### 2026-08-12 — KGC method, scaling, band control, manual editing

- **KGC clustering added as a second ML method** (was Requested work
  item 1). `cpp/kgc2` is built on demand (rebuilt whenever the binary is
  older than the source), run on the ramdisk in a per-fire working
  directory, and band 1 of its six-band product — the binary selected
  class — becomes `<fire>_classified.bin`. From there the existing
  pipeline is reused unchanged: brushing with the BRUSH parameters,
  agreement, ML area, `serial_1.png` thumbnail, the results gallery with
  its Accept button, polygonisation and export. An `hdbscan` checkbox
  (unchecked = KGC) swaps the parameter sections and the Map Fire
  buttons, with BRUSH visible in both because brushing runs on whichever
  mask was produced. Runs in the background, concurrently across fires,
  with a cancel that terminates the subprocess.

  *Original request, kept for reference:*

  **Add KGC algorithm to fire mapping ML methods**
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

- **Custom scaling** (`scaling.py`): ten methods with formulae shown in
  the picker — per-band 0-1, percentile trim per band and
  intensity-based (P%, no-clip, right-only/left-only), global min-max,
  per-pixel min-max, per-pixel L2, z-score, robust z-score, log/dB.
  Applied AFTER band selection and only to the classifier input and
  Download imagery; the displayed panes are deliberately untouched.
- **Custom bands** picker with an explicit band list that overrides the
  checkbox rules. Checkbox changes apply INCREMENTALLY — each click
  adds or removes only the bands that box governs, so hand-picked
  choices survive unrelated toggles.
- **"Diff only"** mode: keeps only the anomaly bands, implying pre-fire
  excluded and contradicting exclude-diff; the contradiction is resolved
  in one place (`band_select`) so every consumer agrees.
- **Manual eraser** with a square cursor sized in image pixels,
  live per-pixel preview taken from the post-fire imagery, Revert to a
  pre-session snapshot, and "Outside BCWS only" so a stroke straddling
  the official boundary trims only the outside. E toggles it, bound on
  window in the capture phase so focus cannot swallow the key.
- **"Restrict hint to BCWS perimeter"**: clips whichever hint is
  selected, affecting the preview, the agreement score and the
  clustering input alike.
- **"Clip to BCWS perimeter"** after brushing, in all three result
  paths (KGC, sweep, rebrush).
- **Per-fire persistence** of every new setting plus an opaque
  `ui_state`, so a fire re-opens with the same layers, sources, split,
  parameters and results. `kgc_params` records what actually ran, and
  takes precedence over the GUI copy on restore.
- **Robustness**: KGC clears stale products before launching and rejects
  a product older than the run; the written class mask is read back and
  checked for size, geotransform and projection; geotransforms are
  compared, not just dimensions; `ensure_geo()` repairs lost map info on
  open and at startup.
- **Interlaced GIF**: blink comparator built from the frames actually
  displayed (overlays included), so it works for any view/source
  combination.
- **Fixed**: `find_classified()` was being passed a fire NUMBER instead
  of the fire, so perimeter vectorization never found the raster;
  rebrush edited a different file from the one Accept promotes;
  `handlers/fire.py` scaling edits had silently failed to apply;
  "Hint Size" in the header was actually the BCWS-reported
  `fire_size_ha` and is now labelled "BCWS Size".

### 2026-08-10

- **New hint option: "BCWS perimeter".** Rasterises every BCWS fire
  polygon intersecting the AOI into a hint mask -- deliberately not
  filtered to one fire number, since this system detects burn rather
  than attributing it. Uses the same storage, per-source naming,
  mtime invalidation and CLI contract as the red-wins hints.
  "Red wins (post)" remains the default.
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

---

## Notes

- The C++ tools live in `wps-research/cpp` and are **not** rebuilt by
  this app. `class_brush.exe` in particular must be recompiled after
  source changes or the flag probe will keep it in degraded mode.
- Diagnostics added during debugging are intended to be **permanent**;
  they are what made several of the above findable at all.
- IMPROVEMENTS.md is superseded by this file. Its tiers and quick-wins
  table are reproduced above unchanged; delete the old file to avoid
  two lists drifting apart.
- Several items marked Done are verified by logic and unit-level tests
  only. GDAL and a C++ toolchain were not available where the changes
  were written, so scaling, hint restriction, geo repair and the KGC
  build path are exercised for the first time on the server.

## MRAP back end: record the acquisitions actually used

The GUI now dates delivered products by the newest Sentinel-2
acquisition behind the imagery the classification ran on
(`delivery.acquisition_datetime`). For **L2 recent** this is exact: the
application composites the product itself and records the acquisition
datetime of every contributing file in the `_dates.json` sidecar.

For **MRAP** it is an estimate, and it cannot be better than that from
inside the GUI. The MRAP composite is assembled by a back-end process
that reports which tiles it scanned but not which pixels it took, so
the GUI assumes that any tile intersecting the AOI contributed
cloud-free pixels, and reports the newest acquisition that process
would have considered. That may name an acquisition which contributed
nothing over this particular AOI.

**Proposed fix (back end, not the GUI):** when MRAP updates a pixel,
record the source acquisition datetime for that pixel — either as a
per-pixel date index raster alongside the composite (the same shape as
the L2 `_dates.json` coverage map) or, at minimum, a per-tile list of
the acquisitions that actually supplied pixels in the update. The GUI
would then read it and report an exact time, and the manifest's
"estimated" caveat could be dropped.

Until then the archive manifest states plainly that the MRAP time is an
estimate and why, so nobody downstream mistakes it for a confirmed
acquisition time.
