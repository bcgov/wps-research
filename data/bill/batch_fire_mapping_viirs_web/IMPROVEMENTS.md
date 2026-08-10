# Potential improvements — batch fire mapping web app

Prioritised by impact on the actual job: **a fire manager or geospatial
analyst producing a defensible fire perimeter, quickly and
interactively.**

Sources: this codebase (~21k lines across 28 modules + handlers), the
wider `wps-research/py` toolset (~287 scripts), and issues observed
while debugging this session.

Rough effort: **S** hours · **M** a day or two · **L** a week+ ·
**XL** a project.

---

## Tier 0 — Correctness and trust
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

## Tier 1 — Speed of the core loop
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

## Tier 2 — Interaction and editing
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

## Tier 3 — Mapping quality
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

## Tier 4 — Situational awareness
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

## Tier 5 — Robustness and operations
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

## Tier 6 — Bigger bets
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

## Quick wins worth doing first

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
