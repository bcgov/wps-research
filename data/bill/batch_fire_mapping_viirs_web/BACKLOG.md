# Backlog — batch_fire_mapping_viirs_web
Single working doc and, since 2026-10-01, the only Markdown file in the source tree (`BACKLOG.md`); absorbed ideas.md, IMPROVEMENTS.md, PLAN.md, README.md, BRUSHING.md, FIRE_MAPPING_ALGORITHM.md and the older 32 KB in-tree BACKLOG.md [src 10-01: the tree's other five .md files are empty (0 bytes) after their content was confirmed here — PLAN.md, FIRE_MAPPING_ALGORITHM.md and BRUSHING.md verbatim in E2, E4 and E5 (E4 except one home path, written `~/`; reader's notes added 2026-10-01, marked as such); README.md (E1), IMPROVEMENTS.md (C2) and the old BACKLOG.md reworded, with every identifier, path, number and code-block token present (README's quick-start passwords replaced by placeholders)].
Parts: A backlog (committed) · B decisions (blocked on judgement, not effort) · C parking lot (uncommitted ideas) · D completed (+ reasoning worth keeping) · E reference (how app + algorithms work) · F working agreement (how changes are made, verified, delivered) · G caveats (what this doc is not).
Notation: effort S hours · M 1–2 days · L week+ · XL project. Paths relative to `data/bill/batch_fire_mapping_viirs_web/` unless absolute. S2 = Sentinel-2. "The Maintainer" = the person who owns the codebase and the server, sets the requirements and applies the ZIPs; "the assistant" = Claude. "src 10-01" = checked against `fire_mapping_source_after_part3.zip` on 2026-10-01 (the 60 files part #3 didn't change are byte-identical to the 09-29 source, so notes on them carried over; notes touching the six code files part #3 changed were re-checked in the code); "src 09-29" = checked against `fire_mapping_source_current.zip` on 2026-09-29 (kept where it records history). "p2#n [zip]" / "p3#n [zip]" = part #2 / part #3 thread delta item n + the ZIP that shipped it; "p2 On" / "p3 On" = that thread's open item n. ⚑dep = names deprecated methods (HDBSCAN, t-SNE, Random Forest/RF): the project rule keeps them out of USER documentation; this file is for developers, so they are kept here, flagged, as history — not current guidance (The Maintainer, 2026-10-01; F6). The pipeline is deprecated but still selectable (E4 note, B D7).

## Start here
- App: BC Wildfire Service Predictive Services web app for near-real-time fire and burned-area mapping from S2 + VIIRS.
- No source code here (this file lives in the source tree as `BACKLOG.md`). App ≈61k lines (src 10-01: 60 files — the code plus `recommended_settings.yaml`; Markdown and the logo PNG not counted — 60,940 lines by `wc -l`, a net +1,090 from part #3; 59,850 at 09-29 by the same count; earlier `fire_mapping_source_export.zip` had 110 files). Get the current source ZIP from The Maintainer (latest `fire_mapping_source_after_part3.zip`, 2026-10-01; before it `fire_mapping_source_current.zip`) before any code change; don't infer the implementation from this doc (G1).
- Read: A1 + A4 (open work, known defects) → F (standing instructions, not preferences) → F7 (past mistakes, several recurred — read before diagnosing) → E (E3: how a mapping run works today).
- Ground rules: never modify `kgc.cpp`/`kgc.h`; ZIP unzips inside `wps-research/` (paths begin `data/bill/batch_fire_mapping_viirs_web/`); say explicitly the server must be restarted after applying; doc vs source disagree → source wins.
- Server paths: source `~/GitHub/wps-research/data/bill/batch_fire_mapping_viirs_web/`; outputs/instance `~/GitHub/wps-research/data/bill/fire_mapping_results_viirs/`; ramdisk working files `/ram/`; durable stack store `<outputs>/2026_mapping_results/.stacks/`; per-fire web cache `<outputs>/2026_mapping_results/.web_cache/<FIRE>/`; MRAP S2 archive `/data/mrap_bc/`.
- Updating: new request → A; uncommitted idea → C; finished → D with its reasoning (never delete D history: it records why things are as they are); mistake by either party → F7 row (+ F1–F6 rule if it generalises); new standing instruction → F1–F6; new Part → Parts line. Since 2026-10-01 this is the tree's only .md: coding threads edit it in place (the other five .md files are empty; their content is in E1, E2, E4, E5 and C2).

# Part A — Backlog (committed; the question is when)
## A1. Requested work
(Item 0, Sources panel during build, 0a–0d → done, D1.)
1. Line/rectangle annotation beyond the brush: lines cut a classification (sever a spurious link between burn patches); rectangles = include/exclude regions (force burned/unburned before or after classification); must render in the previews' crop pixel space (survive AOI-grid re-render); exportable with result. Partly done 2026-08-12: eraser = square-box REMOVAL (image-pixel size, live preview, Revert, optional "Outside BCWS only", E toggles); its coordinate conversion, in-place mask editing, re-scoring reusable. Outstanding: LINE tool (thin cut, not blob erase); RECTANGLE regions incl. ADD (force burned — eraser only clears); annotations persisted as vectors beside the result, so each edit can be reviewed/undone singly, not only session-reverted. [src 10-01: none exist.]
2. Calibrate the "Expected next coverage" forecast against what arrived (historical coverage polygons): ESA KML footprints are simplified (corners joined) → edges approximate; real per-acquisition coverage now in `<stack>_dates.json`; compare predicted vs realised per relative orbit → empirical correction, or replace plan geometry with historical polygons keyed by relative orbit; also gives a realistic probability of usable imagery (cloud included) instead of "planned = covered". (= C2 #30)
3. BCWS output file naming for exports (download bundle: classified raster, KML/shapefile perimeter, comparison figures); needs the authoritative convention (fire number, date, product type, version/iteration) first; apply at export time only, never to internal cache names (internal paths stay stable).
4. S2 download/compositing: compute + store per-frame valid-data footprints when a SAFE zip lands, not when a fire needs it (today coverage is known only once an AOI stack is built) → accurate "what do we actually have" per tile/date without opening zips; faster AOI builds (skip frames known not to cover); feeds item 2. Store beside the zip or in a small index keyed (tile, acquisition).
5. Catalogue what the app stores, where, format, how long (none exists): MRAP mosaics, L2A SAFE zips, AOI stacks on `/ram`, previews + proxies, hint masks, classified rasters, serial run state, `fire_state.yaml`, VIIRS `.nc` + shapefiles, acquisition-plan cache, exports; each with format, path, producer, consumer, lifetime (session/daily/permanent), approx size. Prerequisite for retention policy, disk-space guards, backup. Since part #3 the app also stores per-fire ML masks `<cache>/ml_masks/<n>.png` + `.json` (p3#5) and Results thumbnails `previews/serial_<id>.low.jpg`, ≤400 px (p3#4).
6. Reliable VIIRS search/download/processing: disabled by default (`VIIRS_DOWNLOAD_ENABLED = False`, year_viirs.py; src 10-01 still; override `--enable_viirs_download`); ~735 cached `.nc` fail to parse every startup. Quarantine unreadable `.nc` instead of retrying forever (= C2 #4); verify downloads (size/checksum) before accepting; per-AOI download as the normal path, re-enabled by default once reliable; clear reporting when VIIRS is unavailable, so the red-wins fallback is an explicit choice, not a silent one.
7. Auto-regenerate VIIRS (LAADS) keys (they expire; renewal manual): first check for a programmatic token refresh / long-lived app credential (avoids browser automation); else a headless browser for the Earthdata login; detect expiry from 401/403, renew automatically, log clearly if renewal fails. [src 10-01: a 401/403 is already classified as `bad_token` (year_viirs.py); nothing renews.]
8. Separate active fire/hotspots from burned area (today one "fire/burn" class): different products, users, validation; VIIRS hotspots = active fire; red-wins + BCWS perimeter hints ≈ burned area; all feed one binary classifier. Likely a third class or two passes, plus an export decision. An official BCWS perimeter includes burned AND still-burning ground.
9. MRAP back end: record the acquisitions actually used (was misfiled after D3). The GUI dates delivered products by the newest S2 acquisition behind the classified imagery (`delivery.acquisition_datetime`). L2 recent: exact (the app composites it; `_dates.json` records every contributing file's acquisition datetime). MRAP: an estimate that can't improve inside the GUI — the back end reports tiles scanned, not pixels taken, so the GUI assumes every AOI-intersecting tile gave cloud-free pixels and reports the newest acquisition that process would have considered (maybe one that contributed nothing here). Fix (back end, not GUI): per updated pixel, record its source acquisition datetime — a per-pixel date-index raster beside the composite (same shape as the L2 `_dates.json` coverage map) or at least a per-tile list of acquisitions that supplied pixels → GUI reports exact time; the manifest's "estimated" caveat goes. Until then the archive manifest says the MRAP time is an estimate and why, so nobody downstream takes it as confirmed [src 10-01: delivery.py].
10. Done → D1 p3#4 [persistence_thumbs] (the ML classification pane view waiting for the Results thumbnails, p2 O1; full text there).
11. A separate ML classification for each pane in split view (added in part #3 as the in-tree BACKLOG.md's item 9, p3#9; renumbered because A1 #9 was taken): let each split pane show its own classification, so two Results runs can be compared side by side and flickered, each over its own pane's imagery. Today one Results selection sets "ML classification" in both panes; each pane's source selector already chooses the imagery underneath — the classification is laid over that pane's own post-fire source (p3#5). Needed: a selection per pane (e.g. a Results click applies to the active pane), both selections saved with the fire and restored, a per-pane marker in the Results list, and the eraser editing the run shown in the pane being drawn on. "ML Classification — before brushing" is still drawn by the server over the left pane's imagery and ignores the Results selection (p3 O6); move it onto the same per-pane layer at the same time (part #3 deferred that move: p3 rejected (c)). [src 10-01: not implemented (p3 O5).]
12. Bring the user docs/QRG up to date with part #3: they don't yet cover single-pane flicker (p3#7), the remembered right pane (p3#6, #15) or the per-pane ML layer (p3#5) (p3 O15).

## A2. Quick wins (cheapest C2 items, disproportionate value)
#4 quarantine corrupt VIIRS files (S) · #38 health endpoint (S) · #18 keyboard shortcuts (S) · #32 sort fire list by actionability (S) · #19 persist zoom/pan per fire (S) · #1 AOI-grid invariant test (M). Done, removed: #3 fix the `class_brush` argument bug; #10 warm only the current source.

## A3. Persistence and storage
Move AOI stacks off the ramdisk — largely done (`durable.py`, D1): stacks on `/ram` (tmpfs) are lost on a power cut/reboot → every fire rebuilds on next use; `ensure_fire_stack_present()` handles that correctly but the cost is per fire; KGC scratch (`.kgc_knn_*`, `.kgc_dedup`) shares the ramdisk. Proposed: scratch stays on `/ram` (it needs the speed), stacks → `<output_root>/.stacks/` (read sequentially); needs a migration since `crop_bin` holds absolute paths (grid validation would catch any mismatch).
Still to do (#2 resolved → D1 Persistence):
1. Retention policy for dated composites now mirrored to disk (every day × every fire = unbounded). [src 10-01: none in durable.py; since part #2 products can be deleted by hand in Sources.]
3. `purge_other_aoi_stacks()` (aoi_stack.py) is defined, never called [src 10-01 still]; nothing reclaims `/ram` during a run; KGC scratch runs to hundreds of MB per fire.
4. READY/MAPPED retention, filed as "pinned indefinitely: a season of prepared-but-unaccepted fires can hold the cache above budget permanently". src 10-01 (cache_retention.py): soft pin only (rebrush-dirty fires too) — the size sweep skips them, the age sweep (`max_age_days`, default 30) evicts them and demotes READY/MAPPED → PENDING; hard pins = PREPARING/MAPPING. Remaining: they can hold the cache above `max_gb` (default 20) for up to `max_age_days`.
5. Deleting a fire leaves its AOI stacks behind (ramdisk and durable store). `orphan_stack_sets()` and `purge_orphan_stack_sets()` (durable.py) back an API report, `GET /api/orphans`, and a purge, `/api/orphans/purge`, that removes only the prefixes it is given (handlers/fire_list.py); no page calls them and nothing runs them automatically [src 10-01]. Decide whether deleting a fire should purge its stack set (cf. #1, A1 #5).

## A4. Known defects and unverified fixes
- `drawErasePreview()` is called in the overlay draw path (templates/fire_mapping.html) but defined nowhere; pre-existing; throws when reached [src 10-01 still; was D1 "Known, not fixed"].
- The Sources row "Loading…" note flashes briefly for cached images (client ticket); left as is (p2 O8).
- Browser-verify (p2 O5): the hint mask follows linked pan/zoom (`hint_mask_follows_linked_pane`, not runtime-tested); the right view selector stays enabled; thumbnails don't delay the panes. Thumbnail loading was reworked in p3#4; see the part #3 runtime checks below.
- Deleted-product resurrection (D1 p2#17): the original sequence was never confirmed from a log (The Maintainer's log not supplied); the fixes cover every mechanism found (p2 O9). Part #3 left the tombstones intact: p3#17's switch-back never targets an unbuilt product, and a page-queued rebuild of a remembered source was rejected because `/build_products` lifts the tombstone (p3 rejected (b)).
- Runtime-check part #3 on the server (p3 O1; everything in part #3 was tested only with syntax checks and stubbed harnesses): ML layer render (GDAL) and composition (canvas) (p3#5); the eraser on the composed view; single-pane flicker (p3#7); restart with a Results card selected — console `[ui] restoring: … run=<id>`, `[ui] saved selected_run=…` (p3#3, #13); a /ram-clear restart of an MRAP fire (p3#18); the thumbnail proxy's GDAL read / JPEG write (p3#4, stub only); a batch with KGC and one with "hdbscan (deprecated)" ticked, including Cancel Batch (p3#19). The Maintainer ran [persistence_thumbs] and [ml_layer_split_state]; [keyed_persistence] (p3#10–#18) not yet reported back (10-01); [batch_kgc_default] (p3#19) new.
- Results run not recalled after a restart (p3 O2): the actual cause was never reproduced; p3#10–#13 close the holes found — unconfirmed until the restart check above passes.
- Left pane's remembered product (p3 O4): if it isn't built and no full re-prepare happens (e.g. one stack purged), the page shows the loaded product with no automatic return. Option: send the remembered key and adopt it as for the right pane (p3#15).
- `getPreviewObjectURL` (templates/fire_mapping.html) uses an undefined `viewKey` in its provenance-mismatch warning — a latent ReferenceError (p3 O8) [src 10-01 still].
- `prebuild_other_source` (prepare.py) never runs: it checks the current product's stash (`_preview_stash_dir` defaults to `crop_bin`); if it ran, it would switch back by source (newest), not by key (p3 O9) [src 10-01 still].
- Pre-existing race: a pane load that finishes during press-and-hold flicker is undone on release, split and single-pane (p3 O10).
- Two tabs changing the same setting are still last-writer-wins; delta saves (p3#10) only protect settings a tab didn't touch (p3 O11).
- The "preparing" wait (p3#12) has no timeout or cancel; it logs every ~30 s (p3 O12).
- Check that dated L2 fires always persist `l2_start_date`: L2 re-prepare relies on it and falls back to `l2_reference_date` (newest acquisition) when it is empty (p3 O14).
- Split-view flicker's pixel-size correction (`beginFlicker`, templates/fire_mapping.html) effectively never runs: it reads this pane's `naturalWidth` after swapping in the other pane's already-decoded image, so both widths are the other's. Harmless — the image box keeps its size, so the swapped picture lands on the same ground anyway — but dead code; the single-pane peek (p3#7) deliberately has none. Found in part #3 [src 10-01].
- Run raster lookup order differs: `_ml_mask_raster` (handlers/serial.py, p3#5) takes a run's recorded `classified` path first, then the conventional `<fire>_serial_<id>_classified.bin`; `handle_api_serial_image` takes the conventional name first. The same file in practice; they diverge only if a run's record points elsewhere. Found in part #3 [src 10-01].

# Part B — Decisions needed (blocked on judgement, not effort)
- D1. How to visualise scaled imagery. Custom scaling (per-band 0-1, percentile trims, global/per-pixel min-max, per-pixel L2, z-score, robust z, log/dB) changes what "similar" means to the clustering, but scaled data isn't a picture: after z-score or per-pixel L2 there is no natural black or white point, so ANY rendering needs a second, display-only stretch; if that is invisible, the operator sees an appearance produced by a transform they didn't choose. Options, none implemented [src 10-01]: A (M) a "Scaled" view-dropdown entry — false colour, fixed display stretch (e.g. 2-98% per band), pane labelled `Scaled: robust_z (display 2-98%)`; honest (stretch named, separate from the analysis scaling); risk: visually comparing two scalings compares their INTERACTION with the stretch, which can invert the apparent result. B (M) a before/after histogram panel, no image — per-band raw vs scaled histograms with min/median/max; answers the real question, "what did this do to my data?", and can't mislead like A. C both — B default, A for spot checks, stretch always named on screen. Recommendation: B first; cheaper, and it shows whether a scaling helps before a clustering run is spent finding out.
- D2. Scaling also in `cpp/kgc2/kgc.cpp`? It is Python at `reduced_stack()`, covering KGC, the t-SNE/RF/HDBSCAN pipeline and Download imagery in one place. A C++ version in `kgc.cpp` was requested; there alone it would leave the other two paths unscaled. Options: keep the single Python version; duplicate in C++ (two places to keep in step); move to C++ and have the other paths call the binary. Note 09-29: the rule "never modify `kgc.cpp`/`kgc.h`" (F3) excludes `kgc.cpp` itself; a C++ route needs a separate file. ⚑dep
- D3. S2 tile labels: fire-number labels are now white on opaque black; tile labels (e.g. `10UCD`) are still magenta on 55%-transparent black [src 10-01: black at alpha 0.55, fire_mapping.html and new_fire.js]. Match them, or keep them visually distinct?
- D4. BCWS cuts per source (p2 O3): each source's cut is frozen at its first cut and BCWS is refreshed only at startup/admin, so per-source cuts are implicitly multi-temporal. Choose one current cut per fire, labelled per-source cuts, or a hybrid. The fallback uses the loaded source's cut when the stack isn't on the ramdisk; the BCWS hint raster is also a per-source snapshot.
- D5. Rebuild the right pane's remembered source after a /ram clear? (p3 O3) It isn't rebuilt automatically: the right pane shows the left's source until it is built some other way (p3#15). Option: a server-side rebuild that respects deletions (a page-queued rebuild was rejected: `/build_products` lifts the deletion tombstone, p3 rejected (b)).
- D6. Which run "ML classification" shows when no Results card is selected (p3 O7): the layer shows the newest run (what the eraser edits, p3#5), while the header ML area (`runMlArea`) prefers the accepted run, so the two can disagree.
- D7. Retire the deprecated pipeline from the GUI and code? [src 10-01] The fire page still offers it: an `hdbscan` checkbox (`use-hdbscan`) whose tooltip names t-SNE + Random Forest + HDBSCAN, and a T-SNE parameter section (templates/fire_mapping.html); the fire list's batch settings carry t-SNE defaults and an "HDBSCAN fan-out" label, and its batch bar an "hdbscan (deprecated)" checkbox (templates/fire_list.html); 21 source files reference it (progress stages, workers, handlers, `mapping_cmd.py`, …). Users see those names in the GUI, though the rule keeps them out of user documentation. Options: remove it from the GUI only (KGC the sole method); remove the code too; or keep it as a hidden fallback. Batch mapping no longer depends on it: since p3#19 the fire list's "Map Selected (with settings)" uses KGC by default and keeps this pipeline (`_serial_map_worker` over `recommended_settings.yaml`) as its "hdbscan (deprecated)" option. C2 #11 depends on the answer. ⚑dep

# Part C — Parking lot (not committed; moves to A when someone decides to do it)
## C1. New ideas
- 0.1 MRAP-type products straight from S2 ZIPs. Today a dated MRAP composite is a clip of a province-wide `<date>_mrap.bin` the back end already built → tied to the nightly mosaic: a date with no mosaic can't be produced, and the mosaic carries the whole province when an AOI needs a few tiles. From the L2A ZIPs any date with acquisitions is reachable and only intersecting tiles are read. Machinery exists: `l2_recent.py` selects, reads and composites from ZIPs; `cloud_cover.py` lists and reads them over `/vsizip//vsicurl/` without downloading. Open: MRAP's cloud-free selection is per pixel across a window — reproduce that rule, don't assume it; the result must be distinguishable from a clipped provincial mosaic (not pixel-identical).
- 0.2 Estimates from measured history (`stage_timings.yaml`). `progress.py` keeps per-stage running medians for KGC in `<shared_root>/stage_timings.yaml` and estimates from them; nothing else does — preparation, cloud-cover retrieval and multi-date builds use hard-coded constants or within-run figures [src 10-01: still KGC only]. Extending that one mechanism grounds every estimate in this server's measured behaviour. Settle first: key by AOI size (700x500 vs 2000x2600 aren't comparable; one median across both is poor); how much history (stable, yet tracking hardware/data-volume change). Reporters (what; timings persisted?): KGC run (`progress.py`, `workers.py`) stage, x/y, elapsed, ETA — yes · preparation (`prepare.py`) 5 stages, weighted fraction, ETA — no · cloud cover (`cloud_cover.py`) days done/total, ETA — no · multi-date builds (`fire_mapping.html`) x of y, elapsed, ETA — no · AOI stack build (`aoi_stack.py`) fraction — no · VIIRS download (`viirs_worker.py`) fraction, elapsed — no · acquisition plans (`acq_plans.py`) fetch age, coverage — no · overview generation (`overview.py`) progress — no · download archive (`delivery.py`) building/ready, size — no timing.
- 0.3 Erasing in the hint view. The eraser edits a result (`prepareEraseCanvases()`, `eraseAt()` on the `result`/`result_prebrush` views; edits written back to the classified raster). The hint mask isn't editable, so a hint wrong in one corner (e.g. a cloud edge caught by red-wins) can only be fixed by changing hint mode, restricting to the BCWS perimeter, or mapping then erasing the result; erasing the hint fixes it where the operator sees it, before the clustering consumes it. Not small: (a) the hint is per (product × mode) — `_redwins/<mode>_<product>_hint.bin`, derived from a stack, several modes — so an edit attaches to one, and it must be decided whether the edit follows or is discarded on a mode/product switch (the result raster has no such question: one per run); (b) since the mask-layer change a hint view is the product's post-fire image with `hintmask_<mode>.png` over it, while the eraser's canvases come from the single displayed image → erasing a hint means editing the mask layer and re-rendering both `hintmask_<mode>.png` and the composited `hint_<mode>.png` fallback so they agree. Do deliberately, recording the persistence decision; don't fold into other work.
- 0.4 Cross-fire prefetch of the next/previous fire's first image; URL must match exactly. Offered (p2 O2).
- 0.5 Startup sweep of legacy `/ram` orphans: 138-byte `*.bin.hdr`, `*.bin.post_overlays.json` (a manual `rm` list was given for `.post`/`.tmp`/`.lock`). Optional, not done (p2 O4).
- 0.6 Memory panel row for decoded images/canvas, Used only (no real capacity API). Offered, not done (p2 O6).
- 0.7 Backfill a per-date tile list for legacy L2 sources (they lack one and use the AOI's tiles). Optional (p2 O7).
- 0.8 Housekeeping (p3 O13): `<cache>/ml_masks/` isn't cleared at sweep start (stale files are harmless and go with the fire's cache); `rerender_run_overlays` (mapping.py) still re-renders every run overlay on each left switch, though the panes no longer show them (thumbnails and delivery still do).

## C2. Improvement ideas, by tier
From the former IMPROVEMENTS.md ("Potential improvements"): prioritised by impact on the actual job — a fire manager or geospatial analyst producing a defensible fire perimeter, quickly and interactively. Sources: this codebase (then ~21k lines, 28 modules + handlers), the `wps-research/py` toolset (~287 scripts), issues seen while debugging. (Old BACKLOG.md sections: requested work; decisions awaiting input; improvement ideas; quick wins; done, newest first with reasons.) Numbers are stable IDs; done items moved to D.

T0 Correctness and trust — if wrong, everything downstream is wrong and users stop believing the tool:
1. AOI-grid invariant regression test (M): split-view misalignment recurred ~6× in one session because nothing enforced "all previews share one geotransform"; build a fire, run a sweep, assert every preview and every `serial_N` shares the crop's grid — would have caught every recurrence. [src 10-01: no tests in tree; grid now pinned at creation, p2#4.]
2. Golden-fire regression suite (L): 2–3 hand-checked fires run end-to-end nightly, agreement % within tolerance; catches silent quality regressions from parameter/dependency changes (now invisible until someone eyeballs a result).
3. → D2 2026-08-10 (class_brush flag fix).
4. Quarantine corrupt VIIRS `.nc` (S): ~735 fail `NetCDF: Unknown file format` every startup and are retried forever; check size/header once, move to `_quarantine/`, log a summary; cuts startup time and log noise. [src 10-01: not done.]
5. Provenance record per exported perimeter (M): source imagery dates + platforms, hint mode, algorithm + parameters, agreement %, operator, timestamp, app version; required for anything entering an official record; cheap, most is tracked. [src 10-01 partial: the archive carries the accepted run's `<fire>_params.yaml`, hint rasters, acquisition times (L2 exact, MRAP estimated) and a contents PDF listing products + dates; no operator or app version.]
6. Readiness check before using a new MRAP mosaic (S): the active date switches the moment `<date>_mrap.bin` appears; an in-place refresh could start a build on a half-written mosaic; check header presence, size vs header, mtime stability first. [src 10-01 partial: mosaics without a `.hdr` are skipped (`list_mrap_dates`, `find_mrap_for_date`); size and mtime unchecked.]
T1 Speed of the core loop (draw AOI → prepare → map → inspect → adjust → export; every second × every fire × every day of a season):
7. AOI stack cache keyed by bbox hash (M): a similar AOI re-extracts the same L2 zips (~13 s/tile); key by rounded bbox + source + date, reuse when a cached AOI contains the new one; biggest single win for iterative work.
8. Tile-level L2 extraction cache (M): decoded 20 m band stack per (tile, acquisition), not per AOI → adjacent fires in the same tile cost almost nothing; bounded on-disk LRU.
9 → D2 2026-08-10 (progressive preview loading). 10 → D1 (warm only the current source).
11. Reuse the t-SNE embedding across settings (M): already cached per setting; the embedding often doesn't depend on the varied parameter (e.g. `hdbscan_min_samples`) — detect that, skip re-embedding; could halve sweep time. Moot while the pipeline is deprecated: see B D7. ⚑dep
12. Cancel in-flight work when a fire is closed or re-prepared (M): abandoned L2 extractions and sweeps keep running, competing for disk/GPU with what the user is viewing. [src 10-01 partial: removing a fire clears its queued product builds; closing/re-preparing cancels nothing.]
13. Batch "prepare all" for a set of fires (M): select N fires, queue overnight; with concurrency (now 2) a morning's work is ready on arrival. [src 10-01 partial: the fire list already has "Map Selected (with settings)" — the selected fires are mapped one after another in the background (`_batch_map_worker`, workers.py; `/api/batch/map`, `/api/batch/status`, `/api/batch/cancel`, handlers/batch.py) — with KGC by default and the deprecated pipeline as an option (p3#19); no prepare-only batch, no scheduling.]
T2 Interaction and editing — where an analyst earns the result, fixing what the model got wrong, fast:
14. Line/rectangle annotation (M) = A1 #1.
15. Undo/redo for all edits (M): brush and annotation edits are one-way; an undo stack turns cautious editing into confident editing. [src 10-01: the eraser has only a whole-session Revert.]
16. Polygon-level accept/reject (M): click connected components to keep/drop; most correction is "that patch isn't part of this fire" — one click each instead of brushing.
17. Side-by-side result comparison across runs (M): one result at a time today; a 2×2 or slider comparison would make choosing between sweep outputs much faster. Two runs side by side in split view = A1 #11.
18. Keyboard shortcuts (S): view switching, flicker, accept, next fire; power users live in this app and mousing to a dropdown for every comparison is slow. [src 10-01: none of these exist; E toggles the eraser (D2 2026-08-12).]
19. Persist zoom/pan per fire (S): reopening returns to the fitted view, losing the area of interest; remember the last viewport. [src 10-01: `ui_state` holds views/panel state — since part #3 also `right_known` and `selected_run`, and it is now actually restored (p3#1, #3, #6) — not zoom/pan.]
20. Edit the AOI after creation (M): a wrong AOI means recreating the fire and losing its history; allow resize/move with re-prepare.
T3 Mapping quality — better perimeters, less manual correction:
21 → D2 2026-08-12 (KGC).
22. Multi-date compositing for cloud gaps (L): cloud is the main reason a pass yields nothing usable; take the best cloud-free pixel across recent acquisitions per pixel instead of newest-wins; big gain in smoky/cloudy periods — exactly active-fire conditions. [src 10-01: L2 recent fills each tile newest-first until 95% of its AOI footprint has data (`FILL_TARGET`, l2_recent.py): gaps are filled, cloud is not.]
23. Explicit cloud/smoke masking (L): use the L2A scene classification layer (already in the SAFE) to exclude cloud/shadow from sampling and the anomaly; cheap relative to its effect on agreement. [src 10-01: no SCL use anywhere in the tree.]
24. dNBR/BARC severity products (M): `barc.py` already exists in the repo; severity classes beside a binary perimeter answer questions the current output can't.
25. Sentinel-1 SAR fallback (XL): cloud-independent; for a fire clouded for a week, the difference between a perimeter and none.
26. Auto-suggest the best run (M): a sweep gives 12 results picked by eye; rank by agreement + area plausibility + edge smoothness, pre-select. [src 10-01 partial: Results badge the highest-agreement run "(best)"; the operator's own selection is saved and restored (p3#3); no area/edge ranking, no pre-selection.]
27. Active learning from operator corrections (XL): every brush stroke is a labelled example; accumulate per region/fuel type and fine-tune; over a season could meaningfully cut correction effort.
28. Uncertainty visualisation (M): show where the classifier was marginal (cluster score near the threshold), so attention goes there instead of scanning the whole perimeter.
T4 Situational awareness — "what should I work on next, and when will I have data?":
29. Province-wide planned-coverage overlay (M; item 5 of the acquisition-plan work, deliberately deferred): tomorrow's swaths on the new-fire map → which active fires get fresh imagery next.
30. Historical coverage calibration (M) = A1 #2.
31. Notify when new imagery lands for a watched fire (M): "tell me when this fire has a new cloud-free image" is the actual question; combine acquisition plans with product arrival. [src 10-01: notifications exist for prepare, mapping, accept and batch events (`_push_notification`); none for new imagery.]
32. Fire list sorted by actionability (S): rank by "new imagery available since last mapped", not creation date → the list becomes a work queue. [src 10-01: columns sort by fire number, year, BCWS/ML size, agreement, created (the default, newest first) and status; no imagery-based key.]
33. Season/day summary view (M): fires mapped, area burned, agreement distribution, imagery availability; for reporting and spotting a systematic problem early.
T5 Robustness and operations — a bad day for infrastructure shouldn't become a bad day for the user:
34. Fix the TLS interception properly (S, but external): an expired self-signed root intercepts HTTPS; worked around for acquisition plans, it will keep breaking other outbound HTTPS; escalate rather than accrete workarounds.
35. Remember per host that Range requests are refused (S): ESA answers 403 to `Range:`; each truncation wastes a round trip and adds log noise. [src 10-01: an acq_plans.py comment now says ESA honours Range and a refused Range ends the resume loop — premise may be stale; no per-host memory.]
36. Disk-space guards on `/ram` (S): every AOI stack lives on the ramdisk; concurrency is now 2 and a large AOI is GBs; check free space before building, fail with a clear message, not a confusing GDAL error. [src 10-01: no guard; KGC only logs ramdisk free space; the Memory panel shows usage.]
37. Structured logging with levels (M): logs are prose `sys.stderr.write` calls; levels + a request id would make debugging like this project's much faster.
38. Health endpoint (S): one JSON with raster date, plan cache age + per-satellite counts, queue depth, ramdisk free, last error per subsystem; screenshot-able, scriptable; would have short-circuited several rounds of back-and-forth. [src 10-01 partial: `GET /api/memory` and `plan_health` (acq_plans.py) exist; no single endpoint.]
39. Config file instead of CLI flags (S): the flag count keeps growing (`--acq_plans_insecure`, `--padding`, `--viirs_concurrent_jobs`, …); a YAML config with current values echoed at startup reduces launch-script drift.
40. Graceful degradation when GDAL/CUDA is missing (M): today a stack trace mid-run; detect at startup, disable the affected feature, say so plainly.
T6 Bigger bets — high ceiling, high cost; discuss before committing:
41. Multi-user awareness (L): who else has a fire open, and whether someone else accepted a result; prevents duplicated or conflicting work as the team grows.
42. Result versioning and audit trail (L): keep every accepted perimeter with its provenance and allow diffing over time; perimeters are revised repeatedly, so the history matters.
43. Server-rendered tiles for very large AOIs (L): previews cap at 2000 px, so a 3000×3200 AOI is already downsampled ~40%; real tiling allows full-resolution inspection of large fires.
44. Web-based hint drawing (M): draw or edit the hint mask directly instead of relying on VIIRS/red-wins; helps small/early fires where the automatic hint is poor. The evidence given ("`Hint Size: 0.0 ha` appears often") was that header, really BCWS `fire_size_ha` (relabelled "BCWS Size", D2 2026-08-12).
45. Time-series animation of fire growth (M): sequence acquisitions to animate progression; compelling for briefings; largely reuses the date-coverage machinery. [src 10-01: the existing GIF is a two-frame blink comparator.]
46. Auto-detect new fires from VIIRS clusters (XL): propose AOIs from hotspot clusters so operators confirm rather than draw; the largest possible cut in per-fire effort.
47. Export directly to BCWS systems (M): beyond file naming, push perimeters into the downstream system so the export step disappears.
48. Offline/degraded mode (L): a field deployment with intermittent connectivity should still map from cached imagery; much groundwork exists (ramdisk cache, local KML fallback).

# Part D — Completed (kept, not deleted: several were subtle; the reasoning matters)
## D1. September 2026 sessions (to 2026-10-01)
### Part #3 thread, 2026-09-29..10-01 (p3#n [zip]; shipped; tested with syntax checks and stubbed harnesses only — see Testing)
ZIPs: [persistence_thumbs] = `fire_mapping_persistence_thumbs.zip` (templates/fire_mapping.html, handlers/fire.py, handlers/serial.py): p3#1–#4 · [ml_layer_split_state] = `fire_mapping_ml_layer_split_state.zip` (templates/fire_mapping.html, mapping.py, handlers/serial.py, handlers/base.py, BACKLOG.md): p3#5–#9 · [keyed_persistence] = `fire_mapping_keyed_persistence.zip` (templates/fire_mapping.html, handlers/fire.py, prepare.py): p3#10–#18 · [batch_kgc_default] = `fire_mapping_batch_kgc_default.zip` (workers.py, handlers/batch.py, templates/fire_list.html, plus the Markdown files): p3#19.
Testing (all items): py_compile + `node --check` on the page script; no definitions lost (each ZIP audited); harnesses ran the real extracted functions against stubbed browser/GDAL — all passed. Nothing was run against a real browser, GDAL or server in the thread. The Maintainer ran [persistence_thumbs] and [ml_layer_split_state]; [keyed_persistence] not reported back as of 10-01; [batch_kgc_default] new. Outstanding runtime checks: A4 (p3 O1).
Saved state restored and kept:
- p3#1 [persistence_thumbs]: saved UI state is actually restored — `handle_api_prepare` (handlers/fire.py) now returns ui_state, kgc_params, scaling, band_override, exclude_* flags, restrict_hint_bcws, serial_run_ids (they had been put in /diagnose); split, both views, right source, parameter boxes and band checkboxes come back; a saved view no longer offered falls back to the default.
- p3#2 [persistence_thumbs]: save gate `_uiStateApplied` — no UI-state save until the saved state is applied, so a slow open after a restart can't overwrite it with blanks. A hole found later (never opened after a "preparing" reply) was fixed in p3#12.
- p3#3 [persistence_thumbs]: Results selection saved and restored (`selected_run`; set before the first pane load, checked against serial_run_ids; `markSelectedSerialCard`/`resetSerialCardStyles`; "Sync params on click" applies the restored run's params; stale selection dropped in `showSerialGallery`; cleared when Map Fire finishes; `viewSerialRun` saves). The Maintainer then reported the run not recalled after a restart → p3#10–#13 (cause never reproduced, A4).
- p3#10 [keyed_persistence]: UI-state saves post only changed settings (`uiStateDelta`, `markUiSaved`, `_uiSavedJson` baseline taken at restore; settings the server lacks are recorded on the first save; stand-ins shown while restoring are never written back), so another open tab can't overwrite a choice it didn't touch.
- p3#11 [keyed_persistence]: hiding or leaving a page sends only unsaved changes (`flushUiState`, pagehide); `_uiSaveTimer` is now cleared when it fires (left set, every hide re-sent the whole state).
- p3#12 [keyed_persistence]: `prepareFire` waits out "preparing" replies (asks again every 2 s) instead of continuing with an empty reply that left the page unrestored and unable to save for the whole visit.
- p3#13 [keyed_persistence]: Results selection kept when /prepare's run list is empty or unknown; cleared, and saved as cleared, only when a real list shows the run gone (`_staleSavedRun`; `restoreFireState` clears on an empty /console list).
Results thumbnails and the ML classification layer:
- p3#4 [persistence_thumbs], closes A1 #10 (was p2 O1: the ML classification pane view waited for the Results thumbnails — both used `/serial/<run>/image`, which may re-render a run's overlay per request; proposed fix: load thumbnails after both panes' images load and serve them from the existing overlay without re-rendering, only the pane view re-rendering): Results thumbnails no longer delay the panes — a `loadPaneImage` wrapper counts `_paneLoadsInFlight`; cards use data-src, `releaseSerialThumbs()` after panes settle or 8 s (`SERIAL_THUMB_CEILING_MS`); `&thumb=1` → `_serial_low_proxy()` (handlers/serial.py) serves `previews/serial_<id>.low.jpg` ≤400 px, remade when the overlay is newer, never re-renders. GDAL read / JPEG write tested with a stub only.
- p3#5 [ml_layer_split_state]: "ML classification" composed per pane in the page — the pane's own post imagery + a transparent layer from new `GET /api/fire/<f>/ml_mask?run=<id>` (`mlCompositeObjectURL`, `mlLayerObjectURL`, `mlMaskUrl`, `mlMaskKey`; 409 → pane retries, other failures → old server composite). Server: `ensure_ml_mask` + `_overlay_mask_on_post(out_dir=)` (mapping.py; `<cache>/ml_masks/<n>.png` + .json, remade on raster/grid/preview-size change), `_ml_mask_raster` + `handle_api_ml_mask` (handlers/serial.py), route (handlers/base.py). Fixes the right pane's imagery not following its source selector; left works the same way; no selection = newest run (what the eraser edits; B D6). Real GDAL render and canvas composition not runtime-tested.
- p3#8 [ml_layer_split_state]: subtitle source tag also on ML classification; `warmAllProducts` warms 'post' when the left view is ML classification.
Split view, the right pane and flicker:
- p3#6 [ml_layer_split_state]: right pane's source and view remembered while split is closed (`_rightStateKnown`, saved as `right_known`; fires saved in split count): first manual open starts from the left; reopen restores (deleted source → left's, missing view → Post-fire); `toggleSplit` close no longer resets; `onPostSourceChange` stops mirroring once known.
- p3#7 [ml_layer_split_state]: press-and-hold flicker in single-pane view shows the remembered right-pane view (`beginSinglePeek`, `prefetchPeek`, `peekSig`, `peekCached`, `resolvePaneObjectURL`, `schedulePeekPrefetch`; subtitle "Right pane: …" while held; `endFlicker` restores). Not runtime-tested.
- p3#15 [keyed_persistence]: right pane keeps a remembered source that isn't built (`_rightWantedKey`/`_rightFallbackKey`; shows the left's meanwhile; `adoptRightWantedIfBuilt` on product-watch/refresh; an explicit right choice cancels it); deleting the remembered right source while split is closed moves it to the next. Not rebuilt automatically after a /ram clear (B D5).
Products kept by key:
- p3#14 [keyed_persistence]: selectors keyed — `setSourceSelectValue` no longer maps a missing dated key to the newest product of its source (bare `l2`/`mrap` still resolve).
- p3#16 [keyed_persistence]: the page switches the server to the chosen product at open when `user_product_key` ≠ `product_key` (`informServerOfProduct`).
- p3#17 [keyed_persistence]: the server keeps the operator's product — `_chosen_product_key` overwrites `fire.user_product` only if that product was deleted (logs once otherwise); the on-demand preview switch-back never targets an unbuilt product (handlers/fire.py).
- p3#18 [keyed_persistence]: `_prepare_fire_sync` (prepare.py) rebuilds an MRAP fire on its remembered night (user_product, else the old stack name; `_build_stack`), falling back to the newest mosaic if that night is gone.
Batch mapping:
- p3#19 [batch_kgc_default]: batch mapping (fire list, "Map Selected (with settings)") uses KGC by default. `_batch_kgc_fire` (workers.py) runs each fire exactly as the fire page's Map Fire does with `hdbscan` unticked — same priming, `run_kgc`, automatic Accept of an only result, failure handling — with the fire's saved `kgc_params` else the defaults, and its loaded source; earlier Results are kept; a fire with no AOI stack or hint is prepared first (the old sweep's test); a fire found preparing or mapping is skipped. The deprecated sweep stays available: tick "hdbscan (deprecated)" beside the button (it sends `method: 'hdbscan'`; that sweep still replaces a fire's earlier Results). `/api/batch/map` takes `method` ('kgc' default, 'hdbscan'; anything else 400); Cancel Batch stops an in-flight KGC run (`kgc_cancel` + subprocess terminate) or, for hdbscan, the sweep as before. Files: workers.py, handlers/batch.py, templates/fire_list.html; the fire page and interactive mapping untouched. Tested with harnesses running the real worker, helper and handlers against stubs (default KGC, the hdbscan option, skips, failures, cancel while preparing and while running, preparation failure, method validation) and the page's `batchMap()` in both checkbox states; not runtime-tested.
Backlog:
- p3#9 [ml_layer_split_state]: the in-tree BACKLOG.md gained "### 9. A separate ML classification for each pane in split view" at the end of its section 1 (edited from the Sep 15 copy) → A1 #11 here.
Rejected / not shipped: (a) a placeholder selector option for a missing dated key — conflicts with the "select existing options only" rule in `setSourceSelectValue`; (b) the page queuing a rebuild of the right pane's remembered source — `/build_products` lifts the deletion tombstone, so it could bring back a deleted source (B D5); (c) moving "ML Classification — before brushing" to the per-pane layer — deferred (A1 #11, p3 O6).
### Part #2 thread, 2026-09-25..29 (p2#n [zip]; shipped + tested unless noted)
ZIPs: p2#3–#12 all shipped in [grid_race_rows_status]; p2#13–#16 in [date_select_delete_blank_preview] (#15 refined in [blank_until_ready]).
Sources panel during build — was A1 item 0, top priority as of 2026-09-25 (defects 0a/0b blocked features 0c/0d):
- 0a, p2#1 [sources_unbuilt_rows]: `/sources` returned an empty list ("No products yet") while the server logged `[products] K71082: mrap_p20260923 (not built yet), l2 (not built yet)`; rows were dropped server-side in `handlers/fire.py` between that log and the JSON (`renderSourcesList` in templates/fire_mapping.html renders whatever it gets, even synthesises pending rows — not the cause); the dated layers appeared only after leaving and re-entering the fire. Now: `built:false` rows for unbuilt products during a build, 409 on deleting them, no premature lookups. p2#11: a `build_status` NameError (`_product_states_payload`) had left Date-select rows without states.
- 0b, p2#3 [grid_race_rows_status]: preview directory-lifecycle race — `prepare` deleted `previews_<prod>/` as a stale stash while up to three threads rendered into it → `os.replace` FileNotFoundError (`post failed`, `pre failed`, `diff1 failed`, `geo record failed`); the same contention showed as "Busy: the background prebuild of the other source is using this fire". Unique temp names per process/thread (preview.py, mapping.py; ship297, 2026-09-25) were shipped first and made no difference. Fix, new `preview_fs.py`: render into `.render_tmp`; per-fire lock around rename/delete; live-directory owner token (a late render of another product is discarded); `geo.json` recorded in every rendered dir, merged under a lock.
- 0c, p2#6: bottom pop-up progress bar removed for layers (kept only for Map Fire).
- 0d, p2#5: row status follows the fire-list model (stage i/n, detail, ETA %, "no change for Nm"), so every per-fire state (e.g. "preparing / Build AOI stack (3/3) / AOI stack ready ~60s left (25%)") has a per-product equivalent, strictly per dated product (one product's state never describes another); building TTL 30 min; the fire's progress shows on idle rows while preparing. p2#12: the stage mapper had matched "read" inside "ready".
- p2#7: Sources polled only while rows are active or the fire is preparing; unbuilt rows not clickable.
- p2#24 [sources_first_split_persist]: page entry loads the Sources list first (≤2.5 s), then imagery; "Preparing…" overlay text removed; split view persists (tri-state `split_at_load`; auto-split only if never unsplit; keepalive save on `pagehide`).
Previews and caches:
- p2#2 [fire_cache_purge_on_remove_and_create]: a recreated fire inherited the old one's previews → `.web_cache/<fire>` purged on remove AND create.
- p2#15 [date_select_delete_blank_preview; refined blank_until_ready]: a pane switched to a not-ready source shows blank (grey) unless the image is already in the tab's memory.
- p2#16: "Preview render failed": warming skipped stashes holding any PNG (hint masks); now requires `post.png`, retries 3×, shows the real reason.
- p2#22 [faster_stepping_and_caching]: server switch after a 600 ms settle (Map Fire flushes it, ≤15 s); neighbour prefetch ±1/±2 plus own views, 3 in parallel, `nobuild`; dead warmers removed; previews use ETag + no-cache (304 revalidation).
Grid and concurrency guards:
- p2#4: authoritative AOI grid pinned at fire creation from the pre-imagery (`median.bin`) rectangle; the L2 composite uses the pin; the pin is kept if it is the bbox's hull; L2 buffer origin check; publish-time grid guard.
- p2#10: display grid guard — live previews served only if their geo matches the stack.
- p2#8: `date_plot`/rebuild return 409 while preparing or unbuilt, and rebuild the requested dated product (no stray undated L2).
- p2#9: a busy `post_source` returns 409; the client retries and the newest step wins.
Date select, deletion, independence from originals:
- p2#13 [date_select_delete_blank_preview]: ready MRAP/L2 rows are never tickable (data-ready; excluded from apply).
- p2#14: Sources delete: rows vanish at once; a pane showing the deleted source moves to the next row down (else the nearest above); the server deletes in the background and switches off the loaded product with busy-retry; deleting the only product is refused.
- p2#17 [delete_tombstones_memory_panel]: deleted products no longer resurrect — tombstones (in memory + manifest `deleted`) hide them from listings and placeholders, re-point a remembered product, 404 a late `post_source`, and are skipped by on-demand switch-back and default refresh; Date select lifts them. Original sequence unconfirmed (A4).
- p2#21 [delete_by_product_key]: deleting by product key removes every file (any leading date, ramdisk + store, orphan `.bin.post.*`, tmp, KGC scratch), never a neighbour's; manifest entries + notes cleared; withheld/retired rows labelled and deletable; retiring removes the store twin + overlays; a failed L2 build removes its buffer; listed rows ignore other copies' withheld/retired notes.
- p2#23 [sources_independent_of_originals]: Sources no longer need MRAP mosaics/zips that were deleted — an existing stack is reused by product key (ramdisk/store) before any source lookup; no duplicate dated-L2 rebuilds on a newer mosaic; Hint view "queued forever" loop fixed.
- p2#18 [memory_panel_refinements, memory_sort_used_label] Memory panel, under Sources. Server rows (`memory_monitor.py`, `GET /api/memory`; singleton, one sampler thread per row, 1 s): RAM, swap, ramdisk, SSD, data (+ store/output/tmp if separate), GPU n, GDAL cache, cgroup (if its limit < RAM), sorted by % used. Browser rows: JS heap, image cache, browser storage, shared across tabs by BroadcastChannel. 11 px grid; Size/Used/Free MB + % Used bar; image cache LRU budget min(1 GB, deviceMemory/4).
Hint mask:
- p2#19 [hint_mask_label_and_display]: GUI "Hint perimeter" → "Hint mask" (stored names unchanged); status-aware loader (409 → wait, 404 → empty reason); impossible masks recorded (`.none`, 30 min/60 s); all modes pre-generated in the warm queue (3 workers); stashes judged by `post.png`; empty rasters not drawn.
- p2#27 [flicker_with_hint_mask; hint_mask_follows_linked_pane]: flicker shows the other pane's imagery AND hint mask; the hint mask follows linked pan/zoom (`syncOtherPane`; not runtime-tested, A4).
Download, BCWS, cloud, views:
- p2#20 [download_accepted_plus_selected]: archive always has the accepted run's source imagery + all hint rasters, plus ticked sources; stem from the accepted product (`params.yaml` `source_product`; MRAP capped at the product date); durable-store products included; L2 `_dates.json`; the PDF lists products + dates; background builder (newest selection wins, per-archive lock, keeps 2 recent); size estimate on the button.
- p2#25 [bcws_label_timestamps]: BCWS labels show retrieval time after the fire number (new cuts exact; old cuts ≤ cut time).
- p2#26 [l2_cloud_from_acquisitions]: L2 cloud cover from recorded (tile, date) acquisitions (`acquisitions` in `_dates.json`), pixel-weighted; pending via `cloud_cover.pending_days` (no re-fetch of "no product"); never the name date. A "nearest-day" patch was rejected and reverted.
- p2#28 [right_view_selector_results]: the right view selector no longer inherits the left's disabled state; "Serial Results" → "Results"; thumbnails lazy, low priority, async.
### Found in the source, not previously recorded (original thread, ≈2026-09-25)
- "Open when ready" → "Open immediately" (fire_list.html): a new fire/AOI opens at any stage except error.
- Per-product state notes, keyed by fire number only, survived deletion and gave the next same-name fire spurious "withheld: not on this AOI" rows; `forget_fire_product_states()` (prepare.py) now clears them on fire deletion (handlers/fire_list.py) [ship296].
- Batch mapping from the fire list (recorded 2026-10-01): select fires → "Map Selected (with settings)", "Cancel Batch", "Delete Selected"; the fires are mapped one after another in the background, ending with a "Batch mapping complete" notification. As found, with the deprecated pipeline only: `_batch_map_worker` (workers.py) ran `_serial_map_worker`, the t-SNE + Random Forest + HDBSCAN sweep over the four settings in `recommended_settings.yaml`, K HDBSCAN replicates each. Since p3#19: KGC by default, that sweep when "hdbscan (deprecated)" is ticked. ⚑dep E1 only says the sibling package's batch flow is unchanged, and that package's README isn't in this tree [src 10-01].
### Products, dates and the source selectors
- Dated products: MRAP and L2 composites identified by source AND date (`mrap_p20260909`, `l2_d20260805`); preview stashes, cache keys, preview URLs, hint masks, flicker and GIF captions key on it → nightly builds no longer overwrite each other.
- One L2 identity: "newest-first from whatever exists" and "start from a chosen date" are the same product — both `l2_d<date>`, same label; a one-time migration renamed artefacts under the old split keys.
- MRAP Date select lists the province-wide mosaics in `/data/mrap_bc`, newest first, and clips the chosen one to the AOI; `ensure_aoi_stack` takes `mrap_date`, so earlier days can be built, not always the newest.
- Multi-select in both Date select modes: sequential builds, an ETA learnt from this fire's own builds, dates already built skipped.
- Step buttons (forward/back in time) beside each selector, shown/hidden with the right-pane controls.
- Selectors sorted by date, newest first, across both sources.
- Live refresh: a polled `/products` endpoint; new products appear without reloading.
- Warm only the current source (was C2 #10 / A2; filed as "12 preview combinations ≈ 80 MB of background traffic per fire open, competing with the image being waited on; warm 6, fetch the other source on demand"): warming the other source had made the server switch the fire there, render and switch back — slow, and why panes briefly showed the wrong product.
### Cloud cover
- `cloud_cover.py`: importable adaptation of `sentinel2_extract_cloud_cover_tiles.py` (keeps level, workers, single-thread, cache, UTM-zone options); lists via the bucket's REST API, so no `s3fs`.
- Per-tile-day cache on the SSD, incremental; days with no products recorded, not re-queried.
- Averaged over the tiles that HAVE data, count shown; a partial average is marked.
- Dialog: retrieval status + ETA; red/green bars with distinct "still retrieving" and "nothing on record" states.
### Access control
- No login for ordinary use; the admin area alone needs username AND password, re-verified every 15 min.
- Open-by-default IP tracking with a single denial state (block).
- "Known addresses": browser, OS, screen size, request count, first and most recent use; persisted on the SSD.
### Persistence and recovery
- Output directory keyed by YEAR, not raster filename, with automatic migration; raster-named dirs made fires vanish whenever a new mosaic arrived.
- Recorded absolute paths rebased onto the current output root.
- Saves can no longer erase a fire's identity (bbox, date range).
- `durable.py`: stacks mirrored `/ram` → `<output_root>/.stacks` in the background, restored on demand before any rebuild; fire identity recovered from sidecar geotransforms when the record is lost.
- GUI state (split view, per-pane products, parameters) persisted, flushed on page exit. Saved, but apart from split on/off not restored until p3#1: `/prepare` never sent the saved state back (hardened in p3#10–#13).
- Caches (was A3 #2: "bring `.download_cache` and `_preview_cache` under `cache_retention.py`, which sweeps only `.web_cache`; a fire downloaded once keeps its archive for ever, and a deleted fire's archive is never cleaned up"): resolved by other means [src 09-29] — `.download_cache` keeps the current archive + the 2 newest others per fire (`delivery.prune_cache`, p2#20) and is cleared on fire removal; `_preview_cache` reaps entries older than 30 min on each call (`_PREVIEW_TTL_S`). Neither is under `cache_retention.py`.
### Correctness fixes worth remembering
- Preview dirs carry a `.product` marker: the stack pointer and the pixels are different things, and trusting the pointer showed one product's imagery under another's name.
- Classification and its hint resolved by product → a run uses the imagery on screen. The hint isn't a score: it tells the clustering which pixels are burned, so a stale one steers the result.
- Overlay caches validate against the crop's actual grid, not mtimes.
- Rebrush falls back to the accepted run's raster when there is no canonical classification.
- Stack headers carry `default bands` → downloaded imagery opens on the post-fire bands, not the pre-fire half (identical across every product).
- The grid check no longer refuses a good build because a KGC scratch file matched its glob.

## D2. Earlier sessions
### 2026-08-12 — KGC method, scaling, band control, manual editing
- KGC clustering as a second ML method (was Requested work item 1, then C2 #21). `cpp/kgc2` built on demand (rebuilt when the binary is older than the source), run on the ramdisk in a per-fire working dir; band 1 of its six-band product (the binary selected class) → `<fire>_classified.bin`; then the existing pipeline unchanged: brushing (BRUSH parameters), agreement, ML area, `serial_1.png` thumbnail, results gallery + Accept, polygonisation, export. An `hdbscan` checkbox (unchecked = KGC) swaps the parameter sections and Map Fire buttons; BRUSH shows in both, since brushing runs on whichever mask was produced. Background, concurrent across fires; cancel terminates the subprocess. Original request ("Add KGC algorithm to fire mapping ML methods"): a selectable classifier beside the t-SNE + Random Forest + HDBSCAN pipeline; slot into the CLI `--method`-style selection so the serial sweep can compare it on the same AOI stack + hint; same result contract (classified `.bin` on the AOI grid, agreement %, ML area) so gallery, agreement scoring and overlays work unchanged; benchmark against the existing method on fires with known-good perimeters before offering it as default. ⚑dep
- Custom scaling (`scaling.py`), ten methods, formulae shown in the picker: per-band 0-1; percentile trim per band and intensity-based (P%, no-clip, right-only/left-only); global min-max; per-pixel min-max; per-pixel L2; z-score; robust z-score; log/dB. Applied AFTER band selection, only to the classifier input and Download imagery; displayed panes deliberately untouched.
- Custom bands picker: an explicit band list overrides the checkbox rules; checkbox changes apply INCREMENTALLY (each click adds/removes only the bands that box governs), so hand-picked choices survive unrelated toggles.
- "Diff only" mode keeps only the anomaly bands (implies pre-fire excluded, contradicts exclude-diff); resolved in one place (`band_select`) so every consumer agrees.
- Manual eraser: square cursor sized in image pixels; live per-pixel preview from the post-fire imagery; Revert to a pre-session snapshot; "Outside BCWS only" (a stroke straddling the official boundary trims only the outside); E toggles it, bound on window in the capture phase so focus can't swallow the key.
- "Restrict hint to BCWS perimeter" clips whichever hint is selected — preview, agreement score and clustering input alike.
- "Clip to BCWS perimeter" after brushing, in all three result paths (KGC, sweep, rebrush).
- Per-fire persistence of every new setting + an opaque `ui_state` → a fire reopens with the same layers, sources, split, parameters, results; `kgc_params` records what actually ran and wins over the GUI copy on restore. [Part #3 found the restore never happened: `/prepare` didn't return `ui_state` or `kgc_params` until p3#1 (2026-09-29); only split on/off partly survived.]
- Robustness: KGC clears stale products before launch and rejects a product older than the run; the written class mask is read back and checked (size, geotransform, projection); geotransforms compared, not just dimensions; `ensure_geo()` repairs lost map info on open and at startup.
- Interlaced GIF: a blink comparator built from the frames actually displayed (overlays included), so any view/source combination works.
- Fixed: `find_classified()` was passed a fire NUMBER instead of the fire, so perimeter vectorization never found the raster; rebrush edited a different file from the one Accept promotes; `handlers/fire.py` scaling edits had silently failed to apply; the header's "Hint Size" was actually BCWS-reported `fire_size_ha`, now labelled "BCWS Size".
### 2026-08-10
- Hint option "BCWS perimeter": rasterises every BCWS fire polygon intersecting the AOI into a hint mask — deliberately not filtered to one fire number, since this system detects burn rather than attributing it; same storage, per-source naming, mtime invalidation and CLI contract as the red-wins hints; "Red wins (post)" stays the default.
- Auto-open a newly created fire when ready: "Open when ready" checkbox beside Logout (default on, remembered); cancelled if the user navigates away from the fire list, so a fire finishing later can't yank them out of what they're doing; also seeded from the page transition, because `new_fire.js` is a cached static file and the sessionStorage handoff alone was unreliable. (Superseded by "Open immediately", D1.)
- Concurrent AOI preparation: `viirs_concurrent_jobs` 1 → 2; work was already queued and backgrounded, but a single dispatch thread made concurrent creation serial in practice.
- `class_brush` flag mismatch fixed (was C2 #3 and an A2 quick win: "`class_brush.exe: brush_size must be > 0` fails on every run, so brush post-processing never happens and results are rougher at the edges; a pure argument mismatch, noticed in logs but never chased"). Against a binary older than `--no-intermediates`, the C++ parser stops at the unknown flag and treats it as the filename, so `brush_size` was parsed from an absolute path as 0. The CLI now probes the binary's usage text and passes only supported flags. Rebuild `cpp/class_brush.exe` to regain `--no-intermediates`. [src 09-29: the probe is in `fire_mapping_cli.py`, outside this tree; brush.py's rebrush path passes positional arguments.]
- Progressive preview loading (was C2 #9: "a ~400 px preview immediately, then the full one; first paint well under a second on a slow link; the JPEG change helped, this removes the wait almost entirely"): a ~400 px JPEG proxy via `?lowres=1`, painted immediately and replaced by the full image, same geographic framing and vectors, so the swap shows only a sharpness change; ~5.06 s → ~35 ms to first paint at the measured link rate [src 09-29: handlers/fire.py].
- JPEG previews + gzip JSON: `pre`/`post`/`diff*` get a JPEG twin (~11× smaller); masks stay PNG; JSON gzipped when the client supports it; both report format and savings to the browser console.
- Faster startup: the daily province-overview regeneration moved off the startup path; only missing overviews block; minutes → seconds on regeneration days.
- Plan completeness diagnostics: per-satellite datatakes/day, window coverage, download size, explicit complete/truncated verdict.
### 2026-08-09 — S2 acquisition plans
- New `acq_plans.py`: fetch, parse, cache ESA's S2A/S2B/S2C plans on the ramdisk; refresh at startup and daily, every 15 min while incomplete.
- "Expected next coverage" panel: every planned pass over the AOI within the horizon, soonest painted on top, with each pass's AOI share and how much is new ground.
- Resilience: parallel downloads; multi-transport fetch (system CA → certifi → `$SSL_CERT_FILE` → curl → unverified as a last resort, because the network intercepts HTTPS with an expired certificate); short-read detection; truncated-KML salvage; per-satellite merge so one failure can't drop a satellite; local-KML offline fallback.
- Per-record validation: the original all-or-nothing check discarded two complete plans over one legal antimeridian coordinate each (longitudes just outside ±180); bad records are now dropped and counted.
### 2026-08-08 — Geometry and display correctness
- AOI padding removed entirely (pinned to 0 in prepare, settings, CLI, persistence; UI control removed): padding was the only thing that moved the AOI window after creation, and every move put previews on a different grid — the root cause of repeated split-view misalignment. All views now share one grid by construction.
- Run overlays re-rendered onto the current AOI grid after any prepare, source switch or sweep, with a self-healing size check when a preview is served.
- Georeferencing shipped with the image (`X-Geo-*` headers), so a pane can never pair with another raster's extent.
- Split-view sync: identical grids copy the transform verbatim; differing grids reconcile through the native CRS.
- Press-and-hold flicker: instant (direct `src` swap) and drift-free (exact transform restore).
- Per-pane view selectors in split view replace the ambiguous single dropdown.
- Empty ML results no longer create a view (an empty mask produced a PNG identical to post-fire, mislabelled "ML classification").
### 2026-08-07 — Results pipeline
- Serial results never appearing, fixed: the mapping CLI writes outputs beside its input image, which had moved to the ramdisk, while the post-run code still looked in the fire cache; classified masks, comparison figures and `serial_N.png` are now found where they land. This was the "all runs failed" symptom despite F1 85–89%.
- Classified-mask naming derived from the stack, not the old `<fire>_crop.bin` convention; legacy names as fallbacks.
- Exhaustive clustering diagnostics on every run: mask/hint pixel counts, geotransforms, intersection/union/IoU, and a named reason for every non-computable agreement.
- Hint view registered whenever a hint mask exists, not only when the generic overlay rendered.
### 2026-08-06 — Performance and data plumbing
- Per-AOI on-demand stacks replace the province-wide stack; L2-recent compositing with parallel per-tile extraction and per-acquisition date attribution.
- "L2 coverage by acquisition" plot, later with satellite prefixes (`S2A+S2B · 2026-08-04`), backfilled for existing fires from SAFE filenames.
- Post-source switching (L2 recent ↔ MRAP composite) with per-source preview stashes and background prebuild of the other source.
- Client-side preview caching and warming, optimistic first paint, `Cache-Control` on previews.
- Timing instrumentation throughout (TTFB vs transfer vs decode vs bake), which made the later diagnosis possible.
- Atomic preview writes (tmp + rename), fixing images served mid-rewrite as correct-width but truncated-height.
- Background prebuild no longer mutates the visible source (new fires had opened on MRAP instead of L2).
- Honest status labels: `cropping` → "Build AOI stack", `Crop:` → `AOI:`, detail strings describe what actually happens.

## D3. Notes
- The C++ tools live in `wps-research/cpp` and are NOT rebuilt by this app (exception: `cpp/kgc2`, built on demand, D2 2026-08-12). `class_brush.exe` must be recompiled after source changes, or the flag probe keeps it in degraded mode.
- Diagnostics added during debugging are permanent; they made several of the above findable at all.
- BACKLOG.md and IMPROVEMENTS.md are superseded by this file (tiers and quick-wins reproduced unchanged); delete the old files so two lists can't drift [src 09-29: both still in the tree; the part #3 thread edits that BACKLOG.md]. Done 2026-10-01: this file replaced the in-tree BACKLOG.md (part #3's one edit merged as A1 #11) and IMPROVEMENTS.md was emptied (0 bytes), as were README.md, PLAN.md, FIRE_MAPPING_ALGORITHM.md and BRUSHING.md (content in E1, E2, E4 and E5).
- Several Done items were verified only by logic and unit-level tests (no GDAL or C++ toolchain where written): scaling, hint restriction, geo repair and the KGC build path were first exercised on the server.

# Part E — Reference
How the application and its algorithms work, compressed from the tree's README.md (E1), PLAN.md (E2), FIRE_MAPPING_ALGORITHM.md (E4) and BRUSHING.md (E5). Written for v1: later entries in Part D supersede parts of it (e.g. per-AOI stacks replaced crops 2026-08-06; padding removed 2026-08-08; `cropping` relabelled "Build AOI stack"; KGC added 2026-08-12; `viirs_concurrent_jobs` default now 2; VIIRS download disabled by default, A1 #6). [src 10-01: the tree has no `tests/` directory.] Since 2026-10-01 Part E is the only copy: the tree's README.md, PLAN.md, FIRE_MAPPING_ALGORITHM.md and BRUSHING.md are empty (0 bytes). E4 is ⚑dep: a verbatim copy describing the deprecated t-SNE + Random Forest + HDBSCAN pipeline, which is still available (its note; B D7). E5 is ⚑dep only because it names that pipeline; brushing itself is current. E3, the current KGC pipeline at a glance, was written from the code on 2026-10-01 and comes before the deprecated E4 on purpose; E2, E4 and E5 carry reader's notes added the same day, marked as such.

## E1. Overview (was README.md)
This version is for operational use: it seeds fire mapping results from NRT (accumulated) VIIRS data. The other version, for historical research, seeds them from historical fire perimeters: [batch_fire_mapping_web](https://github.com/bcgov/wps-research/tree/master/data/bill/batch_fire_mapping_web).

batch_fire_mapping_viirs_web is a sibling of `batch_fire_mapping_web`: same downstream ML pipeline, but the front of the pipeline is a user-defined bbox + VIIRS download workflow instead of a polygon shapefile. Analysts draw a bounding box on the year's overview, name a fire, pick a date range and submit; the server downloads VIIRS active-fire data for that bbox and window, accumulates and rasterizes the hot pixels, derives a tight crop from the actual fire pixels, and seeds the existing GPU mapping pipeline. Once the prepare worker reaches `READY`, every downstream feature (map / serial sweep / rebrush / accept / batch / multi-year) works exactly as in the polygon-driven sibling.

### What changed vs. `batch_fire_mapping_web` (`_web`, polygon → `_viirs_web`, this package)
- Fire source: pre-curated polygon shapefile → user draws a bbox in `/new_fire`.
- Fire identity: `FIRE_NUMBE` from the polygon attribute → user-supplied name (validated).
- VIIRS download: all years, at startup, blocking → per year, at startup, blocking (idempotent); bootstrap also builds a per-year `year_index.gpkg`, so fire creation runs a single bbox-pushdown read instead of walking the per-granule shp tree.
- Crop bounds: polygon geometry intersection + padding → bbox of nonzero VIIRS pixels + padding.
- `--polygon_file` (required positional) → removed. `--perimeter_mode` (`viirs`/`traditional`) → removed (always VIIRS).
- LAADS token: optional (`--skip_download`) → required at `/data/.tokens/laads`.
- New page `/new_fire` (canvas overlay + form); new modules `overview.py`, `viirs_worker.py`; multi-year in both.
The downstream mapping / brush / accept / serial / rebrush / batch flows are unchanged and documented in the source package's [README](../batch_fire_mapping_web/README.md); this README documents only the new front of the pipeline.

### Quick start
From `data/bill/`: `./run_fire_viirs_web.sh`, which invokes:
```bash
python3 -m batch_fire_mapping_viirs_web \
    --rasters /ram/new_cloudfree/pgfc_2023.bin /ram/new_cloudfree/2024_pgfc.bin /ram/new_cloudfree/2025_pgfc.bin \
    --out_root ./fire_mapping_results_viirs --laads_token_file /data/.tokens/laads \
    --user_password <user_password> --admin_password <admin_password>
```
Open `http://localhost:8765` and click **+ New Fire**. The first launch generates one overview PNG per raster (memory-bounded GDAL stride read; a 100 GB raster takes ~30 s); later launches use the cached PNGs (cache key = raster mtime + size).

### Requirements (in addition to everything `batch_fire_mapping_web` needs)
- LAADS DAAC token file at `/data/.tokens/laads` (one line); get one at <https://ladsweb.modaps.eosdis.nasa.gov/profile/#app-keys>; override the path with `--laads_token_file`.
- Network egress from the server to `ladsweb.modaps.eosdis.nasa.gov` (the per-fire download fetches VNP14IMG `.nc` granules).
- `viirs.utils.shapify`, `viirs.utils.accumulate` and `viirs.utils.rasterize` importable from `data/bill/viirs/` (already in this repo); the `netCDF4` Python module (used by `shapify` to read VNP14IMG).
- GDAL / NumPy / GeoPandas / PyYAML / matplotlib / scipy come from the sibling package's existing requirements.

### The `/new_fire` flow
Two panes. Left: the year's overview PNG under a transparent canvas; click-drag draws a yellow rectangle; clicking inside an existing rectangle drag-moves it; **Clear bbox** resets; hovering shows live cursor coords (raster CRS + WGS84) in the toolbar. Right, the form:
- Name: must match `^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$`, no `..` substring, no `/` or `\`; uniqueness checked case-insensitively against existing fires.
- Bounding Box (raster CRS / WGS84): read-only readouts that update as you drag.
- Start / End dates: placeholders `<year>-01-01` and `<year>-12-31` (the overview JSON's `default_start` / `default_end`); empty fields fall through to the placeholders at submit time. Constraints: parseable as `YYYY-MM-DD`; start ≤ end; start ≥ 2012-01-19 (VNP14IMG availability); end ≤ today (server time).
**Confirm & Create** POSTs `/api/fire/create`. On 202 the page redirects to `/`, where the new fire shows as `preparing` with live sub-stage progress. Validation errors come back as `{errors: [{field, message}, ...]}` and render in the right panel without losing form state. The year selector is visible only on multi-year deployments and disabled for non-admins (year switch is admin-only).

### The VIIRS prepare worker
Download + shapify + index build happen once per year at server boot, in `year_viirs.bootstrap_all_years` (idempotent: existing `.nc`, `.shp` and `year_index.gpkg` are reused). Each year's full raster footprint is downloaded for the whole-year window `<year>-01-01` to `<year>-12-31` (or to today if that is in the future) into `<output_root_for_year>/_year_viirs/VNP14IMG/<year>/<jday>/`. After shapify, `year_viirs.build_year_index` consolidates every per-granule `*.shp` under that tree into one GeoPackage, `<output_root_for_year>/_year_viirs/year_index.gpkg` (layer `viirs`, GPKG R-tree on geometry, a text `det_dt` column in compact `YYYYMMDDHHMM`, so date filters reduce to lexicographic comparisons regardless of GDAL/SQLite type coercion). A sidecar `year_index.gpkg.manifest` records the source `.shp` count; the index is rebuilt only when that count or any source `.shp`'s mtime advances; atomic write via `.tmp.gpkg → rename`. Per-fire prepare queries this single file with bbox pushdown instead of opening hundreds of per-granule shapefiles.

Fires submitted via `/api/fire/create` go to a module-level FIFO queue (`viirs_worker._dispatch_queue`) served by `--viirs_concurrent_jobs` parallel workers (default 2). Each fire walks two stages — no download and no shapify (done at boot):
- `accumulating` (cancellable between stages). Fast path (default): `viirs_worker._fast_accumulate_from_index` runs one bbox-pushdown read against `year_index.gpkg`, date-filters in pandas, and writes the per-fire `VIIRS_VNP14IMG_<startdt>_<enddt>.shp` the rasterize step expects. Slow fallback when the index is missing (older deployments, build failure): `viirs.utils.accumulate(...)` walks the per-granule shapefile tree with a bbox filter at read time. The two paths are interchangeable as far as the rest of the pipeline is concerned.
- `cropping` (the last stage; not cancellable). `_tight_bounds_from_shapefile` reads `gdf.total_bounds` from the cumulative shapefile (no full-extent rasterize) and expands by `_RASTERIZE_BUFFER_M` and `padding * max_dim`; `crop_raster` produces `<NAME>_crop.bin`; VIIRS is rasterized onto the cropped extent so the hint aligns to the crop's grid; previews are generated, including a green-tinted hint overlay on `previews/post.png`.
On success the fire flips to `READY` and `is_new=True` (drives the "new" badge); `fire.crop_bin`, `fire.viirs_bin`, `fire.hint_bin`, `fire.acc_start`, `fire.acc_end` are populated and a success toast goes out; from there the pipeline (single-shot map, serial sweep, accept, …) is unchanged from the sibling.
Failure modes set `fire.status = ERROR` with an actionable `fire.error_msg`: `LAADS DAAC rejected the token` (auth failure mid-download); `No VIIRS fire pixels in bbox during the chosen date range.`; `shapify exited with code N`; `GDAL crop failed.`. Network failures inside one day are logged, and fail the worker only when the entire shapify run produces nothing.
Cancel mid-prepare: `POST /api/fire/<NAME>/cancel_create` sets `fire.cancel_event`, SIGTERMs any live subprocess, waits up to ~10 s for the worker to tear down, then drops the FireInfo from `state.fires` and `rmtree`s the cache_dir. Destructive on purpose: the operator chose to abandon the fire.
Server crash mid-prepare: on next boot `init_fires_from_disk` finds the orphan `.web_cache/<NAME>/` with no live worker and flips the fire to `ERROR` with message `interrupted; retry create`; the operator deletes via `/api/fire/<NAME>/remove` (or the admin dashboard) and re-creates. No automatic resume (future work, E2 §13).

### CLI reference
`python -m batch_fire_mapping_viirs_web --rasters R1 [R2 ...] --out_root DIR [options]`; run with `--help` for the canonical list. Argument (default): purpose.
- `--rasters` (required): Sentinel-2 ENVI `.bin` rasters; each filename must contain a unique 4-digit year (1970..now+1).
- `--out_root` (required): mother dir; per-year results land in `<out_root>/<raster_stem>_mapping_results/`.
- `--year` (newest): initial active year; falls back to `<out_root>/active_year.yaml`, then to the newest year.
- `--padding` (`0.0`): ignored — padding was removed (D2 2026-08-08) and the AOI is exactly the bbox drawn; kept so existing launch scripts don't break.
- `--sample_rate` (`0.05`): default sampling rate for the ML pipeline. `--min_samples` (`500`) / `--max_samples` (`30000`): lower/upper bound on per-fire sample size.
- `--viirs_concurrent_jobs` (`2`): prepare jobs run in parallel; the rest queue FIFO. `--viirs_download_workers` (`16`): per-job parallel LAADS download workers. `--viirs_shapify_workers` (`8`): per-job parallel shapify workers.
- `--host`, `--port` (`0.0.0.0:8765`): server bind address.
- `--admin_password`, `--user_password` (required): set both, or use `--insecure_no_auth` (off by default), which disables all auth + IP gating, for isolated environments only — never on a multi-user host. `--admin_username` (`admin`): the username required alongside `--admin_password`.
- `--trust_proxy` (off): honour `X-Forwarded-For` for the client IP (only behind a trusted reverse proxy).
- `--laads_token_file` (`/data/.tokens/laads`): path to the LAADS DAAC token file (one line).
- VIIRS download [src 10-01: these flags were missing here; defaults from `__main__.py`]: `--enable_viirs_download` (off): download NEW VIIRS granules again — disabled by default (`year_viirs.VIIRS_DOWNLOAD_ENABLED`, A1 #6); `--province_wide_viirs_download` (off): the old whole-footprint download at startup, for a one-off backfill (granules are otherwise fetched per AOI); `--skip_viirs_bootstrap` (off): skip the year-wide VIIRS download step at startup; `--viirs_min_interval_minutes` (`60`): attempt that download at most once per N minutes, across restarts (`0` = every start); `--force_viirs_bootstrap` (off): ignore that throttle; `--viirs_download_method` (`curl_primary`, or `urllib_primary`): which transport is tried first; `--parallel_viirs_downloading` (off): download days concurrently across `--viirs_download_workers` threads (default one day at a time, so the log reads in order).
- Overviews: `--overview_force_regeneration` (off): regenerate the per-year overviews even if already done today (default: at most once per BC calendar day); `--disable_overview_force_regeneration` (off): skip the forced regeneration at startup.
- `--acq_plans_cafile` (empty): CA bundle for the acquisition-plan download (e.g. a proxy's root CA).

### HTTP API — what's new
The sibling package's API plus these endpoints, all user+admin:
- GET `/new_fire`: bbox-drawing page + form (HTML).
- GET `/api/year/<y>/overview.png`: per-year overview PNG (cached on disk).
- GET `/api/year/<y>/overview_meta`: sidecar JSON — `geotransform`, `crs_wkt`, `raster_W/H`, `overview_W/H`, `extent_native`, `extent_wgs84`, `default_start`, `default_end`, `cache_key`.
- POST `/api/fire/create`: body `{name, year, bbox_native:[xmin,ymin,xmax,ymax], start, end}`; validates, creates `FireInfo`, enqueues the prepare worker; 202 with `{name, status:'preparing'}`; 400 with `{errors: [{field, message}, ...]}` on validation failure; 409 if the name collides under a race.
- POST `/api/fire/preview_hint`: body `{year, bbox_native, start, end}`; accumulates VIIRS for the bbox + dates from the year-wide shared shp dir, rasterises onto the user's bbox, generates pre-classification preview PNGs, returns `{preview_id, area_ha, views: {hint, post}}`.
- GET `/api/fire/preview_hint/<preview_id>/<view>.png`: the PNG referenced by `preview_hint`; reaped after ~30 min.
- POST `/api/fire/<name>/cancel_create`: cancels an in-flight prepare (SIGTERMs subprocesses, rmtrees the cache, drops the FireInfo); 409 if the fire is already past `PREPARING` (e.g. `READY` — use `/remove` instead).
- POST `/api/fire/<name>/clear_new`: sets `is_new=False`; called fire-and-forget when the user clicks **Open** in the fire list.
`/api/fires` also exposes per fire `is_new`, `error_msg`, `sub_stage`, `sub_stage_idx`, `sub_stage_total`, `sub_stage_detail`, rendered in the fire list as a "new" badge, an inline error line, and a sub-stage progress line under the status pill.

### Output structure
```
<out_root>/  active_year.yaml, sessions.yaml, access_control.yaml, notes.yaml, notifications.yaml,
             stage_timings.yaml, cache_retention.yaml, cancel_audit.yaml
  .web_cache/_overviews/  pgfc_2023.png (per-year overview PNG; cache_key = mtime + size),
                          pgfc_2023.json (sidecar metadata, consumed by /new_fire), ...
  pgfc_2023_mapping_results/  fire_state.yaml, accepted_params.csv
    _year_viirs/  (year-wide VIIRS data, built once at boot)
      VNP14IMG/<year>/<jday>/*.nc + *.shp   (per-granule raw + shapified)
      year_index.gpkg                       (consolidated R-tree-indexed GPKG)
      year_index.gpkg.manifest              (shp-count freshness sidecar)
    .web_cache/<NAME>/  (per-fire prepare cache)
      VIIRS_VNP14IMG_<...>.shp/.dbf/.shx/.prj   (cumulative, from fast path or accumulate)
      _viirs_crop/VIIRS_VNP14IMG_<...>.bin      (crop-aligned rasterize)
      <NAME>_crop.bin                           (ENVI cropped raster)
      <NAME>_serial_<N>_classified.bin          (gallery entries)
      previews/{pre,post,hint,result}.png
    <NAME>/  (promoted on accept; canonical)
      <NAME>_crop.bin_classified.bin, <NAME>.shp / .kml, <NAME>_comparison.png,
      <NAME>_brush_comparison.png, <NAME>_params.yaml (has a new `bbox:` section + `accumulation:` dates)
  2024_pgfc_mapping_results/  ...
```
`<NAME>_params.yaml` carries an extra section so an accepted fire can round-trip back through `init_fires_from_disk` on next boot:
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

### File overview — what's new or changed
Files only in this package:
- `overview.py`: per-year overview PNG + sidecar JSON generator. `generate_overview(raster, png_path, json_path, max_dim=2000)` does a memory-bounded GDAL `ReadAsArray(buf_xsize=, buf_ysize=)`, so a 100 GB raster reads at ~50 MB peak; atomic write (tmp + fsync + rename + parent-dir fsync); reuses `preview.detect_band_groups` to prefer the post group; `overview_is_fresh` and `ensure_overview` implement the cache by `(st_mtime_ns, st_size)`. The sidecar JSON is consumed client-side in `new_fire.js` for pixel ↔ CRS ↔ WGS84 math.
- `year_viirs.py`: year-wide VIIRS bootstrap (download + shapify + index), run once at server boot. `bootstrap_all_years(state)` iterates every (year, raster) and calls `bootstrap_year` → `download_year` (per-day LAADS pulls into `_year_viirs/VNP14IMG/<year>/<jday>/`, parallel via `--viirs_download_workers`) → `shapify_year` (parallel via `--viirs_shapify_workers`; skips already-shapified granules) → `build_year_index` (the single GeoPackage described above). Idempotent via a `.manifest` recording shp count + a per-source-shp mtime check; atomic write (`.tmp.gpkg → rename`); drops the `FID` column geopandas pulls off shapefiles so it doesn't collide with the GPKG primary key.
- `viirs_worker.py`: the 2-stage prepare worker (`accumulating`, `cropping`). `submit_fire(fire)` enqueues; a daemon dispatcher pulls FIFO and calls `_viirs_worker(fire)`, which walks `accumulate_for_fire → _tight_bounds_from_shapefile → crop_raster → rasterize_shapefile (onto crop) → previews`. `accumulate_for_fire` first checks for a matching seeded shapefile (`_seeded_shp_matches_fire`), then tries the fast path (`_fast_accumulate_from_index`: bbox pushdown into `year_index.gpkg`, date filter in pandas, writes a per-fire cumulative `VIIRS_VNP14IMG_<startdt>_<enddt>.shp` matching the slow path's column contract), and falls back to `viirs.utils.accumulate(...)` (per-granule walk with bbox-filtered reads) when the index is missing or unreadable. `cancel_fire(fire)` sets `fire.cancel_event` and SIGTERMs any live subprocess. `_tight_bounds_from_shapefile` reads `gdf.total_bounds` directly (no full-extent rasterize); `_tight_bounds_from_viirs_bin` is retained for tests / re-prepare. Cancellation, subprocess group kills and progress snapshots all go through this module.
- `templates/new_fire.html`: the bbox-drawing page — a static `<img>` overview under an HTML5 `<canvas>` overlay; embeds a small JSON config block for `new_fire.js`.
- `static/new_fire.js`: canvas drag handler (create / move), pixel ↔ raster CRS ↔ WGS84 conversion from the overview JSON, live cursor readout, form validation, POST to `/api/fire/create`; no external libraries.
- Tests: `tests/conftest.py` — synthetic raster + VIIRS bin fixtures (UTM 10N, 30 m/pixel). `tests/test_overview.py`, `test_overview_caching.py` — 12 tests, overview generation + cache-key freshness. `tests/test_bbox_validation.py` — 9, bbox geometry, non-finite and clipping rules. `tests/test_date_defaults.py`, `test_date_validation.py` — 14, placeholder fall-through, ISO parsing, VNP14IMG lower bound, future-date rejection. `tests/test_fire_name_validation.py` — 16, path traversal, case-insensitive uniqueness, leading-punctuation rules. `tests/test_tight_crop.py` — 5, bbox-of-nonzero-pixels + padding math. `tests/test_viirs_worker_cancel.py` — 4, cooperative cancel during download, subprocess kill mid-shapify, cache cleanup, idempotent no-op. `tests/test_fire_create_endpoint.py` — 7, the validation-and-enqueue path of `/api/fire/create`. `tests/test_year_index_fast_path.py` — 6, `build_year_index` (creation, idempotence, rebuild on new shp) and `_fast_accumulate_from_index` (bbox+date filter, empty-result raise, missing-index fallback).

Files changed from the sibling:
- `__main__.py`: drops the `polygon_file` positional, `--perimeter_mode`, `--skip_download`, `--shapify_workers`; adds `--laads_token_file`, `--viirs_concurrent_jobs`, `--viirs_download_workers`, `--viirs_shapify_workers`. Before startup it loads the LAADS token and generates per-year overviews (via `overview.ensure_overview`), then boots the server; calls `app_state.init_fires_from_disk()` instead of `init_fires_from_gdf()`.
- `state.py`: `FireInfo` adds `bbox_native`, `bbox_wgs84`, `viirs_start_date`, `viirs_end_date`, `is_new`, `cancel_event`. `AppState` drops `gdf`, `viirs_gdf`, `polygon_file`, `polygon_gdf_raw`, `viirs_shp_dir`, `viirs_shp_dirs_by_year`, and adds `overview_png_by_year`, `overview_meta_by_year`, `laads_token`, `viirs_jobs`, `viirs_subprocs`, `viirs_concurrent_jobs`, `viirs_download_workers`, `viirs_shapify_workers`. New `init_fires_from_disk()` rebuilds the registry from `<output_root>/<NAME>/<NAME>_params.yaml` (accepted) and `<output_root>/.web_cache/<NAME>/` (in flight; orphaned mid-prepare entries flip to ERROR).
- `app.py`: adds `from . import viirs_worker as _viirs_worker` and a `_viirs_worker.init(app_state, _save_fire_state, _push_notification)` call at the end of `init_app`; every other sibling module's wiring is unchanged.
- `prepare.py`: `_prepare_fire_sync` is now the re-prepare path: when the operator changes padding or wipes the cache, it locates the cached full-extent VIIRS bin, re-derives tight bounds via `viirs_worker._tight_bounds_from_viirs_bin`, re-crops the reference raster, and re-rasterizes the cumulative VIIRS shapefile onto the new crop frame. The polygon-perimeter rasterize and polygon-VIIRS-intersection blocks are deleted; initial prepare lives entirely in `viirs_worker._viirs_worker`; `_accept_fire_sync` is unchanged except for the dropped polygon refs.
- `persistence.py`: `_save_fire_state` adds `bbox_native`, `bbox_wgs84`, `viirs_start_date`, `viirs_end_date`, `is_new`, `error_msg`, `fire_year`, `fire_size_ha`, `fire_date` to the persisted YAML; `_load_fire_state` synthesizes a new `FireInfo` for fires that exist only in `fire_state.yaml` (e.g. hidden + cache wiped); `_switch_year` just swaps the active-year handles and reloads from disk — no polygon re-projection or spatial-filter pass.
- `validation.py`: adds `_validate_fire_name` (regex + traversal + case-insensitive uniqueness), `_validate_date` (strict ISO YYYY-MM-DD), `_validate_date_range` (empty-string default fall-through, `start ≤ end`, `start ≥ 2012-01-19`, `end ≤ today`), `_validate_bbox` (4 finite floats, x/y ordering, raster-extent overlap, clip-to-extent return); the existing `_validate_param` / `_validate_embed_bands` are unchanged.
- `handlers/base.py`: registers six new routes (3 GET + 3 POST) for the bbox-drawing flow.
- `handlers/fire_list.py`: adds `handle_new_fire_page`, `handle_api_year_overview_png`, `handle_api_year_overview_meta`, `handle_api_fire_create`, `handle_api_fire_cancel_create`, `handle_api_fire_clear_new`; extends `handle_api_fires` with `is_new` / `error_msg` / `sub_stage*` keys; drops the `state.polygon_file` reference in the home-page render.
- `templates/fire_list.html`: "+ New Fire" button in the header (links to `/new_fire`); "new" badge in the fire-number cell; sub-stage display in the status cell when status is `preparing` (e.g. *downloading_viirs (1/5) — 3 / 5 days*); inline **Cancel** beside **Open** for in-flight fires; two new helpers, `cancelCreate` and `markFireOpened` (clears `is_new` server-side).
- `static/style.css`: new `.newfire-*`, `.nf-*` and `.status-new` selectors for the new page + badge.
Unchanged from the sibling: every other module — `auth`, `notifications`, `cache_retention`, `progress`, `mapping`, `brush`, `kml`, `templates`, `mapping_cmd`, `workers`, `preview`, `io_utils`, `recommended_settings.yaml`, and the rest of `handlers/`, `templates/` and `static/`.

### Persistence and crash recovery
Same atomic-write pattern as the sibling (`io_utils._atomic_yaml_dump` → tmp + fsync + rename + parent-dir fsync), with two extras. Overview cache: PNG + sidecar JSON are written atomically, so a partial PNG cannot survive a crash; if the JSON is corrupt, `overview_is_fresh` returns False and the next launch regenerates. Orphan in-flight fires: see "Server crash mid-prepare" above — flipped to `ERROR` so the operator can decide whether to delete and recreate (no automatic resume; noted as future work in `PLAN.md` §13, E2).

### Tests
```bash
cd ~/GitHub/wps-research/data/bill
python3 -m pytest batch_fire_mapping_viirs_web/tests/ \
    --ignore=batch_fire_mapping_viirs_web/tests/audit -v
```
Baseline: 111 pass across `test_overview*`, `test_*validation*`, `test_tight_crop`, `test_tight_bounds_from_shapefile`, `test_viirs_worker_cancel`, `test_viirs_worker_progress`, `test_cancel_create_nonblocking`, `test_detective`, `test_fire_create_endpoint`, `test_year_index_fast_path`. The audit-suite tests under `tests/audit/` are the legacy `bash run_all.sh` PASS/FAIL framework imported wholesale from the sibling; they reference the polygon package and are skipped here. A real-LAADS end-to-end test (small AOI + 2-day window against a working token) is left to manual QA — checklist in `PLAN.md` §14 (E2).

### Cross-references
For everything downstream of `READY` — fire-list filtering, single-shot mapping, serial mapping (N×K sweep), rebrush, accept, batch, cache retention, multi-year switching, queue / toasts, KML export, persistence + crash recovery, the full HTTP API, and the file overview for unchanged modules — see the sibling package's [`README.md`](../batch_fire_mapping_web/README.md). The architecture, mixin-globals binding pattern, GPU-lock model and `_wire_handlers` helper-dict design are all identical. The implementation plan that drove this package is [`PLAN.md`](./PLAN.md) (§§0–14), now E2.

## E2. Original build plan, April 2026 — historical (was PLAN.md; verbatim from the tree except three added reader's notes, headings demoted one level)
> Reader's note (added 2026-10-01): the original build plan, written as instructions to whoever built this package. It starts from a full copy of `batch_fire_mapping_web` — the older sibling package that takes fires from a polygon shapefile — and lists the edits that turned that copy into this package. Headings such as Remove / Keep / Add mean: taken out of the copy, kept from it unchanged, newly added. "Current" means the sibling's code as copied; §n refers to the numbered sections below; "the implementing agent" or "next agent" is the builder. It is historical and superseded in places: padding was removed, KGC was added (E3), per-AOI stacks replaced crops, `--viirs_concurrent_jobs` now defaults to 2, a throttled year-wide VIIRS step runs at startup (its downloading of new granules is off by default, A1 #6), and there is no `tests/` directory. For how things work now, use E1, E3, Part D and the code.

*Author: planning conversation, 2026-04-27. Implementing agent: read top-to-bottom, then start at §11 file-map.*

This package is a sibling to `data/bill/batch_fire_mapping_web/`. It replaces the **polygon-driven fire list** with a **user-defined fire** workflow: an analyst draws a bounding box on the year's reference raster, names a fire, supplies a date range, and the server downloads VIIRS data for that bbox/range, accumulates+rasterizes it, derives a tight crop from the actual fire pixels, and seeds the standard ML mapping pipeline.

The downstream mapping/brush/accept/serial flow is **unchanged** from `batch_fire_mapping_web`. Only the *front of the pipeline* changes.

---

### 0. Origin and code-share strategy

- Initial source: full file-by-file copy of `batch_fire_mapping_web/`. Diff against that copy.
- Why duplicate instead of share: the `_web` package is stdlib-only with mixin-globals binding (`handlers/*.py:init`); making it parameterizable on "polygon vs. user-fire" mode would force it through both code paths and pollute the existing audit-clean module. A copy is cheaper and lets the existing package keep evolving.
- Implementing agent should `cp -r` first, then make the edits in §3-§9. Run the test suite (§10) at the end.

---

### 1. Top-level package layout

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

### 2. Conceptual model — what changes

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

### 3. CLI changes — `__main__.py`
> (Reader's note, 2026-10-01: edits to the copied `__main__.py` — Remove = sibling options deleted, Keep = carried over unchanged, Add = new here; "current Steps 1-3" = the sibling's startup steps. Today's options: E1, CLI reference.)

#### Remove
- positional arg `polygon_file`
- `--perimeter_mode` (always viirs)
- `--skip_download` (no startup download)
- `--shapify_workers` (workers are per-fire now; expose as `--viirs_download_workers` and `--viirs_shapify_workers` defaulting to 16 / 8)
- imports: `load_and_filter_polygons`, `load_all_viirs`, `download_viirs`, `shapify_viirs`
- the `_prepare_year_for_viirs` helper

#### Keep
- `--rasters <r1> <r2> ...` (one per year)
- `--out_root`
- `--year` (initial active year)
- `--host`, `--port`, `--admin_password`, `--user_password`, `--insecure_no_auth`, `--trust_proxy`
- year-from-filename detection (`_year_from_filename`)
- per-year output dir derivation `<out_root>/<raster_stem>_mapping_results`

#### Add
- LAADS token load at startup. If `/data/.tokens/laads` missing or unreadable: `sys.exit('ERROR: ...')` with actionable message.
- Per-year overview generation (sync, blocking — one-time cost). Print progress per raster.
- After overview generation, store paths in `app_state.overview_png_by_year` and `app_state.overview_meta_by_year`.
- `app_state.laads_token` field.

#### Startup sequence (replaces current Steps 1-3)
1. Validate `/data/.tokens/laads` exists, load into `app_state.laads_token`.
2. For each `(year, raster)` in `rasters_by_year`:
   - Compute overview cache path: `<shared_root>/.web_cache/_overviews/<raster_stem>.png` and `<...>.json`.
   - Cache key: `(raster_path, st_mtime_ns, st_size)` stored in JSON.
   - If JSON exists and key matches: skip regeneration.
   - Else: call `overview.generate_overview(raster, png_path, json_path, max_dim=2000)` (blocking).
   - Store paths in app_state maps.
3. Init `AppState` with no polygon, no `gdf`, no `viirs_gdf`. Call `state.init_fires_from_disk()` to rebuild fire registry from `<output_root>/` and `<.web_cache>/`.

---

### 4. AppState changes — `state.py`
> (Reader's note, 2026-10-01: changes to the copied sibling's `state.py` — Remove fields / Add fields = fields taken out of / added to its `AppState`; the `FireInfo` subsections below likewise.)

#### Remove fields
- `self.gdf`
- `self.viirs_gdf`
- `self.polygon_file`
- `self.polygon_gdf_raw`
- `self.viirs_shp_dir`
- `self.viirs_shp_dirs_by_year`

#### Add fields
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

#### `FireInfo` additions / changes (`state.py`)
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

#### `FireInfo` field reused or repurposed
- `fire_numbe` → reused as the user-supplied name (validated regex match — same as today's polygon-derived names: `^[A-Za-z0-9][A-Za-z0-9_. -]*$`, no `..`, no `/\\`, length ≤ 64).
- `fire_year` → set to the active year at create time (used only for filtering/display).
- `fire_size_ha` → computed from VIIRS hint area (after rasterize). Until then: `0`.
- `fire_date` → set to `viirs_end_date` for display compatibility.
- `perimeter_type` → always `'viirs'`.

#### Replace `init_fires_from_gdf()` with `init_fires_from_disk()`
- Scan `<output_root>/<NAME>/<NAME>_params.yaml` → load each accepted fire's name + saved bbox/dates.
- Scan `<output_root>/.web_cache/<NAME>/` for in-flight or stale fires; reconstruct `FireInfo` skeleton with `status=PREPARING` (worker re-attaches if still running) or `ERROR` (if no live worker and stage file shows incomplete).
- Persist new fields in `fire_state.yaml` via `persistence._save_fire_state` (already extensible — just add the new keys).

#### Status enum
Keep `FireStatus` exactly as today. The "downloading_viirs / shapifying / accumulating / rasterizing / cropping" sub-stages live in `fire.progress.stage` (already exposed by `progress.py:_progress_snapshot` and polled by the UI). Decision rationale: avoid pushing the UI status filter to seven new values; the existing `PREPARING` status with rich progress payload already gives the operator everything.

---

### 5. Overview module — `overview.py` (NEW)

#### Purpose
Generate ONE downsampled PNG per year-raster, cached on disk, served as a static file. The bbox-drawing UI uses this PNG as its background; pixel→map→lat-lon math runs client-side off the sidecar JSON.

#### API
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

#### Implementation notes
- Band detection: reuse `preview.py:detect_band_groups` against header band names; prefer the `post` group (or `B12/B11/B9` fallback).
- Reproject bbox-corner polygon (raster CRS → EPSG:4326) using the same `_bbox_to_4326` code as `viirs/fp_gui/download_dialog.py:63`.
- Atomic write: write to `*.tmp`, fsync, rename. (Reuse `io_utils._atomic_yaml_dump` pattern.)
- Default dates: derive year from `_year_from_filename(raster_path)`; emit `f'{year}-03-01'` / `f'{year}-10-30'`.

---

### 6. New page — `templates/new_fire.html` + `static/new_fire.js`

#### Layout (single-column on narrow screens, two-column on wide)
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

#### Client-side behaviour (`new_fire.js`)
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

#### Year selector
- Visible only when `len(rasters_by_year) > 1`.
- Admin-only: switching year here calls `/api/year/switch` and reloads `/new_fire` against the new active year.
- For non-admins on multi-year: the dropdown is disabled and shows "(admin only)".

---

### 7. New endpoints

All in `handlers/fire_list.py` (route registration in `handlers/base.py`'s router).

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/new_fire` | user+admin | Render `new_fire.html`. |
| GET | `/api/year/<y>/overview.png` | user+admin | Stream cached PNG. 404 if year invalid. |
| GET | `/api/year/<y>/overview_meta` | user+admin | Stream cached JSON. |
| POST | `/api/fire/create` | user+admin | Body: `{name, year, bbox_native:[x_min,y_min,x_max,y_max], start, end}`. Validates, creates `FireInfo`, enqueues VIIRS prepare worker. Returns 202 + `{name, status:'preparing'}`. |
| POST | `/api/fire/<name>/cancel_create` | creator+admin | Cancel an in-flight prepare. SIGTERM the worker subprocess group, rmtree cache, drop FireInfo. Returns 200 + `{status:'cancelled'}`. |
| POST | `/api/fire/<name>/clear_new` | user+admin | Flips `is_new=False`. Called by `/fire/<name>` page on first paint. |

#### `/api/fire/create` server-side validation (in `validation.py`)
1. `name` — regex `^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$`, reject `..`, reject if already in `state.fires` (case-insensitive on existing names).
2. `year` — must be in `state.rasters_by_year`.
3. `bbox_native` — 4 finite floats, x_min < x_max, y_min < y_max, intersects the year's raster extent.
4. `start`, `end` — parseable `YYYY-MM-DD`. Empty → use the year's defaults from overview JSON. Then: start ≤ end, start ≥ `datetime.date(2012, 1, 19)`, end ≤ today. Reject otherwise with explicit message.
5. Return all validation errors in one response: `{errors: [{field, message}, ...]}` so the form can highlight every problem.

---

### 8. VIIRS prepare worker — `viirs_worker.py` (NEW)

#### Threading model
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

#### Cancellation
- `fire.cancel_event` is checked between stages (already shown).
- The download stage uses `viirs/fp_gui/download_dialog.py:_download_worker` pattern: a `ThreadPoolExecutor(max_workers=16)` whose tasks check `cancel_event.is_set()` at the top. On cancel, `executor.shutdown(wait=False, cancel_futures=True)`.
- Shapify spawns subprocess. Track Popen in `state.viirs_jobs[fire_name]`. Cancel handler calls `os.killpg(proc.pid, SIGTERM)` (mirrors `app.py:_terminate_serial_proc`).

#### Failure modes — explicit
- Token rejected by LAADS → `WorkerError('LAADS DAAC rejected the token; check /data/.tokens/laads.')`.
- Network failure mid-download → log per-day; only fail the worker if 0 .nc files made it.
- 0 fire pixels in AOI → `WorkerError('No VIIRS fire pixels in bbox during the chosen date range.')`. Status flips to ERROR; user can either delete the fire or re-create with a different bbox/date range.
- VIIRS bin all zeros after rasterize → same error message.
- Disk full → propagate OSError.
- Reference raster missing CRS → already raised by `shapify.get_crs_from_raster`.

#### Persistence
- Worker state is reconstructable: `cache_dir` content tells you where you got. On server restart mid-prepare, `init_fires_from_disk` flips the in-flight fire to ERROR (no worker to re-attach to) — user re-creates or restarts the prepare via a new endpoint `POST /api/fire/<name>/retry_create`. (Out of scope for v1 — note as future work.)

---

### 9. Tight-crop derivation — in `viirs_worker.py`

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

### 10. Tests — `tests/`

All tests use `pytest`. Synthetic fixtures live in `conftest.py`.

#### `conftest.py` fixtures
- `tmp_raster_3x3km`: a 100×100-pixel ENVI .bin/.hdr in EPSG:32610 (UTM 10N) covering a known area in BC. Three bands, each filled with deterministic gradients. `mtime`, `size` exposed.
- `tmp_viirs_bin`: same shape as crop, with 5 nonzero pixels at known coords (used for tight-crop testing).
- `mock_laads_token`: monkeypatch a fake `/data/.tokens/laads` to a tmp path.
- `mock_laads_sync`: monkeypatch `viirs.utils.laads_data_download_v2.sync` to write a known fixture .nc to the destination dir without hitting the network.

#### `test_overview.py`
- `test_overview_dimensions_clip_to_max_dim` — overview PNG longest edge ≤ 2000 px.
- `test_overview_sidecar_json_round_trip` — load JSON, verify pixel→map round-trips through GeoTransform and inverse exactly for the corners.
- `test_overview_pre_post_band_detection` — synthetic raster with `pre_B12, pre_B11, pre_B9, post_B12, post_B11, post_B9` band names → overview uses post group.
- `test_overview_default_dates_match_year` — for `pgfc_2023.bin`, sidecar contains `default_start='2023-03-01'`, `default_end='2023-10-30'`.

#### `test_overview_caching.py`
- `test_overview_skipped_when_fresh` — second call with same raster mtime/size does not regenerate (mock the GDAL read and assert call count = 0).
- `test_overview_regenerated_when_raster_changes` — touch raster's mtime, second call regenerates.

#### `test_bbox_validation.py`
- `test_bbox_outside_raster_extent_rejected`
- `test_bbox_zero_area_rejected`
- `test_bbox_partially_outside_clipped` — bbox extending past raster bounds is clipped to extent and accepted.
- `test_bbox_non_finite_rejected` — NaN / inf in any coord rejected.

#### `test_date_defaults.py`
- `test_default_start_is_march_1_of_raster_year` — for `pgfc_2023.bin`, default `start='2023-03-01'`.
- `test_default_end_is_october_30_of_raster_year` — for `pgfc_2023.bin`, default `end='2023-10-30'`.
- `test_defaults_match_active_year_when_multi_raster` — overview JSON for each year contains its own year's defaults.
- `test_empty_dates_in_create_request_use_defaults` — POST with `start=''` and `end=''` resolves to the year's defaults.

#### `test_date_validation.py`
- `test_unparseable_date_rejected` — `'2023-13-01'`, `'not-a-date'`.
- `test_end_before_start_rejected`.
- `test_start_before_2012_01_19_rejected` — VNP14IMG availability lower bound.
- `test_end_in_future_rejected` — end > today (using `freezegun` or monkeypatching `datetime.date.today`).
- `test_start_equals_end_accepted` — single-day range valid.
- `test_iso_format_only` — reject `'2023/03/01'`, accept `'2023-03-01'`.

#### `test_fire_name_validation.py`
- `test_valid_names` — `'C12345'`, `'My Fire 2023'`, `'fire.A'`.
- `test_path_traversal_rejected` — `'../foo'`, `'..foo'`, `'foo/bar'`, `'foo\\bar'`.
- `test_empty_name_rejected`.
- `test_too_long_name_rejected` — > 64 chars.
- `test_duplicate_name_rejected_case_insensitive` — `'fire1'` and `'FIRE1'` collide.
- `test_leading_punctuation_rejected` — `'-foo'`, `'.foo'` (regex starts with `[A-Za-z0-9]`).

#### `test_tight_crop.py`
- `test_tight_crop_one_pixel_fire` — VIIRS bin with 1 pixel at known coords; crop bounds match expected (1×1 + padding).
- `test_tight_crop_padding_clamped_to_raster_extent` — fire near edge → crop clipped to raster.
- `test_tight_crop_zero_pixels_raises` — VIIRS bin all zero raises `WorkerError`.
- `test_tight_crop_padding_zero_returns_exact_bounds` — `padding=0` → exact pixel bbox.

#### `test_fire_create_endpoint.py`
- `test_create_returns_202_and_appears_in_list` — happy path, mocking the LAADS sync.
- `test_create_with_invalid_name_returns_400` — error response body has structured `errors[]`.
- `test_create_with_year_not_in_registry_returns_400`.
- `test_create_dispatches_worker_to_queue` — assert worker thread started with the expected fire.
- `test_concurrent_create_with_same_name_second_returns_409`.
- `test_admin_and_user_can_both_create` — both roles allowed.

#### `test_viirs_worker_cancel.py`
- `test_cancel_during_download_aborts_and_cleans_cache` — start a worker with mocked slow download, hit `/cancel_create`, assert: `state.fires` no longer contains the fire, `cache_dir` is removed, no zombie threads.
- `test_cancel_during_shapify_kills_subprocess` — mock shapify subprocess that sleeps; cancel sends SIGTERM; subprocess terminates.
- `test_cancel_after_ready_no_op` — once status is READY, `/cancel_create` returns 409 (use `/api/fire/<name>/remove` to delete an accepted fire instead).

#### Integration test (deferred to manual QA)
- Real LAADS DAAC end-to-end against a small AOI + 2-day window. Requires real token; not in CI.

---

### 11. Implementation order (concrete steps for the next agent)

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

### 12. Robustness considerations (the "extremely robust" bar)

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

### 13. Out of scope for v1 (note for future)

- Pan/zoom on overview (use Leaflet/MapLibre + tile server). v1 is single static PNG with rectangle drawer.
- Per-user LAADS tokens.
- Renaming a created fire.
- Bulk-create from a CSV of bboxes.
- Sharing downloaded `.nc` files between fires with overlapping bbox/date (cache-key by `(bbox, date)` and dedupe).
- "Retry create" endpoint to resume a failed prepare from the last completed stage.
- Animated VIIRS playback on the fire page (port from `viirs/fp_gui/fire_animation_controller.py`).
- Server-side validation of the LAADS token at startup (currently we trust the file).

---

### 14. Acceptance criteria

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

## E3. KGC mapping — pipeline at a glance (the current method; written from the code, 2026-10-01)
What runs when Map Fire is pressed with `hdbscan` unticked (the default): `handle_api_kgc_map` (handlers/serial.py; `POST /api/fire/<fire>/kgc_map`, cancel `/kgc_cancel`) starts `run_kgc()` (kgc.py) in a background thread. Since p3#19 a batch from the fire list runs the same pipeline for each selected fire (last bullet below). Bracketed names are the progress stages the page shows. KGC takes the same two inputs as the deprecated pipeline (E4) and everything after the clustering is shared with it, so a KGC result is brushed, scored, overlaid, listed in Results, accepted and exported exactly like one from E4.

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                       │
│ AOI stack of the source being mapped (the right-most        │
│ pane's; pre + post S2 bands, MRAP or L2 recent)             │
│ + the selected hint mask (optionally restricted to          │
│ the BCWS perimeter)                                         │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │ STAGE 1 — PREPARE (shared; before Map Fire)              │
    │ prepare.py, aoi_stack.py, l2_recent.py                   │
    │ per-AOI stack on the pinned AOI grid; hint mask;         │
    │ preview PNGs                                             │
    └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │ STAGE 2 — BAND STACK  [kgc_stack]                        │
    │ mapping_cmd.reduced_stack()                              │
    │ keep the selected bands (band_select), then apply        │
    │ the fire's scaling (scaling.py); hint rebuilt for        │
    │ this source if needed; float32 copies (ensure_float32)   │
    └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │ STAGE 3 — KGC CLUSTERING  [kgc_build, kgc_cluster]       │
    │ cpp/kgc2 (outside this tree), built on demand;           │
    │ GPU or CPU build chosen by memory (choose_kgc_build)     │
    │ density-ordered hill climbing on a k-nearest-            │
    │ neighbour graph of a strided pixel sample                │
    │ (~budget_points, default 20 000), K up to kmax           │
    │ (10 000) in steps of kstep (5); the hint selects the     │
    │ burned class at the best K -> six-band product           │
    └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │ STAGE 4 — CLASS MASK  [kgc_classify]                     │
    │ kgc._extract_class_band()                                │
    │ band 1 (the selected class) ->                           │
    │ <fire>_serial_<n>_classified.bin on the AOI grid;        │
    │ "before brushing" overlay rendered (result_prebrush)     │
    └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │ STAGE 5 — BRUSH  [kgc_brush]                             │
    │ kgc._brush_classified() -> brush._run_class_brush_only   │
    │ (the class_brush step of E5); unbrushed mask kept        │
    │ (*_raw.bin); then optional "Clip to BCWS perimeter"      │
    │ (clip_mask_to_bcws)                                      │
    └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │ STAGE 6 — SCORE + PUBLISH  [kgc_figure]                  │
    │ agreement vs. the hint, ML area (ha); overlay            │
    │ serial_<n>.png; <fire>_classified.bin = latest run;      │
    │ new Results entry (fire.serial_results); a fire's        │
    │ only result is accepted automatically                    │
    │ (_auto_accept_first_result)                              │
    └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
      Results gallery -> view / erase / rebrush / Accept
      -> canonical output, polygons, export
```

- Compared with E4: KGC replaces E4's stages 2–6 (sampling, t-SNE, Random Forest, HDBSCAN, cluster classification) with stages 2–4 here; prepare, brush and metrics are shared. E4's stage 1 ("crop + pad") is out of date: the AOI is exactly the drawn box (padding removed, D2 2026-08-08) and stacks are per AOI (D2 2026-08-06).
- Parameters: `KGC_DEFAULTS` (kgc.py) mirror the Args struct in `kgc.cpp` — `kgc_nskip` -1 (stride derived from budget points), `kgc_kmax` 10000, `kgc_kstep` 5, `kgc_patience` 0, `kgc_min_class` -1 (no minimum), `kgc_budget_points` 20000, `kgc_threads` -1 (one per CPU); `build_kgc_cmd()` turns them into the binary's flags. `estimate_memory()` sizes the n × kmax neighbour table, which decides the GPU or CPU build; a compare mode runs both (`compare_cpu_gpu`).
- Batch mapping: the fire list's "Map Selected (with settings)" runs this pipeline for each selected fire by default (`_batch_kgc_fire`, workers.py; p3#19) — the fire's saved KGC parameters (`kgc_params`) else the defaults, its loaded source, earlier Results kept; a fire with no AOI stack or hint is prepared first. Ticking "hdbscan (deprecated)" beside the button runs E4's pipeline instead.

## E4. Fire mapping algorithm — how it works (was FIRE_MAPPING_ALGORITHM.md; verbatim from the tree except one home path written `~/` and two added reader's notes, headings demoted one level) ⚑dep
> Reader's note (added 2026-10-01): this describes the t-SNE + Random Forest + HDBSCAN pipeline, which is DEPRECATED BUT STILL AVAILABLE: ticking `hdbscan` on the fire page (unticked by default) runs it instead of KGC, and the fire list's batch mapping ("Map Selected (with settings)") runs it, over the four settings in `recommended_settings.yaml`, when "hdbscan (deprecated)" is ticked beside the button — unticked, the default since p3#19, a batch uses KGC. KGC is the current method — E3 has its pipeline at a glance. Prepare (stage 1), brushing (stage 7) and metrics (stage 8) are shared with KGC; KGC replaces stages 2–6. Stage 1's "crop + pad" is out of date (E3). Whether to retire it: B D7.

A walk-through of the burn-area mapping pipeline used by
`batch_fire_mapping_web` (and its non-interactive twin
`batch_fire_mapping` / `fire_mapping_cli.py`). The web app is just a UI
wrapper — the actual algorithm lives in
`~/GitHub/wps-research/py/fire_mapping/fire_mapping_cli.py`
plus the C++ helper `cpp/class_brush.cpp`.

---

### 1. The problem

Given a Sentinel‑2 image of an area that burned, decide for every pixel
whether it is **burned** or **unburned**, and produce a clean polygon /
raster of the burn extent.

Inputs:

- **Sentinel‑2 raster** (ENVI `.bin` + header), multi‑band, covering
  the fire season (typically a post-fire scene; the pipeline does not
  require a true pre/post pair — discrimination comes from spectral
  clustering, not differencing).
- **Fire perimeter polygon** (rough boundary, used to crop and as a
  visual reference).
- **Hint mask** — a coarse "this is probably burned" signal. Two
  sources are supported:
  1. **VIIRS active‑fire detections** accumulated over a date window
     around the fire.
  2. **Dominant‑band heuristic** (`dominant_band.py`) — pixels where
     SWIR‑2 (B12) is the brightest band across the stack. Runs
     instantly, no ML, no external data.

The hint is *not* the answer. It is a noisy prior used to
(a) auto-tune cluster size and (b) decide which clusters count as
"burned" once clustering is done.

---

### 2. Pipeline at a glance
> (Reader's note, 2026-10-01: the deprecated pipeline, still available — see the note at the top of E4. The current KGC pipeline at a glance: E3.)

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT                                    │
│  Sentinel-2 raster  +  fire polygon  +  VIIRS shapefile     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 1 — PREPARE  (prepare.py)      │
        │  • crop raster to fire bbox + pad    │
        │  • rasterize VIIRS into hint mask    │
        │  • render preview PNGs               │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 2 — SAMPLE                     │
        │  regular stratified sampling,        │
        │  ~10 000 pixels (sampling.py)        │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 3 — t-SNE  (cuML, GPU)         │
        │  N×B spectral  →  N×2 embedding      │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 4 — RANDOM FOREST regression   │
        │  learn  spectra → t-SNE coords       │
        │  apply to ALL pixels  → 2D map       │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 5 — HDBSCAN  (cuML, GPU)       │
        │  density clusters in 2D embedding    │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 6 — CLASSIFY clusters          │
        │  cluster is burned if                │
        │  precision>50 % OR recall>50 %       │
        │  vs. the hint mask                   │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 7 — BRUSH  (class_brush.cpp)   │
        │  flood-fill, link, threshold,        │
        │  keep largest / union components     │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │ STAGE 8 — METRICS + PERSIST          │
        │  burned area (ha), IoU vs. hint,     │
        │  three-panel comparison PNG          │
        └──────────────────────────────────────┘
                              │
                              ▼
                   Final classified raster
```

Why this shape? Burned and unburned ground have *different spectra*,
but the spectral cloud is curved and noisy — a linear band-difference
threshold (NBR/dNBR) is brittle, especially for partial burn, smoke,
shadows, and water. Embedding the spectra in 2D first and then doing
density clustering lets HDBSCAN find natural groupings without being
told how many classes exist; the hint mask only has to be good enough
to identify *which* of those groupings is "fire".

---

### 3. Stage-by-stage detail

#### Stage 1 — Prepare (`prepare.py::_prepare_fire_sync`)

1. Intersect the fire polygon with the raster footprint (Shapely).
2. Compute the bounding box and apply user‑configurable padding
   (% of fire width / height). Padding lets the model "see" enough
   unburned context to form a contrasting cluster.
3. Crop the raster → `<FIRE>_crop.bin` (+ ENVI header).
4. Rasterize VIIRS detections within `fire_date ± 5 days` into
   `<FIRE>_hint.bin` (binary 0/1) at the crop grid.
5. Generate B12/B11/B9 RGB preview PNGs (with histogram trim) for the
   web UI.

Output: cropped raster, hint mask, optional traditional perimeter
raster, preview PNGs — all under `.web_cache/<FIRE>/`.

#### Stage 2 — Sample (`sampling.py::stratified_sampling`)

t‑SNE on a full Sentinel‑2 crop is too expensive. Instead, draw a
**stratified random sample** of ~10 000 pixels — clamped by the crop
size and `--sample_rate`. By default the sampler aims for 50% of
samples inside the hint mask and 50% outside (`stratify_inside_ratio`,
A1), which keeps the burn class represented even on small fires where
the burn footprint is a tiny fraction of the crop. When one stratum
has too few valid pixels (uninformative hint), the sampler falls back
to uniform random across all valid (non-NaN) pixels. Pass
`--no_stratify` to disable.

For each sampled pixel we also record whether it falls inside the
hint mask — used later for the cluster vote.

#### Stage 3 — t‑SNE embedding (cuML GPU)

```
  input:  N samples × (B + 2)           (scaled spectra + (x, y))
  output: N samples × 2                 (t-SNE coords)
```

t‑SNE places spectrally similar pixels close together in 2D. Burn
scars, healthy vegetation, exposed soil, water, and cloud shadow tend
to fall into separate blobs.

Two preprocessing steps run before t‑SNE:

* **A4 — robust per-band z-score.** Each embed band is centred on its
  median and divided by `1.4826 · MAD` (median absolute deviation).
  Without this, t-SNE's Euclidean distances are dominated by whichever
  band has the largest reflectance scale (typically B12 SWIR-2 over
  forest), drowning out separation between burn and non-burn. Stats
  are computed over the *full image* and applied identically to
  samples and the full-image RF inference, so train/inference scales
  match. Disable with `--no_scale_features`.

* **A8 — spatial coherence features.** Two extra features
  `(x, y)` normalised to `[-1, 1]` and weighted by `--spatial_weight`
  (default 0.3) are appended after the spectral bands. These nudge
  HDBSCAN toward spatially contiguous clusters; two unrelated burns
  in the same crop fall into different clusters even if their spectra
  look similar. Set `--spatial_weight 0` to disable.

Hyperparameters exposed on the CLI:
`perplexity`, `learning_rate`, `max_iter`, `init`, `n_components`,
`random_state`. Defaults live in `recommended_settings.yaml` and are
the first thing analysts tune.

#### Stage 4 — Random Forest extension to the full image

t‑SNE has no `transform()` for new points, so we *learn* the mapping:

- Fit **two Random Forest regressors** (one per t‑SNE axis) using the
  sampled spectra → sampled t‑SNE coords.
- Predict t‑SNE coords for **every** pixel in the crop.

Result: a `H × W × 2` embedding that covers the whole fire, not just
the sample. This is the key trick — t‑SNE's structure on 10k pixels is
extrapolated to millions of pixels by a fast supervised model.

#### Stage 5 — HDBSCAN clustering (cuML GPU)

HDBSCAN groups dense regions in the 2D embedding into clusters and
labels low‑density points as noise (`-1`). It does not need a
predetermined cluster count.

`min_cluster_size` is auto-derived from the hint (A6):

```
hint_burn_proportion = (#hint pixels) / (#sampled pixels)
min_cluster_size = max(5, sample_size · hint_burn_proportion · controlled_ratio)
```

The previous formula used `min(burn_count, non_burn_count) · r`,
which was symmetric in the prior. That made `min_cluster_size` too
large on half-burn fires (eats severity-gradient subclusters) and
too small on tiny fires (over-fragmented background). Tracking the
burn class directly fixes both extremes. `controlled_ratio` (default
0.5) is the user‑facing knob.

HDBSCAN's `approximate_predict` is run only on **finite pixels** —
NaN pixels never enter the cluster (A2) — and we keep the per-pixel
**membership strength** for the vote stage.

#### Stage 6 — Cluster → per-pixel burn probability

Replaces the legacy `precision OR recall > 50 %` hard rule (A5).

For each cluster compute, against the hint mask (over valid pixels
only — NaN/no-data is excluded so it doesn't dilute the prior):

- **Precision** = (cluster ∩ hint) / cluster
- **Recall**    = (cluster ∩ hint) / hint
- **F1**        = 2·P·R / (P + R)
- **Lift**      = (P − burn_prior) / (1 − burn_prior)
- **Score**     = max(0, F1) · max(0, Lift)

Lift penalises clusters whose precision merely tracks the prior
(e.g. a giant background cluster that grazes the hint by chance gets
score ~0 even with high raw recall). The per-pixel burn probability
is the cluster score weighted by HDBSCAN membership strength —
edge-of-cluster pixels contribute proportionally less than core
members:

```
P(burn | x) = score(cluster(x)) · strength(x)
burned     iff  P(burn|x)  >  cluster_score_threshold   (default 0.05)
```

NaN/no-data pixels are forced to `False` so they never enter the
output mask.

Output: binary classification raster (`0` unburned, `1` burned)
plus a soft burn-probability map (kept on the CLI instance as
`last_proba` for diagnostics).

#### Stage 7 — Brush post‑processing (`class_brush.cpp`, called from `brush.py`)

Pure pixel classification leaves salt‑and‑pepper noise and tiny
disconnected blobs. The brush stage cleans up morphologically:

1. **Flood‑fill** 8‑connected components on the binary raster.
2. **Link** components whose bounding boxes are within a sliding
   window of `brush_size` pixels (default 15) using union‑find — this
   merges a fragmented burn scar into one component.
3. **Recode** so labels are contiguous.
4. **Threshold** — discard components smaller than `point_threshold`
   pixels (default 10).
5. Emit one binary raster per surviving component.

**A12 — hint-aware selection (default).** The CLI forces
`--all_segments` from the C++ tool so every component above threshold
appears, then scores each component in Python:

```
precision    = (comp ∩ hint) / comp
recall       = (comp ∩ hint) / hint
proximity    = max(0, 1 − mean_distance_to_hint / (frac · diagonal))
score(comp)  = max(F1, 0.5 · proximity)
```

Proximity uses a Euclidean distance transform of the hint, so a
component that doesn't overlap the hint but sits right next to it
still gets partial credit. Components above
`brush_score_threshold` (default 0.05) are unioned together; if none
clear the bar, the highest-scoring component is kept so the run never
returns an empty mask. Pass `--no_hint_aware_brush` to fall back to
legacy "largest" or `--brush_all_segments` "OR everything" behaviour.

**B3 — `--no-intermediates`.** The C++ tool used to write four full-
image float32 scratch files (`_flood4`, `_link`, `_recode`, `_wheel`)
that the Python wrapper deleted right after the run. The default CLI
now passes `--no-intermediates` so the C++ tool skips the writes
entirely, saving 4× full-image disk I/O per brush call. Pass
`--brush_keep_intermediates` to recover the debug viz.

The pre‑brush mask is preserved as `*_classified_raw.bin` so the
analyst can "rebrush" with different settings without re-running
t‑SNE / RF / HDBSCAN.

#### Stage 8 — Metrics and outputs

- **ML burned area (ha)** = burned‑pixel count × pixel area (from the
  geotransform) / 10 000.
- **Agreement %** = IoU between final ML mask and hint mask. Handles
  rasters cropped at different paddings by aligning on the
  geotransform and computing IoU on the overlap rectangle.
- **Three‑panel comparison PNG**: RGB background + ML perimeter +
  hint + traditional perimeter, for visual review.
- **`accepted_params.csv`** logs every accepted run with its full
  parameter set + agreement %, so good settings can be reused.

---

### 4. Interactive layer (the "web" part)

`workers.py::_serial_map_worker` runs **N recommended settings × K
HDBSCAN replicates** per fire, so the analyst gets a gallery to pick
from instead of a single take‑it‑or‑leave‑it run.

Optimisation: replicate 0 of each setting does the full t‑SNE + RF and
caches the embedding to a `.npz`; replicates 1..K reload that cache
and only re‑run HDBSCAN with jittered `min_samples`:

```
jittered = base + level · sign · step      # 0, +Δ, −Δ, +2Δ, −2Δ, ...
```

That makes the parameter sweep cheap — t‑SNE is the expensive stage,
and we pay it once per setting, not per replicate.

A single `_gpu_lock` serializes all heavy GPU work across the server,
and `_gpu_queue` exposes the wait depth to the UI.

Rebrush is a separate fast path: it reads `*_classified_raw.bin`,
re‑runs only `class_brush.exe` with new parameters, and updates the
output. No GPU needed.

---

### 5. File map (where each piece lives)

| Stage | Code | Notes |
|---|---|---|
| 1. Crop + hint | `prepare.py`, `viirs/utils/{accumulate,rasterize}.py` | called per fire from the web UI |
| 1b. Dominant‑band hint | `py/fire_mapping/dominant_band.py` | fallback when no VIIRS |
| 2. Sampling | `py/fire_mapping/sampling.py::regular_sampling` | |
| 3. t‑SNE | `py/fire_mapping/fire_mapping_cli.py` (cuML) | GPU |
| 4. RF mapping | `fire_mapping_cli.py::rf_regressor` | two regressors, one per axis |
| 5. HDBSCAN | `fire_mapping_cli.py` (cuML) | GPU |
| 6. Cluster vote | `fire_mapping_cli.py` | precision OR recall > 0.5 |
| 7. Brush | `cpp/class_brush.cpp`, wrapped by `brush.py` | flood-fill + link + threshold |
| 8. Metrics + PNG | `mapping.py` | IoU on aligned geotransforms |
| Orchestration | `workers.py`, `app.py` | GPU lock, queue, serial sweep |
| Persistence | `persistence.py`, `accepted_params.csv` | atomic YAML, audit trail |

---

### 6. Why this design over dNBR / thresholding?

Classic burn indices (NBR, dNBR, BAI, NDVI difference) work
band‑arithmetic: one number per pixel, threshold it, done. They are
fast and reproducible but:

- they assume a clean pre/post pair with comparable atmospheric
  conditions,
- they need a per‑fire threshold that varies with vegetation type and
  burn severity,
- they confuse burned ground with bare soil, recent harvests, water,
  shadow, etc.

The pipeline above sidesteps the threshold problem by letting the
data cluster itself in a learned 2D space, and uses the hint only to
*name* clusters, not to threshold pixels. Any noisy "probably burned
somewhere here" signal is enough — VIIRS hotspots, a previous year's
perimeter, even the dominant‑band heuristic. That is what makes the
system robust across fire sizes, sensors‑of‑opportunity, and seasons.

## E5. Brushing — how `class_brush.exe` cleans up the burn mask (was BRUSHING.md; verbatim from the tree except an added reader's note, headings demoted one level) ⚑dep
> Reader's note (added 2026-10-01): brushing is current — KGC results go through the same `class_brush` step (`kgc._brush_classified` → `brush._run_class_brush_only`; E3 stage 5). Flagged ⚑dep only because the text names the deprecated pipeline as where the masks come from.

A full walk-through of the morphological post-processing stage that
turns the raw HDBSCAN classification into the final burn perimeter.
Source of truth: `cpp/class_brush.cpp` (~640 lines), wrapped by
`brush.py` in this package.

---

### 1. What problem brushing solves

HDBSCAN labels every pixel as either "burned" or "not burned" in the
2D t-SNE embedding. That label is **per-pixel and spatially blind** —
the clustering doesn't know that pixel (123,456) is a neighbour of
pixel (123,457). So even a perfect cluster boundary in embedding
space looks like *salt-and-pepper noise* on the actual map: stray
single-pixel "burned" labels in unburned regions, single-pixel
"unburned" holes inside the burn scar, and the burn scar itself often
broken into a main blob plus a fringe of small disconnected fragments.

Three problems to clean up:

1. **Tiny stray fragments** — a few isolated pixels labelled "burned"
   in an otherwise unburned area. Almost always misclassification noise.
2. **Burn-scar fragmentation** — the real burn comes out as one big
   blob plus several smaller pieces nearby. They should be one
   polygon, not many.
3. **Picking the answer** — once cleaned, you want a single binary
   raster: "this is the burn", not a labelled multi-class image.

`class_brush.exe` does all three in six pipeline stages.

---

### 2. Inputs and outputs

```
class_brush.exe <input_mask.bin> <brush_size> <point_threshold> [--all_segments]
```

| Argument | Meaning |
|---|---|
| `input_mask.bin` | Binary classification raster from HDBSCAN: `0` = unburned, non-zero = burned, `NaN` = no-data. ENVI float32. |
| `brush_size` | **Linking-window width in pixels.** Bigger window → more aggressive merging. Default 15. |
| `point_threshold` | **Minimum pixel count** for a component to survive. Default 10. |
| `--all_segments` | Optional. Without this flag (default), only the largest surviving component is kept. With it, every surviving component is written. |

Outputs (all ENVI float32):

| File | Stage that wrote it |
|---|---|
| `<input>_flood4.bin` | Stage 1 (flood-fill labels) |
| `<input>_flood4.bin_link.bin` | Stage 2 (after linking) |
| `<input>_flood4.bin_link.bin_recode.bin` | Stage 3 (contiguous labels) |
| `<input>_flood4.bin_link.bin_recode.bin_wheel.bin` | Stage 4 (RGB visualization) |
| One `.bin` per accepted component | Stage 6 (one-hot mask per component) |

The intermediate files are scratch artefacts. The final answer is
the per-component one-hot raster from Stage 6 — `brush.py` picks the
largest one (or OR's them together with `--all_segments`) and
promotes it back to `<FIRE>_crop.bin_classified.bin`.

---

### 3. The six stages, with code references

```
binary mask                                              0/1 + NaN
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 1 — flood-fill                    │   stage_flood
│  8-connected components                 │   class_brush.cpp:165
│  output: integer labels 1..K            │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 2 — link                          │   stage_link
│  union-find over a brush_size window    │   class_brush.cpp:220
│  merges components within proximity     │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 3 — recode                        │   stage_recode
│  renumber labels to 1..M contiguous     │   class_brush.cpp:276
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 4 — wheel (visualisation only)    │   stage_wheel
│  RGB image with shuffled hues per       │   class_brush.cpp:329
│  label, for QA. Not used by pipeline.   │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│ STAGE 5/6 — count + one-hot output      │   stage_onehot_output
│  drop labels < point_threshold,         │   class_brush.cpp:460
│  pick largest, write per-component      │
│  binary masks                           │
└─────────────────────────────────────────┘
   │
   ▼
binary mask                                              0/1 + NaN
(this is the final burn perimeter)
```

#### Stage 1 — Flood-fill (`stage_flood`, lines 115–198)

Standard iterative flood-fill with **8-connectivity** (a pixel's
neighbours are all 8 surrounding pixels, including diagonals). The
fill distinguishes three pixel kinds:

- `NaN` → preserved as `NaN` (no-data passes through).
- `0.0` → background, label stays 0.
- non-zero → start of a new component, gets label `next_label`,
  flood-fills out to all 8-connected neighbours that share the same
  input value.

Output: a float32 raster where each connected blob has a unique
positive integer label. After this stage, salt-and-pepper noise
shows up as many tiny components.

**Important detail**: the fill uses an explicit stack
(`vector<size_t>`) rather than recursion to avoid stack overflow on
large connected regions. Visited tracking is a separate `uint8_t`
array.

#### Stage 2 — Link (`stage_link`, lines 220–269)

This is the heart of brushing and the one most people misunderstand.
The goal: merge components that are *near each other* even if they
don't physically touch.

Mechanism:

1. **Union-find** initialised with one set per component (label).
2. **Sliding window** of size `nwin × nwin` (where `nwin = brush_size`),
   stepping by `frac = nwin / 2` — so windows overlap by half their
   width.
3. For each window position, collect every distinct non-zero,
   non-NaN label in that window. If more than one label appears in
   the same window, **union all those labels into one set**.
4. After all windows are processed, replace each pixel's label with
   the root of its union-find set.

Effect: if two components are within roughly `brush_size` pixels of
each other in any direction, they end up sharing a label. The
"distance" the linking window sees is bounded — diagonally separated
components may need to fall in the same window to get merged.

**Why this is monotone in `brush_size`**: union-find never splits
sets. A bigger window is a superset of a smaller window's
observations, so any merge that happens at `brush_size = 10` also
happens at `brush_size = 20`. Components only get *more* merged as
`brush_size` grows.

**Code subtlety**: labels are stored as `float` keys in
`unordered_map<float, float>`. Integer-valued floats are exact, so
key lookup is reliable, but the file deliberately re-validates with
`guard_integer_label()` in case a stray fractional value appears.

#### Stage 3 — Recode (`stage_recode`, lines 276–299)

After linking, surviving labels are an arbitrary subset of the
original 1..K (e.g. `{3, 7, 12, 15}` if linking merged everything
else). This stage just renumbers them to `1..M` contiguous, in order
of their original numeric value. Background `0` and `NaN` pass
through unchanged.

Purely cosmetic — needed because Stage 6 iterates `for N = 1 to
n_classes`.

#### Stage 4 — Wheel (`stage_wheel`, lines 329 onward)

Writes a 3-band RGB float32 raster where each component gets a
distinct hue from a shuffled wheel. **Not used downstream** — it's a
QA aid for inspecting the labelling visually in any ENVI viewer.

#### Stage 5/6 — Count + one-hot (`stage_onehot_output`, lines 460–540)

This is where `point_threshold` and `--all_segments` come in.

1. **Pre-scan** (lines 469–475): count pixels per label. No mask
   allocations yet; just a pass over the recoded raster.
2. **Pick the main segment** (lines 478–485): the largest component
   whose pixel count meets `point_threshold`. If no component meets
   the threshold, print "No components found above threshold" and
   exit — the input is empty.
3. **Loop over labels 1..M** (line 494):
   - If the count is below `point_threshold`, **skip** (component
     is noise).
   - If `--all_segments` is off (default) and this isn't the main
     segment, **skip**.
   - Otherwise, build a one-hot binary mask (1 where pixel == this
     label, 0 elsewhere, NaN preserved) and write it to disk.
4. Each accepted component gets its own `.bin` + `.hdr` pair, and a
   line is printed to stdout: `+component <N> <pixel_count>`.

So the default behaviour is **"keep only the largest component above
threshold"** — exactly what you want for a single fire scar.
`--all_segments` is for cases where multiple disjoint scars are
expected.

---

### 4. The two parameters, in plain English

#### `brush_size` (default 15)

**What it controls**: how aggressively nearby fragments get merged
into one component during Stage 2.

**Effect of raising it**: more merging. Components that were separate
become one. The largest component grows or stays the same.

**Effect of lowering it**: less merging. The largest component shrinks
or stays the same.

**Mental model**: imagine each surviving fragment as a magnet, and
`brush_size` as the magnet's reach. Bigger reach → more fragments
clump together.

#### `point_threshold` (default 10)

**What it controls**: minimum pixel count for a component to survive
Stage 5/6.

**Effect of raising it**: stricter. More small components get dropped
as noise. The result polygon shrinks or stays the same.

**Effect of lowering it**: more permissive. Small components survive.
The result polygon grows or stays the same (when `--all_segments`
is on; otherwise only the *largest* surviving component matters, so
lowering this only changes what's eligible to *be* the largest).

#### Interaction

These two are not independent:

- **High `brush_size` + low `point_threshold`**: many small fragments
  get linked into the main blob, and any leftover small ones survive.
  Maximally inclusive — the burn polygon expands to include even
  marginal pixels.
- **Low `brush_size` + high `point_threshold`**: only the genuinely
  large, contiguous burn blob survives. Conservative — the polygon
  is tight around the densest part of the scar.
- **Default (15, 10)**: a middle ground that handles typical
  Sentinel-2 burn scars on 30 m pixels — `brush_size = 15` reaches
  ~450 m, enough to bridge most fragmentation; `point_threshold = 10`
  drops anything smaller than ~9 000 m² (about 1 hectare).

#### Monotonicity (your earlier question)

| Action | Effect on burn polygon area |
|---|---|
| Raise `brush_size` | grows or stays the same |
| Lower `brush_size` | shrinks or stays the same |
| Raise `point_threshold` | shrinks or stays the same |
| Lower `point_threshold` | grows or stays the same |

Each parameter is monotone *in isolation*. The exception, when both
move at once: a low `brush_size` can split a single component into
two, both of which individually clear `point_threshold` — so total
burned-pixel count stays the same, but the polygon is now two
pieces. With `--all_segments` off, only one piece is kept, so the
polygon area can drop sharply. This is the only "non-monotone"
edge case worth knowing about.

---

### 5. How the package consumes brushing

`brush.py::_run_class_brush_only` is the wrapper. It:

1. Spawns `class_brush.exe` as a subprocess via `_stream_subprocess`,
   so its stdout streams into the per-fire console log and the
   `_rebrush_procs` registry can SIGTERM it on cancel.
2. Parses the `+component N <pixel_count>` lines to know which
   one-hot rasters were produced.
3. Either picks the **largest** component file or **OR's all** of
   them (`brush_all_segments`) into a single binary raster.
4. Returns the binary raster to the caller, plus a "cancelled" flag.

The caller (`handlers/rebrush.py` for rebrushing, `workers.py` for
mapping) then:

- Saves the pre-brush mask as `*_classified_raw.bin` if it doesn't
  exist yet (so a subsequent rebrush can re-start from the
  HDBSCAN output, not from an already-brushed mask).
- Overwrites `*_classified.bin` with the brushed result.
- Regenerates `_brush_comparison.png` (raw vs. brushed side-by-side)
  and `_comparison.png` (perimeter overlay on RGB).
- Updates `fire.last_params` with `brush_size`, `point_threshold`,
  `brush_all_segments` so a later accept persists them.

For the **post-accept rebrush** flow (the recent change), the mask
is staged to a per-run filename instead of overwriting the canonical
one, and a gallery entry is appended — same `class_brush.exe`
mechanics, different file destinations.

---

### 6. Edge cases the C++ guards against

- **NaN propagation**: every stage preserves `NaN` pixels untouched.
  No-data never gets accidentally relabelled.
- **Non-integer labels**: `guard_integer_label` exits with an error
  if a label is not an integer-valued float. Defends against
  upstream bugs that might write fractional values.
- **Multi-label one-hot**: `stage_count_onehot` aborts if a
  supposedly one-hot mask contains more than one distinct non-zero
  label. Defends against bugs in Stage 6's mask construction.
- **Allocation overflow**: `mul_overflow` checks `nrow × ncol` for
  size_t overflow on extremely large rasters before allocation.
- **Zero components above threshold**: returns cleanly with a
  message; doesn't crash, doesn't write a bogus output.

---

### 7. What brushing does NOT do

It's worth being explicit about the boundary, because morphological
operations that *sound* similar are absent:

- **No dilation / erosion** — the burn polygon's *outline* is not
  expanded or contracted by `brush_size`. Only component *linking*
  happens. A lone burned pixel surrounded by unburned pixels stays
  exactly that one pixel (until `point_threshold` deletes it).
- **No hole-filling** — small unburned holes inside a burn scar
  remain unburned. If you want them filled, you'd need a separate
  morphological closing.
- **No edge smoothing** — pixel-stair-step edges from HDBSCAN stay
  pixel-stair-step. The polygonize stage that follows (when
  exporting to KML / shapefile) is what produces the polygon vertex
  list, but it doesn't smooth either; for that you'd run
  `ogr2ogr -simplify` on the polygonized output.
- **No reclassification** — brushing operates on *which pixels are
  burned*, not *what kind of burn*. Severity and intensity are
  outside the brush's scope.

If you need any of those, they belong upstream (in HDBSCAN tuning)
or downstream (in vector simplification), not in `class_brush.exe`.

# Part F — Working agreement
Standing instructions, not one-off requests; normalised from 426 of The Maintainer's own prompts across 21 archived transcripts.
- F1 Diagnosis: find the root cause, not the symptom; never paper over a defect with a workaround that hides it; read the relevant code before forming a theory, not after; if not absolutely sure of the cause, ask for logging or clarification before changing any code; state plainly when you don't know something.
- F2 Reasoning: bilateral symmetric reasoning on every proposed change — argue both that it works and that it fails, and resolve the two; trace the program flow through each change and confirm no unintended effects; concurrency — assume multiple threads reach the same code and files at once; don't conflate distinct concepts — keep per-product, per-fire and per-session state separate.
- F3 Implementation: no new bugs; no regressions in behaviour that already works; naming, structure and terminology consistent with the existing code; complete fixes — every instance of the pattern, not the first found; maximum safe parallelism wherever work is independent; asynchronous computation so the GUI and unrelated backend operations are never blocked by work they don't depend on; never modify `kgc.cpp` or `kgc.h`.
- F4 Verification: verify each change before delivering it and say exactly how; confirm every file parses and every script block is syntactically valid; check that names used are actually bound, including across closures; re-check the claim against the evidence before asserting a fix works.
- F5 Communication: be explicit about everything — name what each referenced item is, in the response itself, never making the reader look elsewhere; report what was changed, what was not, and why; own mistakes directly and correct the record when a diagnosis was wrong; treat every factual claim as unproven — "mine as much as your own" — and establish it independently from evidence before relying on it; treat The Maintainer's instructions and suggestions as correct and act on them unless evidence shows otherwise — then say so and show the evidence.
- F6 Delivery: a ZIP that unzips inside `wps-research/`, paths starting `data/bill/batch_fire_mapping_viirs_web/`; attach it in the same message that describes it; state that the server must be restarted after it is applied. Backlog-only revision: present the .md file itself, not a ZIP, unless code changed (The Maintainer, 2026-08-24, original thread). 2026-10-01, the consolidation that also emptied the other .md files: delivered at The Maintainer's request as that ZIP (same path convention) plus the .md presented standalone so it could be read. No passwords or other sensitive information in this file or anything delivered (the tree is public on GitHub): use placeholders such as `<admin_password>`, and write home directories as `~/` (The Maintainer, 2026-10-01). Deprecated methods (HDBSCAN, t-SNE, Random Forest) never appear in user documentation; this developer file keeps them, flagged ⚑dep (The Maintainer, 2026-10-01).

## F7. Known failure modes
From all 21 archived transcripts (875 turns: 426 by The Maintainer, 449 by the assistant): 66 assistant turns explicitly admit an error; 12 turns by The Maintainer report a fix that didn't work. Grouped into recurring classes — what went wrong; evidence; the rule it produced.
Assistant failure classes:
1. String-replace edits matched the wrong occurrence: an anchor appeared earlier in another function, so the patch landed in the wrong place and duplicated a block. "`s.index("w, h = ...")` found an *earlier* occurrence in another function, so the slice duplicated a huge block." Rule: anchors must be unique; verify the match count before editing.
2. Refactor left callers behind: a signature changed, only some call sites updated. "my recent change to `validate_plan_content` returning three values has other callers I missed beyond the two I already updated". Rule: change a signature, then enumerate every caller.
3. Identity keyed on a mutable name, not a stable ID: recreating a deleted fire with the same number inherited the dead one's state. "my carry-forward matches on fire *name*, so recreating a deleted fire with the same ID inherits…"; a fire whose `bbox_native` was another fire's box. Rule: key on identity, not display name; purge all state on delete.
4. Theory formed before reading the code. "I should have read the stylesheet before reasoning about CSS defaults."; the 2026-09-25 temp-filename fix. Rule: F1.
5. Confident claim later reversed once the evidence was re-read. "That line settled it, and I was wrong."; "I was wrong last time to call the sweep cheap."; "You were right and I was wrong." Rule: F4.
6. Regression introduced while adding a feature: GDAL environment variables set at startup broke the reader; `SPLIT_AT_LOAD` set the DOM class without setting `splitActive`. Rule: F3; trace the flow through the change.
7. Implemented the wording, not the intent, or put the change in the wrong function. "I implemented your original wording literally."; "No, I did not implement what you asked. I put the creation-path changes in `_prep…`". Rule: restate the intent before coding; confirm the target function.
8. Scoping and import errors: `from ..aoi_stack` with two dots; a closure bug from the ENVI-header change; a log line placed before `app_state` exists. Rule: check name binding across closures and module load order.
9. Geometry mismatch between layers: overlay geometry computed against full-size rasters while previews were downsampled to 2000 px; syncing by fraction of image, not ground coordinates. Rule: anchor overlays to ground coordinates, never pixel fractions.
10. Silent scope creep: a control added that was not requested and not flagged. "It's harmless (unchecked by default) but I should have said so at the time rather than presenting it as part of your spec." Rule: report what was changed *and* what was added beyond the request.
11. Destructive or unbounded operations: a diagnostic script globbed from `/`, walking the whole filesystem and hanging the server. Rule: bound every filesystem scan to the project tree.
12. Misattributed cause: a defect blamed on recent work when it pre-dated it, or vice versa. "This is a bug I introduced during an earlier widening pass, not something caused by the menu work as the user assumed". Rule: establish when a defect was introduced before attributing it.
13. Specification drift in presentation: orange `(1.0, 0.65, 0.0)` used where ML classification uses red `(0.9, 0.1, 0.0)`; L2 post bands mislabelled. Rule: keep colours, labels and terminology consistent with the existing code.
Shared and human-side failures:
14. Ambiguous initial wording, implemented literally — the assistant built what was written, not what was meant, and didn't ask. "That's my error: I implemented your original wording literally." Rule: the assistant restates intent before coding; The Maintainer flags loose wording.
15. Defect attributed to the most recent change when it pre-dated it. "not something caused by the menu work as the user assumed". Rule: confirm when a defect first appeared before assuming the last change caused it.
16. Work requested past the point where it could be done reliably: on 2026-09-25 the assistant flagged exhausted context, further implementation was requested and produced an unverified fix. Evidence: the conversation in which F7 was written. Rule: stop and hand off rather than ship unverified work.
Meta-pattern: classes 1, 2, 3, 6, 8 are one shape — a change applied in one place when the pattern existed in several; 4, 5, 12 are another — a conclusion reached before the evidence supported it. Those two sentences cover most failures in this project.

# Part G — Caveats and limitations
- G1 Not a requirements specification. Part D records what was changed and why; it can't become a specification able to recreate the app, for structural reasons, not effort: (a) prompts describe deltas, not state — the 426 prompts in 21 transcripts are overwhelmingly "X is broken, fix it" or "change X to Y"; they presuppose the program, and replaying every delta yields no specification because the base they applied to is never stated; (b) most behaviour was never specified in words — of roughly 50,000 lines, most behaviour (data structures, error handling, caching policy, threading, file layout) was written by the assistant without a prompt describing it and exists only in the source; (c) requirements changed silently — many decisions were later reversed without retracting the earlier statement, so a prompt-built specification would hold mutually contradictory requirements with no way to tell which survived; (d) the source is the only complete, self-consistent artefact — `fire_mapping_source_export.zip` (110 files; now `fire_mapping_source_current.zip`) is the specification in the only complete form. Needed instead: a specification derived from the code, module by module, with the transcripts used only to recover the intent behind non-obvious decisions — a substantial separate exercise, done by reading the source, not the chat.
- G2 Limits of F7: classes came from pattern-matching phrases such as "my bug", "I was wrong", "still doesn't work", so failures neither party noticed or named are missing — likely the larger set; The Maintainer's side is under-counted (12 matching turns vs 66 assistant admissions), reflecting who is expected to report errors in this working relationship, not the true distribution; the most recent session was only partly covered (earlier turns compacted to a summary, so no quotations from them).
- G3 Provenance: Parts A–E written and maintained during the sessions; F1–F6 normalised from 426 archived prompts plus two rules added after the 2026-09-25 misdiagnoses; F7 from all 875 archived turns; Part G written 2026-09-29. 2026-09-29: part #2 delta merged, every open item re-checked against `fire_mapping_source_current.zip`, document compressed; Part E re-derived from the tree's README.md, PLAN.md, FIRE_MAPPING_ALGORITHM.md and BRUSHING.md, because the earlier embedding had turned `#` comments inside code blocks into headings. Where Part D or E disagrees with the source, the source wins; neither has been re-verified line by line against the shipped code (the 09-29 check covered the open items in A–C and the notes marked "src 09-29"). 2026-10-01: part #3 delta merged (18 done, 3 rejected, 15 open); open items re-checked against `fire_mapping_source_after_part3.zip` (notes on the six changed code files re-checked in the code; the other 60 files byte-identical to 09-29); the tree's other .md files confirmed contained here — PLAN.md, FIRE_MAPPING_ALGORITHM.md and BRUSHING.md verbatim (E2, E4, E5); README.md, IMPROVEMENTS.md and the old BACKLOG.md by every identifier, path, number and code-block token, with the remaining reworded passages read by hand — then emptied; deprecated method names flagged ⚑dep, not removed. Later on 2026-10-01: the owner's name replaced by "The Maintainer" throughout, the quick-start passwords in E1 by placeholders, and the home directories in four paths (Start here, E1, E4) by `~/` — so E4 is verbatim except for that one path. Review the same day, in the part #3 thread: E1 corrected against `__main__.py` (`--viirs_concurrent_jobs` default 2; `--padding` ignored, default 0.0; 11 missing flags added); line counts corrected (`wc -l`: 60,940 now, 59,850 at 09-29); the D1 and D2 claims that GUI state came back now point to p3#1; ⚑dep wording aligned with the rule (user documentation only); E4's verbatim notes qualified; every item in A–C re-checked against `fire_mapping_source_after_part3.zip`: status notes added to A1 #6, #7 and C2 #11, #13, #17, #22, #23, #31, #32; new A3 #5 and B D7; two part #3 code findings in A4; batch mapping recorded in D1. Then: reader's notes added to E2 (what the build plan is; how to read Remove / Keep / Add), E4 (deprecated but still available) and E5 (brushing is current); E3 added — the KGC pipeline at a glance, written from `kgc.py`, `mapping_cmd.py` and `handlers/serial.py`; batch mapping found to run only the deprecated pipeline (D1, C2 #13, B D7). Later the same day: Part E reordered so the current KGC pipeline comes before the deprecated one — KGC is now E3, the deprecated algorithm E4, brushing E5, every reference updated; p3#19 recorded (batch mapping uses KGC by default, the old pipeline optional) in D1, C2 #13, B D7, A4, E3 and E4.
