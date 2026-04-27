# Fire Mapping Algorithm — How It Works

A walk-through of the burn-area mapping pipeline used by
`batch_fire_mapping_web` (and its non-interactive twin
`batch_fire_mapping` / `fire_mapping_cli.py`). The web app is just a UI
wrapper — the actual algorithm lives in
`/home/bill/GitHub/wps-research/py/fire_mapping/fire_mapping_cli.py`
plus the C++ helper `cpp/class_brush.cpp`.

---

## 1. The problem

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

## 2. Pipeline at a glance

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

## 3. Stage-by-stage detail

### Stage 1 — Prepare (`prepare.py::_prepare_fire_sync`)

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

### Stage 2 — Sample (`sampling.py::regular_sampling`)

t‑SNE on a full Sentinel‑2 crop is too expensive. Instead, draw a
**regular (stratified) grid** of ~10 000 pixels — clamped by the crop
size and `--sample_rate`. Stratification preserves spatial coverage so
both fire and non‑fire regions appear in the sample.

For each sampled pixel we also record whether it falls inside the
hint mask — used later for the cluster vote.

### Stage 3 — t‑SNE embedding (cuML GPU)

```
  input:  N samples × B bands           (float)
  output: N samples × 2                 (t-SNE coords)
```

t‑SNE places spectrally similar pixels close together in 2D. Burn
scars, healthy vegetation, exposed soil, water, and cloud shadow tend
to fall into separate blobs. Hyperparameters exposed on the CLI:
`perplexity`, `learning_rate`, `max_iter`, `init`, `n_components`,
`random_state`. Defaults live in `recommended_settings.yaml` and are
the first thing analysts tune.

### Stage 4 — Random Forest extension to the full image

t‑SNE has no `transform()` for new points, so we *learn* the mapping:

- Fit **two Random Forest regressors** (one per t‑SNE axis) using the
  sampled spectra → sampled t‑SNE coords.
- Predict t‑SNE coords for **every** pixel in the crop.

Result: a `H × W × 2` embedding that covers the whole fire, not just
the sample. This is the key trick — t‑SNE's structure on 10k pixels is
extrapolated to millions of pixels by a fast supervised model.

### Stage 5 — HDBSCAN clustering (cuML GPU)

HDBSCAN groups dense regions in the 2D embedding into clusters and
labels low‑density points as noise (`-1`). It does not need a
predetermined cluster count.

`min_cluster_size` is auto-derived from the hint:

```
hint_burn_proportion = (#hint pixels) / (#sampled pixels)
min_cluster_size = max(5, min(
        sample_size · hint_burn_proportion · controlled_ratio,
        sample_size · (1 - hint_burn_proportion) · controlled_ratio))
```

`controlled_ratio` (default 0.5) is the user‑facing knob; the
formula keeps `min_cluster_size` reasonable whether the fire fills
most of the crop or just a small piece of it.

### Stage 6 — Cluster → burned/unburned vote

For each cluster compute, against the hint mask:

- **Precision** = (cluster ∩ hint) / cluster
- **Recall**    = (cluster ∩ hint) / hint

```
cluster is BURNED  iff  precision > 50 %  OR  recall > 50 %
```

Why OR? The hint is unreliable at *both* extremes:

- A small intense fire may produce a hint that sits well inside the
  true burn footprint → high precision, low recall (the cluster is
  bigger than the hint). OR rule keeps it.
- A sprawling fire with sparse VIIRS coverage produces a hint that
  only samples part of the burn → some clusters cover most of the
  hint (high recall) without covering only the hint (low precision).
  OR rule keeps those too.

Output: binary classification raster (`0` unburned, `1` burned).

### Stage 7 — Brush post‑processing (`class_brush.cpp`, called from `brush.py`)

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

`brush.py` then either keeps the **largest** component or takes the
**logical OR** of all of them (`brush_all_segments`). The pre‑brush
mask is preserved as `*_classified_raw.bin` so the analyst can
"rebrush" with different settings without re-running t‑SNE / RF /
HDBSCAN.

### Stage 8 — Metrics and outputs

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

## 4. Interactive layer (the "web" part)

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

## 5. File map (where each piece lives)

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

## 6. Why this design over dNBR / thresholding?

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
