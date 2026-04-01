# Methodology — batch_fire_mapping

*Last updated: March 30, 2026*

This document describes the classification methodology used by `batch_fire_mapping`, with emphasis on the unsupervised burn-mapping pipeline and the changes made to improve its robustness.

---

## Pipeline overview

For each fire polygon that overlaps the input Sentinel-2 raster:

1. **Crop** the raster to the fire's bounding box (+ configurable padding).
2. **Obtain a burn hint** — a binary raster indicating which pixels are likely burned. VIIRS active fire detections are preferred; the traditional fire perimeter is used as fallback.
3. **Sample** a subset of pixels from the full crop (default 5% of pixels, clamped to [500, 30 000]).
4. **T-SNE** embeds the sampled pixels from spectral space into 2D (GPU-accelerated via cuML).
5. **Random Forest regressors** learn the mapping from spectral bands to T-SNE coordinates on the samples, then predict T-SNE coordinates for every pixel in the crop.
6. **HDBSCAN** clusters the samples in 2D T-SNE space, then `approximate_predict` assigns cluster labels to the full image via the RF-projected coordinates.
7. **Classify** each cluster as burned or unburned based on its overlap with the burn hint.
8. **class_brush** (C++ connected-component post-processing) cleans up the binary map.

---

## Classification detail

### Sampling

Pixels are drawn uniformly at random from the full crop (all land-cover types, not just the hint area). NaN pixels are excluded. The sample size is computed as:

```
sample_size = clip(crop_w * crop_h * sample_rate, min_samples, max_samples)
```

Default: `sample_rate=0.05`, `min_samples=500`, `max_samples=30 000`.

### T-SNE embedding (samples only)

T-SNE is applied to the sampled pixels using their spectral band values (all bands by default, or a user-specified subset). This produces a 2D embedding that captures the spectral structure of the landscape.

| Parameter | Default |
|---|---|
| `perplexity` | 60 |
| `learning_rate` | 200 |
| `max_iter` | 2000 |
| `init` | pca |
| `n_components` | 2 |

### Random Forest image embedding (samples → full image)

T-SNE is non-parametric — it does not define a function that can be applied to new points. To extend the embedding to every pixel in the crop, two independent RF regressors are trained:

- **RF-1**: `spectral_bands → T-SNE dimension 1`
- **RF-2**: `spectral_bands → T-SNE dimension 2`

Both are trained on the sampled pixels (input = band values, target = T-SNE coordinates). They are then applied to every pixel in the crop, producing a 2D "pseudo-embedding" for the full image.

| Parameter | Default |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 15 |
| `max_features` | sqrt |

### HDBSCAN clustering

HDBSCAN is fitted on the real T-SNE embedding (samples only), then `approximate_predict` assigns cluster labels to the RF-projected full image.

#### `min_cluster_size`

| | Old | New |
|---|---|---|
| **Formula** | `min(sample_size * burn_p * ratio, sample_size * (1 - burn_p) * ratio)` | `sample_size * controlled_ratio` |
| **Depends on hint quality** | Yes — `burn_p` comes from the VIIRS raster | No — purely a function of sample size |
| **Default ratio** | 0.5 | 0.05 |
| **Typical value** (10 000 samples) | Varies with burn fraction (e.g. 1 500 at 30% burn) | 500 (fixed) |

**Why the change.** The old formula created a circular dependency: the VIIRS hint quality directly controlled the clustering granularity. If VIIRS overestimated the burn area (common with 375 m pixels), `min_cluster_size` was pushed higher, causing HDBSCAN to merge distinct spectral clusters. If VIIRS underestimated, `min_cluster_size` dropped, fragmenting real clusters. Decoupling removes this feedback loop.

### Cluster → burned/unburned classification

| | Old | New |
|---|---|---|
| **Assumption** | Exactly 2 clusters (label 0 and 1) | Any number of clusters |
| **Decision rule** | Majority vote inside hint picks cluster 0 or 1 | Per-cluster overlap: fraction of that cluster's pixels falling inside the hint |
| **Burned criterion** | Whichever of cluster 0/1 has majority inside hint | Any cluster with >50% of its pixels inside the hint |
| **Multi-cluster handling** | Clusters 2, 3, 4, ... silently ignored | All clusters evaluated; multiple can be labeled burned |
| **Noise handling** | Noise (label −1) implicitly excluded | Noise (label −1) explicitly skipped |

**Old logic:**

```python
masked_cluster = cluster[hint]
valid = masked_cluster[masked_cluster != -1]

if valid.mean() > 0.5:
    classification[cluster == 1] = True
else:
    classification[cluster == 0] = True
```

**New logic:**

```python
for label in unique_labels:
    if label == -1:
        continue
    mask = (cluster == label)
    overlap = (mask & hint).sum() / mask.sum()
    if overlap > 0.5:
        burned_labels.append(label)

for label in burned_labels:
    classification[cluster == label] = True
```

**Why the change.** HDBSCAN can produce more than 2 clusters — for example, different burn severities, ash vs charred vegetation, or spectrally distinct unburned cover types. The old code only ever checked cluster 0 or 1, silently discarding any other cluster even if it represented genuinely burned pixels. The new approach evaluates every cluster independently, so a fire with 3 distinct burned spectral signatures will correctly classify all three.

---

## Burn hint selection

The burn hint is a binary raster used solely to interpret HDBSCAN clusters (which cluster = burned). It is **not** used as training labels.

| | Old | New |
|---|---|---|
| **VIIRS available** | VIIRS raster used as hint; traditional perimeter shown for comparison | Same |
| **VIIRS unavailable** | Generate a `_no_viirs.png` (perimeter outline only); **fire is not classified** | Traditional perimeter polygon **filled and rasterized**, used as the classification hint; fire is fully classified |
| **Neither available** | Partial result (PNG only) | Hard fail — fire skipped entirely |
| **Tracking** | `perimeter_type` not recorded | `perimeter_type: viirs` or `perimeter_type: traditional` saved in `_params.yaml` |

**Why the change.** The traditional fire perimeter and VIIRS detections serve the same conceptual role: an approximate boundary of the burned area. When VIIRS data is missing (common for older fires or regions with persistent cloud cover), there is no reason to skip classification entirely. The filled perimeter polygon provides a reasonable substitute — it may be coarser than VIIRS, but it is sufficient to tell the classifier which spectral clusters correspond to burned land.

### Perimeter rasterization

The traditional perimeter is a vector polygon (shapefile). To use it as a hint, it is rasterized onto the crop grid with `ALL_TOUCHED=TRUE`, producing a filled binary raster where every pixel inside the perimeter = 1. This contrasts with the old approach which only drew the perimeter outline for visual inspection.

---

## Diagnostic outputs

Each fire now produces diagnostic PNGs to aid visual inspection of the input data. Band groups are auto-detected from the ENVI header band names:

| Group | Detection rule | Example band names |
|---|---|---|
| **Pre-fire** | Band name starts with `pre` | `pre MSIL2A 20m: B12 2190nm` |
| **Post-fire** | Band name starts with `pst` or `post` | `pst MSIL2A 20m: B12 2190nm` |
| **Difference 1** | Contains `anomaly1` or `(post-pre)/(post+pre)` | `anomaly1: B12 2190nm (post-pre)/(post+pre)` |
| **Difference 2** | Contains `anomaly2` or `post/pre` | `anomaly2: B12 2190nm post/pre` |

If no keywords are found, the code falls back to positional B12/B11/B9 group detection (first group = pre, second = post).

Up to 4 PNGs are generated per fire:

| File | Content |
|---|---|
| `{fire}_pre.png` | Pre-fire false-colour RGB (first 3 bands of pre group) |
| `{fire}_post.png` | Post-fire false-colour RGB (first 3 bands of post group) |
| `{fire}_diff1.png` | Normalised difference `(post-pre)/(post+pre)` RGB |
| `{fire}_diff2.png` | Ratio `post/pre` RGB |

Each channel is independently stretched to [0, 1] via the 2nd–98th percentile. The `.bin` raster files are preserved alongside the PNGs.

---

## Output per fire

```
fire_mapping_results/<FIRE_NUMBE>/
    <FIRE_NUMBE>_crop.bin/.hdr           # Cropped Sentinel-2 subscene
    <FIRE_NUMBE>_perimeter.bin/.hdr      # Filled traditional perimeter (always)
    VIIRS_VNP14IMG_<s>_<e>.shp           # Accumulated VIIRS (when available)
    VIIRS_VNP14IMG_<s>_<e>.bin/.hdr      # Rasterized VIIRS hint (when available)
    <crop>_classified.bin/.hdr           # Binary burned/unburned classification
    <FIRE_NUMBE>_comparison.png          # Outline comparison figure
    <FIRE_NUMBE>_brush_comparison.png    # Before/after class_brush
    <FIRE_NUMBE>_pre.png                 # Diagnostic: pre-fire RGB
    <FIRE_NUMBE>_post.png                # Diagnostic: post-fire RGB
    <FIRE_NUMBE>_diff1.png               # Diagnostic: normalised difference
    <FIRE_NUMBE>_diff2.png               # Diagnostic: ratio
    <FIRE_NUMBE>_params.yaml             # Full run parameters + perimeter_type
```

---

## Parameters YAML

The `_params.yaml` file now includes a `perimeter_type` field:

```yaml
inputs:
  raster: /path/to/raster.bin
  hint_bin: /path/to/hint_used.bin       # whichever was actually used
  viirs_bin: /path/to/viirs.bin          # null if unavailable
  perimeter_bin: /path/to/perimeter.bin
  perimeter_type: viirs                  # or 'traditional'
```
