# sentinel2_cloud_masking

Per-date Sentinel-2 cloud masking via Random Forest regression (A:B::C:D), followed by optional mask-to-nodata cleaning.

---

## Overview

The pipeline has two stages:

1. **ABCD_MASK** — trains a Random Forest on `(image, cloud_probability_mask)` pairs and predicts a cloud probability map for every date.
2. **MASK_TO_NODATA** — combines the predicted cloud mask with any additional masks (e.g. shadow, nodata) and sets masked pixels to NaN in the output images.

By default the script stops after Stage 1 so you can inspect the predicted masks. Pass `--run_mask2nan` to continue to Stage 2.

---

## Usage

```bash
python3 -m production.sentinel2_cloud_masking <image_dir> <cloud_mask_dir> [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `image_dir` | Directory containing L1C satellite images (`.bin` files). |
| `cloud_mask_dir` | Directory of reference cloud probability masks used to **train** the RF model. |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--run_mask2nan` | `False` | If set, run MASK_TO_NODATA after cloud masking. Without this flag the script stops after Stage 1. |
| `--skip_f` | `5555` | Spatial sampling stride for RF training (higher = faster, less accurate). |
| `--save_model` | `False` | Save the trained RF model as `.pkl` files inside the output dir. |
| `--mask_threshold` | `1e-4` | Probability threshold above which a pixel is considered masked. |
| `--output_dir_mask` | `./cloud_mask_{skip_f}` | Where predicted cloud masks are written. |
| `--output_dir_clean` | `./mask_removed_skip={skip_f}` | Where cleaned (masked-to-NaN) images are written. |
| `--extra_mask_dirs` | `None` | Comma-separated additional mask directories applied during cleaning (e.g. shadow, nodata). |
| `--predicted_cloud` | `None` | Path to an already-predicted cloud mask directory. **Skips ABCD_MASK entirely.** |
| `--n_workers` | `8` | Number of parallel workers for the cleaning stage. |

---

## Examples

### Stage 1 only — predict cloud masks, inspect before deciding

```bash
python3 -m production.sentinel2_cloud_masking \
    C11659/L1C \
    C11659/cloud \
    --skip_f 10000
```

### Full run — predict cloud masks then clean

```bash
python3 -m production.sentinel2_cloud_masking \
    C11659/L1C \
    C11659/cloud \
    --skip_f 10000 \
    --extra_mask_dirs C11659/shadow,C11659/nodata \
    --run_mask2nan
```

### Skip cloud masking, jump straight to cleaning with pre-computed masks

```bash
python3 -m production.sentinel2_cloud_masking \
    C11659/L1C \
    C11659/cloud \
    --predicted_cloud ./cloud_mask_10000 \
    --extra_mask_dirs C11659/shadow,C11659/nodata \
    --run_mask2nan
```

---

## Typical workflow

```
# 1. Run cloud masking only
python3 -m production.sentinel2_cloud_masking C11659/L1C C11659/cloud --skip_f 10000

# 2. Inspect PNGs in ./cloud_mask_10000/

# 3. Happy with results? Run cleaning, skipping the masking step
python3 -m production.sentinel2_cloud_masking C11659/L1C C11659/cloud \
    --predicted_cloud ./cloud_mask_10000 \
    --extra_mask_dirs C11659/shadow,C11659/nodata \
    --run_mask2nan
```

---

## Output

- **`{output_dir_mask}/`** — predicted cloud probability maps as ENVI `.bin/.hdr` files, plus side-by-side `.png` previews (Sen2Cor vs predicted).
- **`{output_dir_clean}/`** — cloud-free images as ENVI `.bin/.hdr` files with masked pixels set to NaN, plus before/after `.png` previews.
- **`{output_dir_mask}/models/`** — trained `.pkl` RF models per date (only if `--save_model` is set).

---

## Interactive prompt

When `--run_mask2nan` is set, the script pauses after cloud masking and asks:

```
[REVIEW] Please check the cloud mask at: /absolute/path/to/cloud_mask_10000
Are you satisfied with the mask? Continue to mask2nodata? [Y/n]:
```

Press **Enter** or type `Y` to proceed. Type `n` to abort — then re-run with `--predicted_cloud` pointing to the mask directory to skip straight to cleaning.