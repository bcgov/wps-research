# Sentinel-2 Cloud Masking

Per-date Sentinel-2 cloud masking via Random Forest regression (A:B::C:D), followed by optional mask-to-nodata cleaning, MRAP compositing, and MP4 generation.

---

## Overview

The pipeline has up to four stages, gated by flags:

1. **ABCD_MASK** — trains a Random Forest on `(image, cloud_probability_mask)` pairs and predicts a cloud probability map for every date.
2. **MASK_TO_NODATA** — combines the predicted cloud mask with any additional masks (e.g. shadow, nodata) and sets masked pixels to NaN. Output is saved to `L{level}_{grid}/` (e.g. `L2_T09UYU/`), inferred automatically from the image metadata.
3. **sentinel2_mrap** — runs MRAP compositing on the cleaned tile folder.
4. **sentinel2_mp4** — generates an MP4 from the cleaned tile folder.

By default the script stops after Stage 1. Pass `--run_mask2nan` to continue to Stages 2–4.

---

## Usage

```bash
python3 -m production.sentinel2_cloud_masking <image_dir> <cloud_mask_dir> [options]
```

---

## Setup: aliases

Add the following to your `~/.bashrc`:

```bash
alias sentinel2_mp4='~/GitHub/wps-research/cpp/sentinel2_mp4 .'
alias sentinel2_mrap.py='python3 ~/GitHub/wps-research/py/sentinel2_mrap.py'
```

Then reload:

```bash
source ~/.bashrc
```

---

### Positional arguments

| Argument | Description |
|---|---|
| `image_dir` | Directory containing L1C/L2A satellite images (`.bin` files). |
| `cloud_mask_dir` | Directory of reference cloud probability masks used to **train** the RF model. |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--run_mask2nan` | `False` | If set, run MASK_TO_NODATA → MRAP → MP4 after cloud masking. Without this flag the script stops after Stage 1. |
| `--skip_f` | `5555` | Spatial sampling stride for RF training (higher = faster, less accurate). |
| `--save_model` | `False` | Save the trained RF model as `.pkl` files inside the output dir. |
| `--mask_threshold` | `1e-4` | Probability threshold above which a pixel is considered masked. |
| `--output_dir_mask` | `./cloud_mask_{skip_f}` | Where predicted cloud masks are written. |
| `--extra_mask_dirs` | `None` | Comma-separated additional mask directories applied during cleaning (e.g. `C11659/shadow,C11659/nodata`). |
| `--predicted_cloud` | `None` | Path to an already-predicted cloud mask directory. **Skips ABCD_MASK entirely.** |
| `--n_workers` | `8` | Number of parallel workers for the cleaning stage. |

---

## Output directory

The cleaning output directory is **always** inferred from the image metadata as `L{level}_{grid}/` (e.g. `L1_T09UYU/` or `L2_T09UYU/`). This is required because `sentinel2_mrap` expects to find cleaned files at exactly that path relative to the working directory.

Make sure you run the script from the directory **one level above** where the tile folder should live.

---

## Examples

### Stage 1 only — predict cloud masks, inspect before deciding

```bash
python3 -m production.sentinel2_cloud_masking \
    C11659/L1C \
    C11659/cloud \
    --skip_f 10000
```

### Full run — predict, clean, MRAP, MP4 (L2 data)

```bash
python3 -m production.sentinel2_cloud_masking \
    C11659/L1C \
    C11659/cloud \
    --skip_f 10000 \
    --extra_mask_dirs C11659/shadow,C11659/nodata \
    --run_mask2nan
```

### Full run with L1 data

The `--L1` flag is handled automatically based on the level extracted from the image filenames — no extra flag needed.

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

```bash
# 1. Run cloud masking only
python3 -m production.sentinel2_cloud_masking C11659/L1C C11659/cloud --skip_f 10000

# 2. Inspect PNGs in ./cloud_mask_10000/

# 3. Happy with results? Run the full pipeline, skipping the masking step
python3 -m production.sentinel2_cloud_masking C11659/L1C C11659/cloud \
    --predicted_cloud ./cloud_mask_10000 \
    --extra_mask_dirs C11659/shadow,C11659/nodata \
    --run_mask2nan
```

---

## Output

- **`./cloud_mask_{skip_f}/`** — predicted cloud probability maps as ENVI `.bin/.hdr` files, plus side-by-side `.png` previews (Sen2Cor vs predicted).
- **`L{level}_{grid}/`** — cloud-free images as ENVI `.bin/.hdr` files with masked pixels set to NaN, plus before/after `.png` previews. Also contains MRAP composites and is the working directory for MP4 generation.
- **`./cloud_mask_{skip_f}/models/`** — trained `.pkl` RF models per date (only if `--save_model` is set).

---

## Interactive prompt

When `--run_mask2nan` is set, the script pauses after cloud masking and asks:

```
[REVIEW] Please check the cloud mask at: /absolute/path/to/cloud_mask_10000
Are you satisfied with the mask? Continue to mask2nodata? [Y/n]:
```

Press **Enter** or type `Y` to proceed. Type `n` to abort — then re-run with `--predicted_cloud` pointing to the mask directory to skip straight to cleaning.
