# s2lookback

A small library for searching, reading, and processing Sentinel-2 imagery and cloud masks over a date range. Built around a shared `LookBack` base class, it provides three ready-to-use workflows: cloud-free compositing (MRAP), cloud mask training and inference (MASK), mask-to-nodata conversion (MASK_TO_NODATA), and per-date A:B::C:D cloud probability prediction (ABCD_MASK).

## Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `base.py` | `LookBack` | Base dataclass — file discovery, image/mask reading, pixel sampling |
| `mrap.py` | `MRAP` | Most-Recent-Available-Pixel compositing with optional lighting adjustment |
| `mask.py` | `MASK` | Progressive Random Forest cloud mask training and inference |
| `mask2nan.py` | `MASK_TO_NODATA` | Sets masked pixels to NaN and saves clean images |
| `abcd_mask.py` | `ABCD_MASK` | Per-date RF regression cloud probability prediction (A:B::C:D) |
| `utils.py` | — | File discovery helpers: `get_ordered_file_dict`, `get_dates_within` |

## Installation

The module is part of the `wps-research` repository. Install the repository dependencies from the root:

```bash
pip install -r requirements.txt
```

Then import directly — no separate install step is needed:

```python
from s2lookback.mrap import MRAP
from s2lookback.mask import MASK
```

## LookBack — Base Class

All workflow classes inherit from `LookBack`. Its key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_dir` | `str` or `list[str]` | required | Path(s) to image directories (`.bin` ENVI files) |
| `mask_dir` | `str` or `list[str]` | `None` | Path(s) to cloud mask directories |
| `output_dir` | `str` | `'s2lookback_temp'` | Where outputs are written |
| `start` | `datetime` | `None` | Start of date range filter |
| `end` | `datetime` | `None` | End of date range filter |
| `mask_threshold` | `float` | `1e-5` | Probability threshold for converting probabilistic masks to boolean |
| `max_lookback_days` | `int` | `7` | Maximum days to look back when filling |
| `n_workers` | `int` | `8` | Number of parallel workers |
| `sample_size` | `int` | `None` | Total pixel sample budget for `sample_datasets` |
| `sample_between_prop` | `dict` | `{'mask': 0.5, 'non_mask': 0.5}` | Fraction of `sample_size` allocated to each class |
| `sample_within_prop` | `dict` | `{'mask': 0.8, 'non_mask': 0.5}` | Maximum fraction of available pixels per class to draw |

### Key Methods

**`get_file_dictionary()`** — Scans `image_dir` and `mask_dir` for `.bin` files, extracts acquisition timestamps from raster metadata, filters to `[start, end]`, and returns a `dict[datetime, dict]` with `image_path` and `mask_path` entries. Raises `ValueError` if no files are found.

**`read_image(date, band_list='all')`** — Reads image bands for a given date. Returns a `numpy.ndarray` of shape `(H, W, B)`. If `image_dir` was a list, returns a list of arrays.

**`read_mask(date, as_prob=False)`** — Reads one or more mask files for a date. Scales masks in `[0, 100]` down to `[0, 1]` automatically. With `as_prob=False` (default): combines all masks with logical OR using `mask_threshold`, returns `(bool_mask, coverage)`. With `as_prob=True`: returns raw float arrays.

**`sample_datasets(img_dat, mask)`** — Randomly samples masked (label=1) and non-masked (label=0) pixels from a pre-filtered image array. Respects `sample_between_prop` and `sample_within_prop`. Returns `(X, y)` suitable for scikit-learn.

**`get_nodata_mask(img_dat, nodata_val=nan)`** — Returns a `(H, W)` boolean mask where all bands are equal to `nodata_val`.

## MRAP — Most-Recent-Available-Pixel Composite

Fills cloud-covered pixels by pulling from the most recent prior acquisition within `max_lookback_days` that has clear data at that location. Remaining gaps are set to NaN.

```python
from s2lookback.mrap import MRAP

mrap = MRAP(
    image_dir='/data/sentinel2/L2',
    mask_dir=['/data/cloud_masks', '/data/shadow_masks'],
    output_dir='/data/output/mrap',
    max_lookback_days=60,
    n_workers=12,
    adjust_lighting=True,
    mask_threshold=0.0001,
)

mrap.fill()
```

**Additional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adjust_lighting` | `False` | Fit a per-band linear regression on overlapping clear pixels to normalize brightness before filling |
| `max_cov_for_adjust_lighting` | `0.7` | Skip lighting adjustment if the main image's cloud coverage exceeds this fraction |
| `min_adjust_sample_size` | `5000` | Minimum overlapping pixels required to fit the lighting model; falls back to direct fill otherwise |

**Output:** For each date, writes `<stem>_cloudfree.bin` (ENVI) and `<date>_cloudfree.png` to `output_dir`.

## MASK — Cloud Mask Training and Inference

Trains a GPU-accelerated Random Forest (cuML) to predict cloud probability from image band values, using the existing cloud mask (e.g., Sen2Cor) as ground truth. Includes per-date lighting normalization against the clearest reference date.

```python
from datetime import datetime
from s2lookback.mask import MASK

masker = MASK(
    image_dir='/data/L1C/resampled_20m',
    mask_dir='/data/cloud_20m',
    output_dir='/data/output/cloud',
    sample_size=5_000,
    progressive_testing=True,
    start=datetime(2025, 8, 20),
    end=datetime(2025, 9, 10),
)

masker.fit()       # train
masker.transform() # predict and save
```

**Additional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `None` | Path to a pre-trained `.joblib` model; skips training if provided |
| `progressive_testing` | `True` | Refit the classifier after each date's samples are added |
| `prediction_threshold` | `0.5` | Probability threshold for classification report |
| `lighting_ref_date` | `None` | Date to use as the lighting reference; auto-selected (clearest date) if `None` |
| `n_feature` | `3` | Number of bands used as features |

**Workflow:**
1. `fit()` — Iterates dates, samples masked/non-masked pixels, fits a Random Forest, saves the model to `output_dir/models/`.
2. `transform()` → `mask_and_save()` — Runs the loaded model over every date, saves predicted probability maps as ENVI `.bin` and a side-by-side comparison PNG.

## MASK_TO_NODATA — Mask Pixels to NaN

Simpler utility: reads each image and its mask, sets all masked pixels to NaN, and saves the result. No machine learning involved.

```python
from s2lookback.mask2nan import MASK_TO_NODATA

cleaner = MASK_TO_NODATA(
    image_dir='/data/L1C',
    mask_dir=['/data/cloud_masks', '/data/shadow_masks', '/data/nodata_masks'],
    output_dir='/data/output/cleaned',
    n_workers=12,
    mask_threshold=0.0001,
)

cleaner.run()
```

**Output:** For each date, writes `<stem>_cloudfree.bin` and a before/after PNG.

## ABCD_MASK — Per-Date A:B::C:D Cloud Probability

Trains a separate Random Forest regression model for every date independently using the scheme: A (image bands) → B (cloud probability). The same image is then used for inference (C=A), producing D (predicted cloud probability).

```python
from s2lookback.abcd_mask import ABCD_MASK

masker = ABCD_MASK(
    image_dir='/data/L1C',
    mask_dir='/data/cloud',
    output_dir='/data/output/abcd',
    skip_f=10_000,
    save_model=False,
)

masker.mask()  # or masker.transform()
```

**Additional parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skip_f` | `5000` | Spatial sampling stride — every `skip_f`-th pixel is used for training |
| `offset` | `0` | Sampling offset |
| `save_model` | `True` | Cache each date's trained model as a `.pkl` in `output_dir/models/` |

**Output:** For each date, writes a single-band float32 ENVI `.bin` and a side-by-side comparison PNG (Sen2Cor vs predicted).

## File Discovery

`utils.get_ordered_file_dict` handles all file matching logic:

- Reads acquisition timestamps from raster metadata (ISO 8601 UTC, via `fire_mapping.raster.Raster`).
- Accepts single paths or lists for both `image_dir` and `mask_dir` — when lists are given, only dates present in **all** directories are kept.
- Single-directory inputs return plain string paths; multi-directory inputs return lists.

## Dependencies

- Python >= 3.10
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `joblib`
- `fire_mapping` (repository raster utilities)
- `cuml` (GPU Random Forest, required by `MASK` only)
- `gdal` (required by `ABCD_MASK` only)
