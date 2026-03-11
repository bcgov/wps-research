# Sentinel‑2 Look‑Back Module

`python3 -m s2lookback` is a small helper library that allows users to search, read, and sample Sentinel‑2 imagery and associated cloud masks in a date‑range based on the filenames.

## Features

- **Date‑ordered lookup** – Build a dictionary of image and mask files by acquisition date.
- **Mask handling** – Supports binary or probabilistic cloud masks and optional thresholding.
- **Random sampling** – Sample masked and unmasked pixels with user‑controlled ratios.
- **Parallel processing** – Uses the `concurrent.futures` API to process multiple dates concurrently.
- **Extensible** – Built on top of the `fire_mapping.raster.Raster` class, so any other raster utility can be dropped in.

## Installation

The module is part of the `wps‑research` repository. To use it, install the repository dependencies:

```bash
# In the repository root
pip install -r requirements.txt
```

The package itself does not require a separate installation step – simply import it in your Python code:

```python
from s2lookback import LookBack
```

## Basic Usage

```python
from datetime import datetime
from s2lookback import LookBack

# Instantiate with the path to your image directory and optional mask directory
lb = LookBack(
    image_dir="/data/sentinel2/2026-03-01",
    mask_dir="/data/sentinel2/masks",
    start=datetime(2026, 3, 1),
    end=datetime(2026, 3, 7),
    mask_threshold=0.1,
    sample_size=5000,
)

# Build the lookup dictionary
file_dict = lb.get_file_dictionary()
print(f"Found {len(file_dict)} timestamps")

# Read data for a particular date
date = next(iter(file_dict))  # just an example
image = lb.read_image(date, band_list=[0,1,2,3])
mask, coverage = lb.read_mask(date)

# Sample pixels
sampled_pixels = lb.sample_datasets(image, mask)
print(sampled_pixels.shape)
```

## API Reference

| Method | Description |
|--------|-------------|
| `get_file_dictionary()` | Build a dictionary keyed by acquisition `datetime`. Each entry contains `image_path` and optionally `mask_path`. |
| `read_image(date, band_list)` | Return a `numpy.ndarray` with the selected bands for the given date. Supports reading multiple images if `image_path` is a list. |
| `read_mask(date, as_prob=False)` | Return a boolean mask (and coverage) or raw probabilistic mask values. |
| `sample_datasets(img_dat, mask)` | Randomly sample masked and unmasked pixels based on `sample_size`, `sample_between_prop`, and `sample_within_prop`. |
| `get_nodata_mask(img_dat, nodata_val)` | Return a mask where no‑data values are True. |

## Dependencies

- Python ≥ 3.10
- `numpy`
- `fire_mapping` (the repository’s own raster utilities)
- `datetime`
- `concurrent.futures`

## Extending

If you have additional mask formats (e.g., GeoTIFF), copy or modify `s2lookback/mask.py` to read those formats. The core `LookBack` class will then automatically use your new reader.

---

© 2026‑WPS‑Research. All rights reserved.
