# SAFE Package

The **SAFE** package provides utilities for handling Sentinel‑2 SAFE archives (L1C and L2A).  It exposes a small public API that lets you

* extract a stack of raster bands in ENVI (BSQ) format,
* optionally resample L1 bands to a common resolution, and
* create cloud‑probability or scene‑classification masks.

## Public API

```python
from SAFE import (
    extract_L1,            # L1C -> ENVI
    extract_L2,            # L2A -> ENVI
    extract_and_resample_L1,  # L1C with resampling
    extract_cloud_single,    # single‑band cloud‑probability
    extract_cloud_zip_root,   # cloud from a zip tree
    extract_scl_mask,          # binary SCL mask
)
```

The package is intentionally lightweight; all heavy I/O is performed with **GDAL** and NumPy.  No side‑effects or external network calls are made.

## Example Usage

```python
# L1C band stack (default: all bands, 20 m resolution)
out = extract_L1("/data/SAFE/L1C_SAFEDIR")

# L2A band stack with optional cloud probability at 20 m
out = extract_L2("/data/SAFE/L2A_SAFEDIR", cloud_prob=True, resolution=20)

# L1C bands resampled to 10 m and stacked
out = extract_and_resample_L1("/data/SAFE/L1C_SAFEDIR", target_resolution=10)

# Extract a binary mask where SCL values 4 (cloud) or 5 (cloud shadow) are present
extract_scl_mask("/data/SAFE/L2A_SCL_DIR", values=[4,5], out_dir="/data/masks")
```

## Module Layout

| Module | Responsibility |
|--------|----------------|
| **extract_L1.py** | Read L1C bands, write ENVI BSQ stack.
| **extract_L2.py** | Read L2A bands (and optional cloud‑probability), write ENVI BSQ stack.
| **resample_L1.py** | Resample all L1C bands to a user‑supplied resolution before stacking.
| **scl_mask.py** | Generate binary masks from existing SCL ENVI files.
| **cloud_L2.py** | Convenience wrappers for cloud‑probability extraction.
| **__init__.py** | Re‑exports the public functions.

## Implementation Notes

* All functions return the path to the created ENVI file.
* GDAL is used in memory‑friendly mode (`MEM`) for intermediate steps.
* The code follows the Sentinel‑2 data model defined in the `info.py` module.
* Thread‑pooling is employed in `scl_mask.py` to speed up mask extraction.

## Dependencies

* `osgeo` (GDAL)
* `numpy`

Both packages are required for the raster I/O and manipulation.

---

Feel free to extend the API or add new helper functions in the `SAFE/` directory. The `__all__` list in `__init__.py` controls what is publicly available.
