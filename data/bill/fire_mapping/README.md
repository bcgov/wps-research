# Fire Mapping Toolkit

A set of tools for Sentinel-2 based burn area analysis — from raw ENVI rasters to classified burn severity maps. Covers spectral indices (NBR/dNBR), per-pixel change detection, unsupervised ML-based burn mapping, and batch preprocessing utilities.

<!-- IMAGE: insert fire mapping result screenshot here -->

---

## File Overview

| File | Purpose |
|------|---------|
| `raster.py` | Core `Raster` class wrapping GDAL. Reads ENVI `.bin` files, extracts bands by number or name, exposes georeference info and acquisition timestamp, and validates spatial compatibility between two rasters. |
| `barc.py` | Computes NBR (Normalized Burn Ratio) from Band 8 + Band 12, and dNBR between pre/post-fire dates. Scales result to BARC 256 and plots a 4-class severity map (Unburned / Low / Medium / High) with area percentages. |
| `change_detection.py` | Computes a normalized per-pixel difference `(post − pre) / (post + pre)` across all bands between two dates and visualizes it side-by-side with the pre and post imagery. |
| `fire_mapping.py` | Interactive GUI for unsupervised burn area mapping. Uses T-SNE + Random Forest + HDBSCAN to cluster pixels into burned/unburned. Accepts a burn hint (polygon file, SWIR-wins rule, or dNBR threshold) and saves the final classification as an ENVI file. |
| `dominant_band.py` | Identifies pixels where a chosen band has the highest value compared to all others. Used as a fast "SWIR wins" heuristic to approximate burned areas without a reference mask. |
| `sampling.py` | Pixel sampling utilities: random row sampling (with optional NaN filtering), in/out-polygon sampling (preserving true population proportions), and unconstrained regular sampling. Used internally by `fire_mapping.py`. |
| `polygon.py` | Splits a raster into two sets of pixels — inside and outside a polygon mask — returning both data arrays and their flat indices in the original image. |
| `photos.py` | Batch-converts `.bin` ENVI rasters in a directory to `.png` files using global percentile normalization. Runs conversions in parallel with `ThreadPoolExecutor`. |
| `plot_tools.py` | Lightweight matplotlib helpers: `plot()` for a single image (auto grayscale for 2D data), and `plot_multiple()` for a configurable grid of images. |
| `extract_temp.py` | Batch driver that processes all `.SAFE` folders in parallel, extracting and resampling selected bands (B08, B09, B11, B12) to 20 m resolution via `SAFE.extract_and_resample_L1`. |

---

## Quick Usage

### Plot a raster
```bash
python3 -m fire_mapping.raster path/to/file.bin
python3 -m fire_mapping.raster path/to/file.bin --band_list=1,2,3
python3 -m fire_mapping.raster path/to/file.bin --mask_file=polygon.bin
```

### Compute NBR / BARC burn severity
```bash
# NBR for a single date
python3 -m fire_mapping.barc file.bin

# dNBR + BARC 256 severity map between two dates
python3 -m fire_mapping.barc pre.bin post.bin
```

### Visualise change between two dates
```bash
python3 -m fire_mapping.change_detection pre.bin post.bin
```

### Interactive burn mapping GUI
```bash
# With an automatic hint (no polygon needed)
python3 -m fire_mapping.fire_mapping image.bin

# With a rasterized polygon as the hint
python3 -m fire_mapping.fire_mapping image.bin polygon.bin
```

### Band dominance (quick burned-area heuristic)
```bash
python3 -m fire_mapping.dominant_band file.bin
```

### Batch export PNGs
```bash
python3 -m fire_mapping.photos   # runs on current directory
```

### Extract & resample SAFE folders
```bash
python3 -m fire_mapping.extract_temp
```

---

## Python API

```python
from fire_mapping.raster import Raster
from fire_mapping.barc import NBR, dNBR, plot_barc
from fire_mapping.change_detection import change_detection
from fire_mapping.sampling import regular_sampling, in_out_sampling
from fire_mapping.polygon import split_in_out
from fire_mapping.dominant_band import dominant_band

# Load a raster and read specific bands
raster = Raster("image.bin")
rgb = raster.read_bands([1, 2, 3])       # (H, W, 3) float32
nir = raster.get_band(8)                 # (H, W) by Sentinel-2 band number
print(raster.acquisition_time)

# Compute burn severity
nbr_pre, nbr_post, dnbr = dNBR(raster_pre=pre, raster_post=post)
plot_barc(dnbr, start_date="2024-07-01", end_date="2024-08-15")

# Sample pixels respecting in/out polygon ratio
indices, samples, ratio = in_out_sampling(
    raster_filename="image.bin",
    polygon_filename="polygon.bin",
    in_sample_size=5000
)
```

---

## Notes

- All raster files are expected in **ENVI format** (`.bin` + `.hdr`).
- Band numbers follow Sentinel-2 convention (e.g. Band 8 = NIR, Band 12 = SWIR).
- `fire_mapping.py` requires an **NVIDIA GPU** for rendering and depends on `machine_learning.dim_reduce` (T-SNE) and `machine_learning.cluster` (HDBSCAN).
- Use `raster.can_match_with(other)` to verify two rasters share the same grid and projection before combining them.
