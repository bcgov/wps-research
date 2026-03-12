# WPS Research — Remote Sensing Toolkit

A collection of satellite image processing tools developed for wildfire analysis at the BC Wildfire Service, built around Sentinel-2 and VIIRS data.

---

## VIIRS Hotspot Processing

[`viirs/`](https://github.com/bcgov/wps-research/tree/master/data/bill/viirs)

<img src="images/viirs.png" width="60%">

End-to-end pipeline for VIIRS active fire pixel data (VNP14IMG). Downloads raw NetCDF files from NASA LAADS DAAC, reprojects fire detections onto a Sentinel-2 reference grid, and surfaces them in an interactive GUI. Supports multi-date accumulation and rasterization into binary fire masks for burn severity analysis.

---

## Sentinel-2 Fire Mapping

[`fire_mapping/`](https://github.com/bcgov/wps-research/blob/master/data/bill/fire_mapping)

<img src="images/fire_mapping.png" width="60%">

Lightweight toolkit for fire-mapping workflows on Sentinel-2 imagery. Provides band reading, fire perimeter masking, flexible sampling strategies (random, in/out polygon, regular grid), band dominance analysis, and PNG thumbnail generation. Modules are self-contained and designed for direct use in analysis scripts.

---

## Sentinel-2 Cloud Masking

[`cloud_masking/`](https://github.com/bcgov/wps-research/tree/master/data/bill/cloud_masking)

Multi-stage pipeline for per-date cloud masking using Random Forest regression. Trains on image/cloud-probability-mask pairs, predicts cloud probability maps, and feeds results into MRAP compositing. Includes an interactive review step before the full pipeline commits, and exports final outputs as MP4.

---

## s2lookback

[`s2lookback/`](https://github.com/bcgov/wps-research/tree/master/data/bill/s2lookback)

Shared library underpinning the cloud masking and compositing workflows. Handles date-ordered file discovery, image and mask I/O, pixel sampling, and parallel processing. Exposes four ready-to-use classes — `MRAP`, `MASK`, `MASK_TO_NODATA`, and `ABCD_MASK` — each built on a common `LookBack` base.
