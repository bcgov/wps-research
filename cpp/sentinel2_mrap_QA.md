# MRAP Quality Control Program

Quality control tool for validating MRAP (Most Recent Available Pixel) products from Sentinel-2 cloud-free compositing.

## Requirements

- GDAL library (libgdal-dev)
- C++17 compiler

## Build

```bash
# Install GDAL if needed
sudo apt-get install libgdal-dev

# Build
make
```

Or compile directly:

```bash
g++ -O3 sentinel2_mrap_QA.cpp -o sentinel2_mrap_QA $(gdal-config --cflags --libs)
```

## Usage

```bash
# Process all L2_* folders in current directory
./sentinel2_mrap_QA

# Process a specific folder
./sentinel2_mrap_QA L2_T09UYU
```

## QA Checks

The program performs two validation checks on each MRAP product:

### 1. NaN Consistency (per file)

Verifies that for each pixel, if any band contains NaN, all bands contain NaN. This ensures the cloud masking was applied consistently across all spectral bands.

### 2. Temporal Consistency (across files)

Verifies that pixels which have valid (non-NaN) data never revert to NaN in subsequent MRAP products. Since MRAP compositing only updates pixels when new valid data is available, the valid pixel count should monotonically increase over time.

## Output

The program outputs:
- Per-file statistics (dimensions, valid/NaN pixel counts)
- Error messages for any QA failures
- Summary of total errors across all folders

Exit codes:
- 0: All QA checks passed
- 1: QA failures detected (or no folders found)

## Example Output

```
MRAP Quality Control Program
============================
Found 2 L2_* folder(s)

=== Processing: ./L2_09UYU ===
Found 5 MRAP files

[1/5] 20251001T192611
  File: ./L2_T09UYU/S2C_MSIL2A_20251001T192611_..._MRAP.bin
  Dimensions: 10980 x 10980 x 10 bands
  NaN consistency: OK
  Valid pixels: 89234567 (74.1%)
  NaN pixels: 31228033 (25.9%)

[2/5] 20251103T192611
  ...
  Temporal consistency: OK (+1234567 newly valid pixels)

============================================================
SUMMARY
============================================================
Folders processed: 2
Folders passed:    2
Folders failed:    0

Total NaN consistency errors: 0
Total NaN reversion errors:   0

*** ALL QA CHECKS PASSED ***
```



