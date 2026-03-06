#!/usr/bin/env python3
"""
viirs/utils/accumulate_fp.py
=========================
Standalone tool to accumulate VIIRS VNP14IMG fire pixel shapefiles
into a single output shapefile with age tracking.

Usage (CLI):
    # Basic — reprojects everything to EPSG:4326 (WGS84 lat/lon)
    python accumulate_fp.py /path/to/shapefiles 20250401 20250930

    # With reference raster — reprojects to match the raster's CRS
    python accumulate_fp.py /path/to/shapefiles 20250401 20250930 -r sentinel2.bin

    # Custom output path
    python accumulate_fp.py /path/to/shapefiles 20250401 20250930 -r sentinel2.bin -o output.shp

Usage (as a function):
    from accumulate_fp import accumulate
    gdf = accumulate(
        shp_dir="/path/to/shapefiles",
        start_str="20250401",
        end_str="20250930",
        reference_raster="/path/to/sentinel2.bin",
        output_path="my_output.shp",
    )
"""

import os
import re
import sys
import glob
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex to extract the datetime token from the 3rd underscore-delimited field.
# Matches YYYYMMDD or YYYYMMDDTHHMM anywhere in the filename.
DATETIME_PATTERN = re.compile(r"(\d{8}T\d{4}|\d{8})")

# For parsing
FMT_WITH_TIME = "%Y%m%dT%H%M"
FMT_DATE_ONLY = "%Y%m%d"

# Default CRS when no reference raster is provided
FALLBACK_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_datetime(s: str) -> datetime:
    """
    Parse a datetime string in either YYYYMMDDTHHMM or YYYYMMDD format.
    If no time component, assumes T0000 (start of day).
    """
    s = s.strip()
    if "T" in s:
        return datetime.strptime(s, FMT_WITH_TIME)
    else:
        return datetime.strptime(s, FMT_DATE_ONLY)


def extract_datetime_from_filename(stem: str) -> Optional[datetime]:
    """
    Given a filename stem like 'VIIRS_VNP14IMG_20250902T1920',
    split by '_' and look at the 3rd field (index 2) for a datetime.
    Falls back to scanning the entire stem if the 3rd field doesn't match.
    """
    fields = stem.split("_")

    # Try the 3rd field first (index 2) — the expected location
    if len(fields) >= 3:
        match = DATETIME_PATTERN.fullmatch(fields[2])
        if match:
            return parse_datetime(match.group(1))

    # Fallback: search the whole stem for any datetime token
    match = DATETIME_PATTERN.search(stem)
    if match:
        return parse_datetime(match.group(1))

    return None


def build_output_filename(
    sample_stem: str,
    start_dt: datetime,
    end_dt: datetime,
) -> str:
    """
    Build the output filename from the first two fields of a sample
    shapefile name, plus start and end datetime as 3rd and 4th fields.

    Example:
        Input stem:  VIIRS_VNP14IMG_20250902T1920
        Output:      VIIRS_VNP14IMG_20250401T0000_20250930T2359.shp
    """
    fields = sample_stem.split("_")

    # Keep the first two fields (e.g. VIIRS, VNP14IMG)
    prefix = "_".join(fields[:2]) if len(fields) >= 2 else sample_stem

    start_str = start_dt.strftime(FMT_WITH_TIME)
    end_str = end_dt.strftime(FMT_WITH_TIME)

    return f"{prefix}_{start_str}_{end_str}.shp"


def get_crs_from_raster(raster_path: str) -> str:
    """
    Read the CRS from a raster file (ENVI .bin/.hdr or GeoTIFF) using GDAL.

    Parameters
    ----------
    raster_path : str
        Path to the raster file (.bin, .hdr, .tif, etc.)

    Returns
    -------
    str — WKT projection string
    """
    from osgeo import gdal
    gdal.UseExceptions()

    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {raster_path}")

    crs = ds.GetProjection()
    if not crs:
        raise RuntimeError(
            f"Raster has no projection metadata: {raster_path}\n"
            f"Make sure the .hdr file contains projection info."
        )

    ds = None  # close
    return crs


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def accumulate(
    shp_dir: str,
    start_str: str,
    end_str: str,
    reference_raster: Optional[str] = None,
    target_crs: Optional[str] = None,
    output_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Scan a directory for VIIRS fire shapefiles, filter by date range,
    reproject to a common CRS, accumulate into a single GeoDataFrame
    with age tracking, and optionally write to a shapefile.

    Parameters
    ----------
    shp_dir : str
        Directory to recursively scan for .shp files.
    start_str : str
        Start datetime as YYYYMMDD or YYYYMMDDTHHMM.
    end_str : str
        End datetime as YYYYMMDD or YYYYMMDDTHHMM.
    reference_raster : str, optional
        Path to a raster file whose CRS will be used as the target
        projection. This is the recommended approach — ensures the
        accumulated output matches your analysis raster exactly.
    target_crs : str, optional
        Explicit CRS string (e.g. "EPSG:3005"). Overrides reference_raster
        if both are provided. If neither is given, falls back to EPSG:4326.
    output_path : str, optional
        Path to write the output shapefile. If None, an auto-generated
        name is used in shp_dir.

    Returns
    -------
    gpd.GeoDataFrame — the accumulated fire pixels with age_days column.
    """

    # ---- Step 1: Determine target CRS ----
    if target_crs is not None:
        common_crs = target_crs
    elif reference_raster is not None:
        if not os.path.exists(reference_raster):
            raise FileNotFoundError(f"Reference raster not found: {reference_raster}")
        common_crs = get_crs_from_raster(reference_raster)
    else:
        common_crs = FALLBACK_CRS

    # Build a human-readable label like "EPSG:3005 (NAD83 / BC Albers)"
    crs_label = str(common_crs)
    try:
        from pyproj import CRS as ProjCRS
        _crs = ProjCRS.from_user_input(common_crs)
        epsg = _crs.to_epsg()
        name = _crs.name
        if epsg:
            crs_label = f"EPSG:{epsg} ({name})"
        else:
            crs_label = name or str(common_crs)[:80]
    except Exception:
        # pyproj not available or CRS unrecognised — try regex fallback
        import re as _re
        epsg_match = _re.search(r'AUTHORITY\["EPSG","(\d+)"\]', str(common_crs))
        if epsg_match:
            crs_label = f"EPSG:{epsg_match.group(1)}"

    if target_crs is not None:
        print(f"[INFO] Target CRS (explicit): {crs_label}")
    elif reference_raster is not None:
        print(f"[INFO] Target CRS (from {os.path.basename(reference_raster)}): {crs_label}")
    else:
        print(f"[INFO] No reference raster provided, using fallback: {crs_label}")

    # ---- Step 2: Parse the requested date range ----
    start_dt = parse_datetime(start_str)
    end_dt = parse_datetime(end_str)

    print(f"[INFO] Requested range: {start_dt} → {end_dt}")
    print(f"[INFO] Scanning directory: {shp_dir}")

    # ---- Step 3: Find all shapefiles recursively ----
    shp_files = glob.glob(os.path.join(shp_dir, "**", "*.shp"), recursive=True)
    print(f"[INFO] Found {len(shp_files)} .shp files total")

    if not shp_files:
        print("[WARN] No shapefiles found. Exiting.")
        return gpd.GeoDataFrame()

    # ---- Step 4: Parse datetime from each filename, filter by range ----
    matched: Dict[datetime, str] = {}
    skipped = 0

    for fpath in shp_files:
        stem = Path(fpath).stem
        dt = extract_datetime_from_filename(stem)

        if dt is None:
            skipped += 1
            print(f"  [SKIP] No datetime found in: {stem}")
            continue

        if start_dt <= dt <= end_dt:
            matched[dt] = fpath

    # Sort chronologically
    matched = dict(sorted(matched.items()))

    print(f"[INFO] {len(matched)} files within date range "
          f"({skipped} skipped, {len(shp_files) - len(matched) - skipped} outside range)")

    if not matched:
        print("[WARN] No files in the requested date range. Exiting.")
        return gpd.GeoDataFrame()

    # ---- Step 5: Load, reproject, and concatenate ----
    frames: List[gpd.GeoDataFrame] = []

    for dt, fpath in matched.items():
        try:
            gdf = gpd.read_file(fpath)
        except Exception as exc:
            print(f"  [WARN] Could not read {fpath}: {exc}")
            continue

        # Reproject to common CRS
        if gdf.crs is not None and gdf.crs != common_crs:
            src_crs = gdf.crs
            gdf = gdf.to_crs(common_crs)
            # Update UTM columns if they exist (they're now in the new CRS)
            if "x" in gdf.columns and "y" in gdf.columns:
                gdf["x"] = gdf.geometry.x
                gdf["y"] = gdf.geometry.y
            elif "easting" in gdf.columns and "northing" in gdf.columns:
                gdf["easting"] = gdf.geometry.x
                gdf["northing"] = gdf.geometry.y
            elif "utm_x" in gdf.columns and "utm_y" in gdf.columns:
                gdf["utm_x"] = gdf.geometry.x
                gdf["utm_y"] = gdf.geometry.y
            print(f"  Loaded {len(gdf):>6} features from {os.path.basename(fpath)}  "
                  f"(reprojected {src_crs} → {crs_label})")
        else:
            print(f"  Loaded {len(gdf):>6} features from {os.path.basename(fpath)}")

        gdf["detection_datetime"] = dt
        gdf["source_file"] = os.path.basename(fpath)
        frames.append(gdf)

    if not frames:
        print("[WARN] All files failed to load. Exiting.")
        return gpd.GeoDataFrame()

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

    # ---- Step 6: Compute age_days as decimal days ----
    combined["age_days"] = combined["detection_datetime"].apply(
        lambda det: round((end_dt - det).total_seconds() / 86400.0, 2)
    )

    # ---- Step 7: Build output path if not provided ----
    if output_path is None:
        sample_stem = Path(list(matched.values())[0]).stem
        out_name = build_output_filename(sample_stem, start_dt, end_dt)
        output_path = os.path.join(shp_dir, out_name)

    # ---- Step 8: Prepare columns for shapefile ----
    combined["det_dt"] = combined["detection_datetime"].apply(
        lambda dt: dt.strftime("%Y-%m-%d %H:%M")
    )
    combined.drop(columns=["detection_datetime"], inplace=True)

    # Rename columns that exceed 10 chars (shapefile DBF limit)
    shp_rename = {
        "source_file": "src_file",
        "confidence": "confidence",  # 10 chars, fine
    }
    for c in combined.columns:
        if len(c) > 10 and c not in shp_rename:
            shp_rename[c] = c[:10]

    rename_map = {k: v for k, v in shp_rename.items()
                  if k in combined.columns and k != v}
    if rename_map:
        print(f"[INFO] Shortened column names for shapefile: {rename_map}")
        combined.rename(columns=rename_map, inplace=True)

    # ---- Step 9: Write output ----
    combined.to_file(output_path)

    # ---- Step 10: Print summary ----
    youngest = combined["age_days"].min()
    oldest = combined["age_days"].max()
    mean_age = combined["age_days"].mean()

    print()
    print("=" * 60)
    print("  ACCUMULATION SUMMARY")
    print("=" * 60)
    print(f"  Total features:    {len(combined)}")
    print(f"  Source files:      {len(matched)}")
    print(f"  Date range:        {start_dt.strftime('%Y-%m-%d %H:%M')} → "
          f"{end_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Target CRS:        {crs_label}")
    print(f"  Youngest pixel:    {youngest:.2f} days")
    print(f"  Oldest pixel:      {oldest:.2f} days")
    print(f"  Mean age:          {mean_age:.2f} days")
    print(f"  Output written to: {output_path}")
    print("=" * 60)

    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Accumulate VIIRS fire pixel shapefiles into one file with age tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With reference raster (recommended — matches your analysis CRS)
  python accumulate_fp.py ./shapefiles 20250401 20250930 -r sentinel2.bin

  # With explicit CRS
  python accumulate_fp.py ./shapefiles 20250401 20250930 --crs EPSG:3005

  # No reference — falls back to EPSG:4326 (WGS84 lat/lon)
  python accumulate_fp.py ./shapefiles 20250401 20250930

  # Custom output path
  python accumulate_fp.py ./shapefiles 20250401 20250930 -r sentinel2.bin -o accumulated.shp
        """,
    )

    parser.add_argument(
        "shp_dir",
        help="Directory to recursively scan for .shp files.",
    )
    parser.add_argument(
        "start",
        help="Start datetime: YYYYMMDD or YYYYMMDDTHHMM.",
    )
    parser.add_argument(
        "end",
        help="End datetime: YYYYMMDD or YYYYMMDDTHHMM.",
    )
    parser.add_argument(
        "-r", "--reference",
        default=None,
        help="Reference raster file (.bin, .hdr, .tif). "
             "All shapefiles will be reprojected to match its CRS. "
             "This is the recommended approach.",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="Explicit target CRS (e.g. EPSG:3005). "
             "Overrides --reference if both are given.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output shapefile path. Default: auto-generated in shp_dir.",
    )

    args = parser.parse_args()

    accumulate(
        shp_dir=args.shp_dir,
        start_str=args.start,
        end_str=args.end,
        reference_raster=args.reference,
        target_crs=args.crs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()