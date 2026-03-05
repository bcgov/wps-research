#!/usr/bin/env python3
"""
viirs/utils/accumulate_fp.py
=========================
Standalone tool to accumulate VIIRS VNP14IMG fire pixel shapefiles
into a single output shapefile with age tracking.

Usage (CLI):
    python accumulate_fire_pixels.py /path/to/shapefiles 20250401T0000 20250930T2359
    python accumulate_fire_pixels.py /path/to/shapefiles 20250401 20250930
    python accumulate_fire_pixels.py /path/to/shapefiles 20250401 20250930 -o output.shp

Usage (as a function):
    from accumulate_fire_pixels import accumulate
    gdf = accumulate(
        shp_dir="/path/to/shapefiles",
        start_str="20250401",
        end_str="20250930",
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


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def accumulate(
    shp_dir: str,
    start_str: str,
    end_str: str,
    output_path: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Scan a directory for VIIRS fire shapefiles, filter by date range,
    accumulate into a single GeoDataFrame with age tracking, and
    optionally write to a shapefile.

    Parameters
    ----------
    shp_dir : str
        Directory to recursively scan for .shp files.
    start_str : str
        Start datetime as YYYYMMDD or YYYYMMDDTHHMM.
    end_str : str
        End datetime as YYYYMMDD or YYYYMMDDTHHMM.
    output_path : str, optional
        Path to write the output shapefile. If None, an auto-generated
        name is used in shp_dir.

    Returns
    -------
    gpd.GeoDataFrame — the accumulated fire pixels with age_days column.
    """

    # ---- Step 1: Parse the requested date range ----
    start_dt = parse_datetime(start_str)
    end_dt = parse_datetime(end_str)

    print(f"[INFO] Requested range: {start_dt} → {end_dt}")
    print(f"[INFO] Scanning directory: {shp_dir}")

    # ---- Step 2: Find all shapefiles recursively ----
    shp_files = glob.glob(os.path.join(shp_dir, "**", "*.shp"), recursive=True)
    print(f"[INFO] Found {len(shp_files)} .shp files total")

    if not shp_files:
        print("[WARN] No shapefiles found. Exiting.")
        return gpd.GeoDataFrame()

    # ---- Step 3: Parse datetime from each filename, filter by range ----
    #   We only keep files whose detection datetime falls within
    #   [start_dt, end_dt].  Files without a parseable datetime are skipped.
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

    # ---- Step 4: Load and concatenate all matching shapefiles ----
    #   Each file gets a 'detection_datetime' column stamped from its filename.
    frames: List[gpd.GeoDataFrame] = []

    for dt, fpath in matched.items():
        try:
            gdf = gpd.read_file(fpath)
        except Exception as exc:
            print(f"  [WARN] Could not read {fpath}: {exc}")
            continue

        gdf["detection_datetime"] = dt
        gdf["source_file"] = os.path.basename(fpath)
        frames.append(gdf)
        print(f"  Loaded {len(gdf):>6} features from {os.path.basename(fpath)}")

    if not frames:
        print("[WARN] All files failed to load. Exiting.")
        return gpd.GeoDataFrame()

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

    # ---- Step 5: Compute age_days as decimal days ----
    #   age = (requested end_dt − detection_datetime) in fractional days.
    #   Rounded to 2 decimal places.  E.g. 1 day = 1.00, 1 day 1.5h = 1.06
    combined["age_days"] = combined["detection_datetime"].apply(
        lambda det: round((end_dt - det).total_seconds() / 86400.0, 2)
    )

    # ---- Step 6: Build output path if not provided ----
    if output_path is None:
        sample_stem = Path(list(matched.values())[0]).stem
        out_name = build_output_filename(sample_stem, start_dt, end_dt)
        output_path = os.path.join(shp_dir, out_name)

    # ---- Step 7: Prepare columns for shapefile ----
    #   Shapefile DBF format has a 10-character column name limit.
    #   Convert datetime to string and shorten long column names.
    combined["det_dt"] = combined["detection_datetime"].apply(
        lambda dt: dt.strftime("%Y-%m-%d %H:%M")
    )
    combined.drop(columns=["detection_datetime"], inplace=True)

    # Rename columns that exceed 10 chars
    shp_rename = {
        "source_file": "src_file",
        "confidence": "confidence",  # 10 chars, fine
    }
    # Auto-shorten anything else over 10 chars
    for c in combined.columns:
        if len(c) > 10 and c not in shp_rename:
            shp_rename[c] = c[:10]

    # Only rename columns that actually exist and need shortening
    rename_map = {k: v for k, v in shp_rename.items()
                  if k in combined.columns and k != v}
    if rename_map:
        print(f"[INFO] Shortened column names for shapefile: {rename_map}")
        combined.rename(columns=rename_map, inplace=True)

    # ---- Step 8: Write output ----
    combined.to_file(output_path)

    # ---- Step 9: Print summary ----
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
  # With time specified
  python accumulate_fire_pixels.py ./shapefiles 20250401T0000 20250930T2359

  # Date only (assumes T0000 for both)
  python accumulate_fire_pixels.py ./shapefiles 20250401 20250930

  # Custom output path
  python accumulate_fire_pixels.py ./shapefiles 20250401 20250930 -o accumulated.shp
        """,
    )

    parser.add_argument(
        "shp_dir",
        help="Directory to recursively scan for .shp files.",
    )
    parser.add_argument(
        "start",
        help="Start datetime: YYYYMMDD or YYYYMMDDTHHMM. "
             "If no time, assumes T0000.",
    )
    parser.add_argument(
        "end",
        help="End datetime: YYYYMMDD or YYYYMMDDTHHMM. "
             "If no time, assumes T0000.",
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
        output_path=args.output,
    )


if __name__ == "__main__":
    main()