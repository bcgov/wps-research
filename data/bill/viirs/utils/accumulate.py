#!/usr/bin/env python3
"""
viirs/utils/accumulate.py
=========================
Standalone tool to accumulate VIIRS VNP14IMG fire pixel shapefiles
into a series of cumulative output shapefiles with age tracking.

For a date range [start, end], the tool produces one shapefile per unique
detection date.  Each file contains ALL detections from the first file
up to (and including) that date.

Output filenames use the ACTUAL detection datetimes from the source files:
    VIIRS_VNP14IMG_{first_detection_dt}_{last_detection_dt_in_this_batch}.shp

The first file has identical 3rd and 4th fields (start == end).

Usage (CLI):
    python accumulate_fp.py ./shapefiles 20250401 20250930 -r sentinel2.bin -o ./output

Usage (as a function):
    from accumulate import accumulate
    paths = accumulate(
        shp_dir="/path/to/shapefiles",
        start_str="20250401",
        end_str="20250930",
        reference_raster="/path/to/sentinel2.bin",
        output_dir="./accumulated",
    )
"""

import os
import re
import sys
import glob
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from collections import OrderedDict

import geopandas as gpd
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATETIME_PATTERN = re.compile(r"(\d{8}T\d{4}|\d{8})")

FMT_WITH_TIME = "%Y%m%dT%H%M"
FMT_DATE_ONLY = "%Y%m%d"

FALLBACK_CRS = "EPSG:4326"

_DBF_MAX_COL = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_datetime(s: str) -> datetime:
    """Parse YYYYMMDDTHHMM or YYYYMMDD."""
    s = s.strip()
    if "T" in s:
        return datetime.strptime(s, FMT_WITH_TIME)
    return datetime.strptime(s, FMT_DATE_ONLY)


def extract_datetime_from_filename(stem: str) -> Optional[datetime]:
    """
    Extract datetime from the 3rd underscore-delimited field of *stem*.
    Falls back to a full-stem scan if the 3rd field doesn't match.
    """
    fields = stem.split("_")
    if len(fields) >= 3:
        match = DATETIME_PATTERN.fullmatch(fields[2])
        if match:
            return parse_datetime(match.group(1))
    match = DATETIME_PATTERN.search(stem)
    if match:
        return parse_datetime(match.group(1))
    return None


def get_crs_from_raster(raster_path: str) -> str:
    """Read CRS (WKT) from a GDAL-readable raster."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {raster_path}")
    crs = ds.GetProjection()
    if not crs:
        raise RuntimeError(f"Raster has no projection metadata: {raster_path}")
    ds = None
    return crs


def _crs_label(common_crs) -> str:
    """Return a human-readable CRS label."""
    try:
        from pyproj import CRS as ProjCRS
        _crs = ProjCRS.from_user_input(common_crs)
        epsg = _crs.to_epsg()
        name = _crs.name
        if epsg:
            return f"EPSG:{epsg} ({name})"
        return name or str(common_crs)[:80]
    except Exception:
        import re as _re
        m = _re.search(r'AUTHORITY\["EPSG","(\d+)"\]', str(common_crs))
        if m:
            return f"EPSG:{m.group(1)}"
    return str(common_crs)[:80]


def _shorten_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rename columns exceeding the DBF 10-char limit."""
    rename = {}
    for c in gdf.columns:
        if len(c) > _DBF_MAX_COL and c != "geometry":
            rename[c] = c[:_DBF_MAX_COL]
    if rename:
        gdf = gdf.rename(columns=rename)
    return gdf


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def accumulate(
    shp_dir: str,
    start_str: str,
    end_str: str,
    reference_raster: Optional[str] = None,
    target_crs: Optional[str] = None,
    output_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """
    Scan *shp_dir* for VIIRS fire shapefiles, filter by [start, end],
    sort explicitly by detection datetime, and write one cumulative
    shapefile per unique detection date into *output_dir*.

    Output filenames use actual detection datetimes from source files:
        VIIRS_VNP14IMG_{first_dt}_{last_dt_in_batch}.shp
    The first output file has identical start and end timestamps.

    Returns a list of output file paths.
    """

    def _log(msg):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    # ---- 1. Determine target CRS ----
    if target_crs is not None:
        common_crs = target_crs
    elif reference_raster is not None:
        if not os.path.exists(reference_raster):
            raise FileNotFoundError(
                f"Reference raster not found: {reference_raster}")
        common_crs = get_crs_from_raster(reference_raster)
    else:
        common_crs = FALLBACK_CRS

    label = _crs_label(common_crs)
    _log(f"[INFO] Target CRS: {label}")

    # ---- 2. Parse date range ----
    start_dt = parse_datetime(start_str)
    end_dt = parse_datetime(end_str)
    _log(f"[INFO] Range: {start_dt} \u2192 {end_dt}")

    # ---- 3. Find shapefiles ----
    shp_files = glob.glob(
        os.path.join(shp_dir, "**", "*.shp"), recursive=True)
    _log(f"[INFO] Found {len(shp_files)} .shp files total")
    if not shp_files:
        _log("[WARN] No shapefiles found.")
        return []

    # ---- 4. Parse datetimes, filter, and SORT ----
    matched: Dict[datetime, str] = {}
    skipped = 0
    for fpath in shp_files:
        stem = Path(fpath).stem
        dt = extract_datetime_from_filename(stem)
        if dt is None:
            skipped += 1
            continue
        if start_dt <= dt <= end_dt:
            matched[dt] = fpath

    # Explicit chronological sort
    sorted_items: List[Tuple[datetime, str]] = sorted(
        matched.items(), key=lambda kv: kv[0])

    _log(f"[INFO] {len(sorted_items)} files in range "
         f"({skipped} unparseable, "
         f"{len(shp_files) - len(sorted_items) - skipped} outside range)")

    if not sorted_items:
        _log("[WARN] No files in requested range.")
        return []

    # ---- 5. Prepare output directory ----
    if output_dir is None:
        output_dir = os.path.join(shp_dir, "accumulated")
    os.makedirs(output_dir, exist_ok=True)

    # ---- 6. Fixed start datetime = first file's actual detection dt ----
    fixed_start_dt = sorted_items[0][0]
    _log(f"[INFO] First detection: {fixed_start_dt.strftime(FMT_WITH_TIME)}")

    # ---- 7. Group by calendar date, load incrementally ----
    date_groups: OrderedDict[date, List[Tuple[datetime, str]]] = OrderedDict()
    for dt, fpath in sorted_items:
        d = dt.date()
        date_groups.setdefault(d, []).append((dt, fpath))

    unique_dates = list(date_groups.keys())  # already sorted
    total_dates = len(unique_dates)

    running_frames: List[gpd.GeoDataFrame] = []
    written_paths: List[str] = []

    for date_idx, cur_date in enumerate(unique_dates):
        group = date_groups[cur_date]
        # Sort within the day by time
        group_sorted = sorted(group, key=lambda kv: kv[0])

        # Load this date's files
        day_frames = []
        for dt, fpath in group_sorted:
            try:
                gdf = gpd.read_file(fpath)
            except Exception as exc:
                print(f"  [WARN] Could not read {fpath}: {exc}")
                continue

            # Reproject if needed
            if gdf.crs is not None and gdf.crs != common_crs:
                gdf = gdf.to_crs(common_crs)
                for xc, yc in [("x", "y"), ("easting", "northing"),
                                ("utm_x", "utm_y")]:
                    if xc in gdf.columns and yc in gdf.columns:
                        gdf[xc] = gdf.geometry.x
                        gdf[yc] = gdf.geometry.y

            gdf["detection_datetime"] = dt
            gdf["source_file"] = os.path.basename(fpath)
            day_frames.append(gdf)

        if not day_frames:
            continue

        day_gdf = gpd.GeoDataFrame(pd.concat(day_frames, ignore_index=True))
        running_frames.append(day_gdf)

        # The end datetime for the filename:
        #   - First batch: must equal fixed_start_dt exactly (start == end)
        #   - Later batches: last file's actual detection datetime in this batch
        if date_idx == 0:
            batch_end_dt = fixed_start_dt
        else:
            batch_end_dt = group_sorted[-1][0]

        # Build cumulative GeoDataFrame
        combined = gpd.GeoDataFrame(
            pd.concat(running_frames, ignore_index=True))

        # Compute age_days relative to the batch end datetime
        combined["age_days"] = combined["detection_datetime"].apply(
            lambda det: round((batch_end_dt - det).total_seconds() / 86400.0, 2))

        # Sort combined by detection_datetime
        combined.sort_values("detection_datetime", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        # Prepare for shapefile write
        out = combined.copy()
        out["det_dt"] = out["detection_datetime"].apply(
            lambda d: d.strftime("%Y-%m-%d %H:%M"))
        out.drop(columns=["detection_datetime"], inplace=True)
        out = _shorten_columns(out)

        # Build filename from actual detection datetimes
        start_tag = fixed_start_dt.strftime(FMT_WITH_TIME)
        end_tag = batch_end_dt.strftime(FMT_WITH_TIME)
        stem = f"VIIRS_VNP14IMG_{start_tag}_{end_tag}"
        out_path = os.path.join(output_dir, f"{stem}.shp")
        out.to_file(out_path)
        written_paths.append(out_path)

        _log(f"  [{date_idx + 1}/{total_dates}] {cur_date}  "
             f"({len(combined)} features) \u2192 {os.path.basename(out_path)}")

    _log(f"[INFO] Wrote {len(written_paths)} cumulative shapefiles "
         f"to {output_dir}")

    return written_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Accumulate VIIRS fire pixel shapefiles "
                    "(one cumulative file per detection date).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python accumulate.py ./shapefiles 20250401 20250930 -r sentinel2.bin -o ./output
  python accumulate.py ./shapefiles 20250401 20250930 --crs EPSG:3005 -o ./output
  python accumulate.py ./shapefiles 20250401 20250930 -o ./output
        """,
    )
    parser.add_argument("shp_dir",
                        help="Directory to scan for .shp files.")
    parser.add_argument("start",
                        help="Start datetime: YYYYMMDD or YYYYMMDDTHHMM.")
    parser.add_argument("end",
                        help="End datetime: YYYYMMDD or YYYYMMDDTHHMM.")
    parser.add_argument("-r", "--reference", default=None,
                        help="Reference raster for target CRS.")
    parser.add_argument("--crs", default=None,
                        help="Explicit target CRS (e.g. EPSG:3005).")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: <shp_dir>/accumulated).")

    args = parser.parse_args()
    accumulate(
        shp_dir=args.shp_dir,
        start_str=args.start,
        end_str=args.end,
        reference_raster=args.reference,
        target_crs=args.crs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()