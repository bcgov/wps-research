#!/usr/bin/env python
"""
Convert VNP14IMG NetCDF fire pixel data to a Shapefile in UTM projection
matching Sentinel-2 imagery.

Usage:
    # Current directory (finds all .nc recursively)
    python vnp14_to_shp.py --utm-zone 9 --hemisphere N

    # A directory (finds all .nc recursively)
    python vnp14_to_shp.py /data/viirs/2025 --utm-zone 9 --hemisphere N

    # Single file
    python vnp14_to_shp.py VNP14IMG.A2025245.1012.002.nc --utm-zone 9 --hemisphere N

    # Glob pattern
    python vnp14_to_shp.py /data/viirs/2025/282/*.nc --utm-zone 9 --hemisphere N

    # With bbox and parallel workers
    python vnp14_to_shp.py /data/viirs --bbox -126.07 52.18 -124.37 53.21 --utm-zone 9 --hemisphere N -w 8
"""

import argparse
import sys
import os
import glob
import numpy as np
import netCDF4 as nc
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pyproj import Transformer
from datetime import datetime
from multiprocessing import Pool
from functools import partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_nc_files(input_paths):
    """
    Resolve input path(s) to a list of .nc files.

    Rules:
        - No paths given          → recursively scan cwd
        - Path is a directory     → recursively scan it
        - Path is a .nc file      → use it directly
        - Path is a glob pattern  → expand it
    """
    # Default to current directory if nothing provided
    if not input_paths:
        input_paths = ['.']

    nc_files = []

    for p in input_paths:
        if os.path.isdir(p):
            # Recursively find all .nc in this directory
            found = sorted(glob.glob(os.path.join(p, '**', '*.nc'), recursive=True))
            nc_files.extend(found)

        elif os.path.isfile(p) and p.lower().endswith('.nc'):
            # Single .nc file
            nc_files.append(p)

        else:
            # Try as a glob pattern
            expanded = sorted(glob.glob(p))
            # Keep only .nc files from the expansion
            expanded_nc = [f for f in expanded if f.lower().endswith('.nc')]
            if expanded_nc:
                nc_files.extend(expanded_nc)
            elif os.path.isfile(p):
                # It's a file but not .nc — warn and skip
                print(f"[WARN] Not a .nc file, skipping: {p}")
            else:
                print(f"[WARN] No .nc files matched: {p}")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in nc_files:
        af = os.path.abspath(f)
        if af not in seen:
            seen.add(af)
            unique.append(f)

    return unique


def get_epsg(zone, hemisphere):
    """Get EPSG code for a given UTM zone and hemisphere."""
    if hemisphere.upper() == 'N':
        return 32600 + zone
    else:
        return 32700 + zone


def extract_datetime(ds):
    """Extract datetime from NetCDF global attributes."""
    for attr in ['StartTime', 'PGE_StartTime']:
        if attr in ds.ncattrs():
            val = ds.getncattr(attr)
            try:
                return datetime.strptime(val.strip(), '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                pass

    if 'RangeBeginningDate' in ds.ncattrs() and 'RangeBeginningTime' in ds.ncattrs():
        date_str = ds.getncattr('RangeBeginningDate').strip()
        time_str = ds.getncattr('RangeBeginningTime').strip()
        try:
            return datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            pass

    fname = os.path.basename(ds.filepath())
    parts = fname.split('.')
    year = int(parts[1][1:5])
    jday = int(parts[1][5:])
    hhmm = parts[2]

    return datetime(year, 1, 1) + pd.Timedelta(days=jday - 1,
                                                 hours=int(hhmm[:2]),
                                                 minutes=int(hhmm[2:]))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_file(nc_path, utm_zone=None, hemisphere=None, bbox=None, output=None):
    """Process a single VNP14IMG NetCDF file. Returns output path or None."""

    if not os.path.exists(nc_path):
        print(f"Error: File not found: {nc_path}")
        return None

    print(f"Reading {nc_path} ...")

    ds = nc.Dataset(nc_path, 'r')
    granule_dt = extract_datetime(ds)

    print(f"  Granule datetime (UTC): {granule_dt.strftime('%Y-%m-%d %H:%M')}")

    lat = np.array(ds['FP_latitude'][:])
    lon = np.array(ds['FP_longitude'][:])
    frp = np.array(ds['FP_power'][:])
    conf = np.array(ds['FP_confidence'][:])

    extra_vars = {}
    for var_name in ['FP_T4', 'FP_T5', 'FP_day', 'FP_line', 'FP_sample']:
        if var_name in ds.variables:
            extra_vars[var_name] = np.array(ds[var_name][:])

    ds.close()

    n_total = len(lat)
    if n_total == 0:
        print("  No fire pixels found. Skipping.")
        return None

    print(f"  Total fire pixels in granule: {n_total}")

    # --- Apply bounding box filter ---
    if bbox is not None:
        west, south, east, north = bbox
        print(f"  Applying bbox filter: W={west} S={south} E={east} N={north}")
        mask = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)

        lat = lat[mask]
        lon = lon[mask]
        frp = frp[mask]
        conf = conf[mask]

        for var_name in extra_vars:
            extra_vars[var_name] = extra_vars[var_name][mask]

        n_fires = len(lat)
        print(f"  Fire pixels after bbox filter: {n_fires} (dropped {n_total - n_fires})")
        if n_fires == 0:
            print("  No fire pixels within bounding box. Skipping.")
            return None
    else:
        n_fires = n_total

    print(f"  Lat range:  {np.nanmin(lat):.4f} to {np.nanmax(lat):.4f}")
    print(f"  Lon range:  {np.nanmin(lon):.4f} to {np.nanmax(lon):.4f}")
    print(f"  FRP range:  {np.nanmin(frp):.2f} to {np.nanmax(frp):.2f} MW")

    # --- UTM zone and hemisphere ---
    _utm_zone = utm_zone
    _hemisphere = hemisphere
    print(f"  Using UTM zone: {_utm_zone}{_hemisphere}")

    epsg = get_epsg(_utm_zone, _hemisphere)
    print(f"  Target CRS: EPSG:{epsg}")

    # --- Reproject to UTM ---
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)

    # --- Build GeoDataFrame ---
    data = {
        'latitude': lat,
        'longitude': lon,
        'utm_x': utm_x,
        'utm_y': utm_y,
        'FRP_MW': frp,
        'confidence': conf
    }

    for var_name, var_data in extra_vars.items():
        col_name = var_name.replace('FP_', '')
        data[col_name] = var_data

    geometry = [Point(x, y) for x, y in zip(utm_x, utm_y)]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=f'EPSG:{epsg}')

    # --- Output path ---
    if output is None:
        out_dir = os.path.dirname(nc_path) or '.'
        timestamp = granule_dt.strftime('%Y%m%dT%H%M')
        out_path = os.path.join(out_dir, f'VIIRS_VNP14IMG_{timestamp}.shp')
    else:
        out_path = output

    gdf.to_file(out_path, driver='ESRI Shapefile')
    print(f"\n  Shapefile written to: {out_path}")
    print(f"  CRS: EPSG:{epsg} (UTM Zone {_utm_zone}{_hemisphere})")
    print(f"  Features: {len(gdf)}\n")
    return out_path


def _worker(nc_path, utm_zone, hemisphere, bbox):
    """Wrapper for multiprocessing (output always auto-generated)."""
    try:
        return process_file(nc_path, utm_zone=utm_zone, hemisphere=hemisphere, bbox=bbox)
    except Exception as e:
        print(f"Error processing {nc_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert VNP14IMG fire pixels to Shapefile in Sentinel-2 UTM projection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Current directory (finds all .nc recursively)
  python vnp14_to_shp.py --utm-zone 9 --hemisphere N

  # A directory
  python vnp14_to_shp.py /data/viirs/2025 --utm-zone 9 --hemisphere N

  # Single file
  python vnp14_to_shp.py VNP14IMG.A2025245.1012.002.nc --utm-zone 9 --hemisphere N

  # Glob + bbox + parallel
  python vnp14_to_shp.py /data/viirs/2025/282/*.nc --bbox -126.07 52.18 -124.37 53.21 --utm-zone 9 --hemisphere N -w 8
        """,
    )
    parser.add_argument('nc_paths', nargs='*', default=[],
                        help='Path(s) to .nc file(s), directory, or glob pattern. '
                             'Default: current directory (recursive).')
    parser.add_argument('--utm-zone', type=int, default=None,
                        help='UTM zone number')
    parser.add_argument('--hemisphere', type=str, default=None, choices=['N', 'S'],
                        help='Hemisphere N or S')
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
                        default=None,
                        help='Bounding box to clip fire pixels (lon_min lat_min lon_max lat_max)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output shapefile path (only valid for single file input)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    # Resolve inputs to a list of .nc files
    nc_files = find_nc_files(args.nc_paths)

    if not nc_files:
        print("No .nc files found.")
        sys.exit(0)

    n_files = len(nc_files)
    print(f"Found {n_files} .nc file(s). Workers: {args.workers}\n")

    if args.output and n_files > 1:
        print("Warning: -o/--output ignored for batch mode (multiple files). "
              "Output names will be auto-generated.\n")
        args.output = None

    if n_files == 1 or args.workers <= 1:
        results = []
        for f in nc_files:
            r = process_file(f, utm_zone=args.utm_zone, hemisphere=args.hemisphere,
                             bbox=args.bbox, output=args.output)
            results.append(r)
    else:
        worker_fn = partial(_worker,
                            utm_zone=args.utm_zone,
                            hemisphere=args.hemisphere,
                            bbox=args.bbox)
        with Pool(processes=min(args.workers, n_files)) as pool:
            results = pool.map(worker_fn, nc_files)

    written = [r for r in results if r is not None]
    print(f"\nDone. {len(written)}/{n_files} shapefiles written.")


if __name__ == '__main__':
    main()