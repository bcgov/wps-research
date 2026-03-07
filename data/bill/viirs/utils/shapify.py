#!/usr/bin/env python
"""
viirs.utils.shapify
=========================

Convert VNP14IMG NetCDF fire pixel data to a Shapefile in a projected CRS.

By default, auto-detects a UTM zone from the data. If a reference raster is
provided (-r), all pixels are projected to the raster's CRS instead — this
is the recommended approach when working with a specific analysis raster
(e.g. Sentinel-2 in BC Albers EPSG:3005).

Notice:
    If you shapify with a raster reference, all output shapefiles share the
    same CRS.  You do NOT need to pass the raster again in accumulate or
    the GUI — they will all concatenate cleanly.

Usage:
    # Auto UTM (default)
    python -m viirs.utils.shapify /data/viirs/2025

    # Match a reference raster (recommended)
    python -m viirs.utils.shapify /data/viirs/2025 -r sentinel2.bin

    # Explicit CRS
    python -m viirs.utils.shapify /data/viirs/2025 --crs EPSG:3005

    # Force a specific UTM zone
    python -m viirs.utils.shapify /data/viirs/2025 --utm-zone 9 --hemisphere N

    # With bounding box filter and workers
    python -m viirs.utils.shapify /data/viirs/2025 -r sentinel2.bin --bbox -126.07 52.18 -124.37 53.21 -w 8
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
from pyproj import Transformer, CRS
from datetime import datetime
from multiprocessing import get_context, Pool
from functools import partial


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _remove_shapefile(shp_path: str):
    """Delete all sidecar files belonging to a shapefile."""
    base = os.path.splitext(shp_path)[0]
    for ext in (
        '.shp', '.shx', '.dbf', '.prj', '.cpg',
        '.sbn', '.sbx', '.fbn', '.fbx',
        '.ain', '.aih', '.ixs', '.mxs',
        '.atx', '.shp.xml', '.qpj',
    ):
        candidate = base + ext
        if os.path.exists(candidate):
            os.remove(candidate)
            print(f"  Removed existing: {candidate}")


def find_nc_files(input_paths):
    if not input_paths:
        input_paths = ['.']
    nc_files = []
    for p in input_paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, '**', '*.nc'), recursive=True))
            nc_files.extend(found)
        elif os.path.isfile(p) and p.lower().endswith('.nc'):
            nc_files.append(p)
        else:
            expanded = sorted(glob.glob(p))
            expanded_nc = [f for f in expanded if f.lower().endswith('.nc')]
            if expanded_nc:
                nc_files.extend(expanded_nc)
            elif os.path.isfile(p):
                print(f"[WARN] Not a .nc file, skipping: {p}")
            else:
                print(f"[WARN] No .nc files matched: {p}")
    seen = set()
    unique = []
    for f in nc_files:
        af = os.path.abspath(f)
        if af not in seen:
            seen.add(af)
            unique.append(f)
    return unique


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def get_epsg(zone, hemisphere):
    if hemisphere.upper() == 'N':
        return 32600 + zone
    else:
        return 32700 + zone


def auto_utm(lon, lat):
    med_lon = float(np.nanmedian(lon))
    med_lat = float(np.nanmedian(lat))
    zone = int((med_lon + 180) / 6) + 1
    zone = max(1, min(60, zone))
    hemisphere = 'N' if med_lat >= 0 else 'S'
    return zone, hemisphere


def get_crs_from_raster(raster_path: str) -> str:
    """Read the CRS from a raster file (ENVI .bin/.hdr or GeoTIFF) via GDAL."""
    from osgeo import gdal
    gdal.UseExceptions()

    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {raster_path}")

    wkt = ds.GetProjection()
    if not wkt:
        raise RuntimeError(
            f"Raster has no projection metadata: {raster_path}\n"
            f"Make sure the .hdr file contains projection info."
        )
    ds = None
    return wkt


def crs_label(crs_input) -> str:
    """Return a human-readable label like 'EPSG:3005 (NAD83 / BC Albers)'."""
    try:
        _crs = CRS.from_user_input(crs_input)
        epsg = _crs.to_epsg()
        name = _crs.name
        if epsg:
            return f"EPSG:{epsg} ({name})"
        return name or str(crs_input)[:80]
    except Exception:
        return str(crs_input)[:80]


# ---------------------------------------------------------------------------
# Datetime extraction
# ---------------------------------------------------------------------------

def extract_datetime(ds):
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

def process_file(nc_path, utm_zone=None, hemisphere=None, bbox=None,
                 output=None, target_crs=None):
    """
    Convert one .nc file to a shapefile.

    Parameters
    ----------
    nc_path : str
    utm_zone : int, optional — forced UTM zone (ignored if target_crs is set)
    hemisphere : str, optional — 'N' or 'S' (ignored if target_crs is set)
    bbox : tuple, optional — (west, south, east, north) in WGS84 degrees
    output : str, optional — output path
    target_crs : str, optional — WKT or EPSG string to project into.
        If set, overrides utm_zone/hemisphere entirely.
    """
    if not os.path.exists(nc_path):
        print(f"Error: File not found: {nc_path}")
        return None

    print(f"\nReading {nc_path} ...")
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

    # ---- Bounding box filter (applied in WGS84 before projection) ----
    if bbox is not None:
        west, south, east, north = bbox
        print(f"  Applying bbox filter: W={west} S={south} E={east} N={north}")
        mask = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
        lat, lon, frp, conf = lat[mask], lon[mask], frp[mask], conf[mask]
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

    # ---- Determine output CRS ----
    if target_crs is not None:
        # Reference raster or explicit CRS provided — use it directly
        out_crs = CRS.from_user_input(target_crs)
        print(f"  Target CRS: {crs_label(out_crs)}")
    else:
        # Fall back to auto/manual UTM
        _utm_zone = utm_zone
        _hemisphere = hemisphere
        if _utm_zone is None or _hemisphere is None:
            auto_zone, auto_hemi = auto_utm(lon, lat)
            if _utm_zone is None:
                _utm_zone = auto_zone
            if _hemisphere is None:
                _hemisphere = auto_hemi
            print(f"  Auto-detected UTM zone: {_utm_zone}{_hemisphere}")
        else:
            print(f"  Using UTM zone: {_utm_zone}{_hemisphere}")

        epsg = get_epsg(_utm_zone, _hemisphere)
        out_crs = CRS.from_epsg(epsg)
        print(f"  Target CRS: {crs_label(out_crs)}")

    # ---- Project ----
    transformer = Transformer.from_crs("EPSG:4326", out_crs, always_xy=True)
    proj_x, proj_y = transformer.transform(lon, lat)

    # ---- Build GeoDataFrame ----
    data = {
        'latitude': lat, 'longitude': lon,
        'x': proj_x, 'y': proj_y,
        'FRP_MW': frp, 'confidence': conf,
    }
    for var_name, var_data in extra_vars.items():
        data[var_name.replace('FP_', '')] = var_data

    geometry = [Point(px, py) for px, py in zip(proj_x, proj_y)]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=out_crs)

    # ---- Output path ----
    if output is None:
        out_dir = os.path.dirname(nc_path) or '.'
        timestamp = granule_dt.strftime('%Y%m%dT%H%M')
        out_path = os.path.join(out_dir, f'VIIRS_VNP14IMG_{timestamp}.shp')
    else:
        out_path = output

    # ---- Replace existing shapefile (all sidecar files) if present ----
    if os.path.exists(out_path):
        print(f"  Output exists — replacing: {out_path}")
        _remove_shapefile(out_path)

    gdf.to_file(out_path, driver='ESRI Shapefile')
    print(f"\n  Shapefile written to: {out_path}")
    print(f"  CRS: {crs_label(out_crs)}")
    print(f"  Features: {len(gdf)}\n")
    return out_path


def _worker(nc_path, utm_zone, hemisphere, bbox, target_crs):
    try:
        return process_file(nc_path, utm_zone=utm_zone, hemisphere=hemisphere,
                            bbox=bbox, target_crs=target_crs)
    except Exception as e:
        print(f"Error processing {nc_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert VNP14IMG fire pixels to Shapefile.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto UTM (default)
  python -m viirs.utils.shapify /data/viirs/2025

  # Match a reference raster (recommended)
  python -m viirs.utils.shapify /data/viirs/2025 -r sentinel2.bin

  # Explicit CRS
  python -m viirs.utils.shapify /data/viirs/2025 --crs EPSG:3005

  # Force UTM zone
  python -m viirs.utils.shapify /data/viirs/2025 --utm-zone 9 --hemisphere N

  # With bbox and parallel workers
  python -m viirs.utils.shapify /data/viirs/2025 -r s2.bin --bbox -126.07 52.18 -124.37 53.21 -w 8
        """,
    )

    parser.add_argument('nc_paths', nargs='*', default=[],
                        help='Path(s) to .nc file(s), directory, or glob. Default: cwd.')
    parser.add_argument('-r', '--reference', default=None,
                        help='Reference raster file (.bin, .hdr, .tif). '
                             'All output shapefiles will use its CRS. Recommended.')
    parser.add_argument('--crs', default=None,
                        help='Explicit target CRS (e.g. EPSG:3005). Overrides --reference.')
    parser.add_argument('--utm-zone', type=int, default=None,
                        help='UTM zone number (ignored if -r or --crs is set)')
    parser.add_argument('--hemisphere', type=str, default=None, choices=['N', 'S'],
                        help='Hemisphere (ignored if -r or --crs is set)')
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'),
                        default=None, help='Bounding box filter in WGS84 degrees.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output shapefile path (single file mode only).')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Number of parallel workers.')

    args = parser.parse_args()

    # ---- Resolve target CRS ----
    target_crs = None
    if args.crs is not None:
        target_crs = args.crs
        print(f"[INFO] Using explicit CRS: {crs_label(target_crs)}")
    elif args.reference is not None:
        if not os.path.exists(args.reference):
            print(f"Error: Reference raster not found: {args.reference}")
            sys.exit(1)
        target_crs = get_crs_from_raster(args.reference)
        print(f"[INFO] Using CRS from {os.path.basename(args.reference)}: {crs_label(target_crs)}")

    # ---- Find files ----
    nc_files = find_nc_files(args.nc_paths)
    if not nc_files:
        print("No .nc files found.")
        sys.exit(0)

    n_files = len(nc_files)
    print(f"Found {n_files} .nc file(s). Workers: {args.workers}\n")

    if args.output and n_files > 1:
        print("Warning: -o/--output ignored for batch mode.\n")
        args.output = None

    # ---- Process ----
    if n_files == 1 or args.workers <= 1:
        results = [
            process_file(f, utm_zone=args.utm_zone, hemisphere=args.hemisphere,
                         bbox=args.bbox, output=args.output, target_crs=target_crs)
            for f in nc_files
        ]
    else:
        worker_fn = partial(_worker, utm_zone=args.utm_zone,
                            hemisphere=args.hemisphere, bbox=args.bbox,
                            target_crs=target_crs)
        with get_context('spawn').Pool(processes=min(args.workers, n_files)) as pool:
            results = pool.map(worker_fn, nc_files)

    written = [r for r in results if r is not None]
    print(f"\nDone. {len(written)}/{n_files} shapefiles written.")


if __name__ == '__main__':
    main()