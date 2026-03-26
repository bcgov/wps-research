#!/usr/bin/env python3
"""
batch_fire_mapping/run_fire_mapping.py
======================================
High-level batch fire-mapping script.

For every fire polygon in the input shapefile that overlaps the input
Sentinel-2 raster, this script:

  1.  Downloads VIIRS VNP14IMG data for the full season (skips days that
      are already on disk).
  2.  Converts downloaded .nc files to shapefiles via viirs.utils.shapify
      (skips granules whose .shp already exists).
  3.  For each fire polygon:
        a.  Crops the main raster to a box that is at most <crop_buffer_px>
            pixels wider than the polygon on each side.
        b.  Finds all VIIRS pixels that fall inside the fire polygon and
            determines the actual accumulation end date (= latest detection
            inside the polygon).
        c.  Accumulates VIIRS shapefiles from (FIRE_DATE − 5 days) to the
            end date found in step b.
        d.  Keeps only the final (most complete) accumulated shapefile in
            the fire output folder and rasterizes it onto the cropped raster.
        e.  Rasterizes the traditional fire polygon perimeter for the
            comparison panel.
        f.  Calls fire_mapping_cli.py with the cropped raster, the VIIRS
            binary hint, and all model hyperparameters.

Output structure
----------------
<output_root>/
    fire_mapping_results/
        <FIRE_NUMBE>/
            <FIRE_NUMBE>_crop.bin          # cropped Sentinel-2 raster
            <FIRE_NUMBE>_crop.hdr
            VIIRS_VNP14IMG_<s>_<e>.shp    # final accumulated shapefile
            VIIRS_VNP14IMG_<s>_<e>.bin    # rasterized VIIRS hint
            <FIRE_NUMBE>_perimeter.bin     # rasterized fire polygon
            <crop_name>_classified.bin     # fire_mapping_cli.py output
            <FIRE_NUMBE>_comparison.png    # 3-panel comparison figure

Usage
-----
    python batch_fire_mapping/run_fire_mapping.py  POLYGONS.shp  RASTER.bin  [options]

Example
-------
    python batch_fire_mapping/run_fire_mapping.py         \\
        IN_HISTORICAL_FIRE_POLYGONS_SVW.shp               \\
        C11659/S2C_MSIL1C_20251014T192401_...20m.bin      \\
        --year 2025 --crop_buffer_px 100 --output_dir results/
"""

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
import os
import sys

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
# wps-research root — two levels above bill/
_REPO_ROOT    = os.path.dirname(os.path.dirname(_PROJECT_ROOT))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import datetime
import glob
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Project utilities
# ---------------------------------------------------------------------------
from viirs.utils.accumulate import accumulate, extract_datetime_from_filename
from viirs.utils.rasterize  import rasterize_shapefile
from viirs.utils.shapify    import (
    process_file as shapify_file,
    get_crs_from_raster,
    get_extent_from_raster,
)


# ===========================================================================
# Terminal output helpers
# ===========================================================================

def _box(title: str, lines: list = None, char: str = '=', width: int = 68):
    """Print a labelled box to stdout."""
    content = [title] + (lines or [])
    w = max(width, *(len(l) + 4 for l in content))
    bar = char * w
    print(f'\n{bar}')
    print(f'  {title}')
    if lines:
        print(char * w)
        for l in lines:
            print(f'  {l}')
    print(f'{bar}')


def _info(msg: str):
    print(f'  [INFO]  {msg}')


def _warn(msg: str):
    print(f'  [WARN]  {msg}')


def _skip(fire_numbe: str, reason: str):
    _box(f'SKIP  {fire_numbe}', [reason], char='-', width=68)


# ===========================================================================
# Raster helpers
# ===========================================================================

def get_raster_info(raster_path: str):
    """Return (crs_wkt, geotransform, width, height)."""
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Cannot open raster: {raster_path}')
    crs = ds.GetProjection()
    gt  = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    ds = None
    return crs, gt, w, h


def raster_native_extent(gt, W: int, H: int):
    """Return (xmin, ymin, xmax, ymax) in the raster's native CRS."""
    xs = [gt[0], gt[0] + W * gt[1]]
    ys = [gt[3], gt[3] + H * gt[5]]
    return min(xs), min(ys), max(xs), max(ys)


def raster_extent_to_wgs84(crs_wkt: str, xmin, ymin, xmax, ymax):
    """
    Convert a bounding box in the raster's CRS to WGS84 (W, S, E, N).
    Used to build LAADS DAAC download URLs.
    """
    src = osr.SpatialReference()
    src.ImportFromWkt(crs_wkt)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    ct = osr.CoordinateTransformation(src, dst)

    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    lats, lons = [], []
    for x, y in corners:
        lat, lon, _ = ct.TransformPoint(x, y)
        lats.append(lat)
        lons.append(lon)
    return min(lons), min(lats), max(lons), max(lats)   # W, S, E, N


def crop_raster(src_path: str, dst_path: str,
                xmin: float, ymin: float,
                xmax: float, ymax: float) -> bool:
    """
    Crop *src_path* to the projected window [xmin,ymin,xmax,ymax] and write
    ENVI .bin to *dst_path*.  Returns True on success.
    """
    # gdal.Translate projWin = [ulx, uly, lrx, lry]
    result = gdal.Translate(
        dst_path, src_path,
        projWin=[xmin, ymax, xmax, ymin],
        format='ENVI',
    )
    if result is None:
        return False
    result = None   # flush
    return True


# ===========================================================================
# Token management
# ===========================================================================

_TOKEN_PATH = '/data/.tokens/laads'


def load_token() -> str:
    """
    Try to read the LAADS DAAC token from the standard path.
    If not found, prompt the user in the terminal.
    """
    if os.path.exists(_TOKEN_PATH):
        token = Path(_TOKEN_PATH).read_text().strip()
        if token:
            _info(f'LAADS token loaded from {_TOKEN_PATH}')
            return token

    print('\n' + '=' * 60)
    print('  LAADS DAAC token not found.')
    print(f'  Expected path: {_TOKEN_PATH}')
    print('  Get a free token at: https://ladsweb.modaps.eosdis.nasa.gov')
    print('=' * 60)
    token = input('  Paste your LAADS token here: ').strip()

    if not token:
        raise RuntimeError('No LAADS token provided.  Aborting.')

    save = input(f'  Save token to {_TOKEN_PATH}? [y/N]: ').strip().lower()
    if save == 'y':
        os.makedirs(os.path.dirname(_TOKEN_PATH), exist_ok=True)
        Path(_TOKEN_PATH).write_text(token)
        _info(f'Token saved to {_TOKEN_PATH}')

    return token


# ===========================================================================
# VIIRS download
# ===========================================================================

def download_viirs(
    raster_path: str,
    start_dt: datetime.datetime,
    end_dt:   datetime.datetime,
    token:    str,
    viirs_save_dir: str,
    max_workers: int = 16,
):
    """
    Download VNP14IMG .nc files for [start_dt, end_dt] restricted to the
    raster's bounding box.  Already-downloaded days are skipped.

    Files are saved to:
        <viirs_save_dir>/VNP14IMG/<YYYY>/<DDD>/
    """
    from viirs.utils.laads_data_download_v2 import sync

    crs_wkt, gt, W, H = get_raster_info(raster_path)
    xmin, ymin, xmax, ymax = raster_native_extent(gt, W, H)
    west, south, east, north = raster_extent_to_wgs84(
        crs_wkt, xmin, ymin, xmax, ymax)

    _info(f'WGS84 bbox  W={west:.4f}  S={south:.4f}  '
          f'E={east:.4f}  N={north:.4f}')

    product = 'VNP14IMG'
    interval = datetime.timedelta(days=1)
    days = []
    d = start_dt
    while d <= end_dt:
        days.append(d)
        d += interval

    _info(f'Date range: {start_dt.date()} → {end_dt.date()}  '
          f'({len(days)} days)')

    # Collect days that have not been downloaded yet
    pending = []
    for day in days:
        jday = day.timetuple().tm_yday
        day_dir = os.path.join(
            viirs_save_dir, product,
            f'{day.year:04d}', f'{jday:03d}')
        existing = glob.glob(os.path.join(day_dir, '*.nc'))
        if existing:
            pass   # already downloaded
        else:
            pending.append(day)

    _info(f'Days already downloaded: {len(days) - len(pending)}  '
          f'|  Pending: {len(pending)}')

    if not pending:
        _info('All days already on disk — skipping download.')
        return

    completed = {'n': 0}

    def _download_one(day):
        jday = day.timetuple().tm_yday
        url = (
            f'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?'
            f'products={product}&'
            f'temporalRanges={day.year}-{jday}&'
            f'regions=%5BBBOX%5DN{north:.6f}%20S{south:.6f}'
            f'%20E{east:.6f}%20W{west:.6f}'
        )
        day_dir = os.path.join(
            viirs_save_dir, product,
            f'{day.year:04d}', f'{jday:03d}')
        os.makedirs(day_dir, exist_ok=True)
        try:
            sync(url, day_dir, token)
        except Exception as exc:
            _warn(f'Download error for {day.date()}: {exc}')
        completed['n'] += 1
        print(f'  [{completed["n"]:3d}/{len(pending)}]  '
              f'{day.strftime("%Y-%m-%d")}  done', flush=True)

    _box(f'Downloading {len(pending)} days of VIIRS VNP14IMG',
         [f'Workers: {max_workers}',
          f'Save dir: {viirs_save_dir}'])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_one, day): day for day in pending}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:
                _warn(f'Unhandled download error: {exc}')

    _info('Download complete.')


# ===========================================================================
# Shapify (nc → shp)
# ===========================================================================

def _shapify_one_worker(args):
    """
    Module-level worker so it can be pickled by ProcessPoolExecutor.
    args = (nc_path, target_crs, raster_extent)
    """
    nc_path, target_crs, raster_extent = args
    try:
        shapify_file(nc_path, target_crs=target_crs, raster_extent=raster_extent)
    except Exception as exc:
        return f'WARN: shapify failed for {os.path.basename(nc_path)}: {exc}'
    return None


def shapify_viirs(viirs_save_dir: str, raster_path: str, workers: int = 8):
    """
    Convert any .nc file in *viirs_save_dir* that does not yet have a
    corresponding .shp to a shapefile in the raster's CRS.
    Already-shapified granules are skipped.

    Uses ProcessPoolExecutor (not threads) because netCDF4 / GDAL / pyproj
    are not thread-safe and will segfault under ThreadPoolExecutor.
    """
    target_crs     = get_crs_from_raster(raster_path)
    raster_extent  = get_extent_from_raster(raster_path)  # (x_min,x_max,y_min,y_max)

    nc_files = sorted(glob.glob(
        os.path.join(viirs_save_dir, '**', '*.nc'), recursive=True))

    pending = []
    for nc in nc_files:
        nc_dir    = os.path.dirname(nc)
        shp_files = glob.glob(os.path.join(nc_dir, '*.shp'))
        if not shp_files:
            pending.append(nc)

    _info(f'Shapify: {len(nc_files)} .nc files found, '
          f'{len(pending)} need conversion.')

    if not pending:
        _info('All granules already shapified — skipping.')
        return

    _box(f'Shapifying {len(pending)} granules', [f'Workers: {workers}'])

    work_args = [(nc, target_crs, raster_extent) for nc in pending]

    if workers <= 1 or len(pending) == 1:
        for i, a in enumerate(work_args, 1):
            msg = _shapify_one_worker(a)
            if msg:
                print(f'  {msg}', flush=True)
            print(f'  [{i}/{len(pending)}] done', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_shapify_one_worker, a): a for a in work_args}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    msg = fut.result()
                    if msg:
                        print(f'  {msg}', flush=True)
                except Exception as exc:
                    print(f'  WARN: shapify error: {exc}', flush=True)
                print(f'  [{i}/{len(pending)}] done', flush=True)

    _info('Shapify complete.')


# ===========================================================================
# Load all VIIRS shapefiles (once, before the fire loop)
# ===========================================================================

def load_all_viirs(shp_root: str, raster_crs: str) -> gpd.GeoDataFrame:
    """
    Load every .shp inside *shp_root* into a single GeoDataFrame.
    Adds columns: detection_datetime (datetime), detection_date (date).
    Reprojects to *raster_crs*.
    """
    shp_files = sorted(glob.glob(
        os.path.join(shp_root, '**', '*.shp'), recursive=True))

    _info(f'Loading {len(shp_files)} VIIRS shapefiles ...')

    frames = []
    for fpath in shp_files:
        stem = Path(fpath).stem
        dt   = extract_datetime_from_filename(stem)
        if dt is None:
            continue
        try:
            gdf = gpd.read_file(fpath)
            gdf['detection_datetime'] = dt
            gdf['detection_date']     = dt.date()
            frames.append(gdf)
        except Exception as exc:
            _warn(f'Could not read {os.path.basename(fpath)}: {exc}')

    if not frames:
        _info('No VIIRS shapefiles loaded.')
        return gpd.GeoDataFrame()

    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

    # Reproject to raster CRS if needed
    if result.crs is not None and str(result.crs) != raster_crs:
        result = result.to_crs(raster_crs)

    _info(f'VIIRS GeoDataFrame: {len(result):,} points in total.')
    return result


# ===========================================================================
# Rasterize a polygon (fire perimeter)
# ===========================================================================

def rasterize_polygon(polygon_geom, crs, ref_raster_path: str,
                      dst_path: str):
    """
    Rasterize a single polygon geometry onto the grid of *ref_raster_path*
    and write ENVI .bin to *dst_path*.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_shp = os.path.join(tmpdir, 'perim.shp')
        gdf = gpd.GeoDataFrame(geometry=[polygon_geom], crs=crs)
        gdf.to_file(tmp_shp, driver='ESRI Shapefile')

        ref    = gdal.Open(ref_raster_path, gdal.GA_ReadOnly)
        shp_ds = ogr.Open(tmp_shp)
        layer  = shp_ds.GetLayer()

        out = gdal.GetDriverByName('ENVI').Create(
            dst_path,
            ref.RasterXSize, ref.RasterYSize, 1, gdal.GDT_Float32,
        )
        out.SetGeoTransform(ref.GetGeoTransform())
        out.SetProjection(ref.GetProjection())

        band = out.GetRasterBand(1)
        band.Fill(0.0)
        band.SetNoDataValue(0.0)

        gdal.RasterizeLayer(
            out, [1], layer,
            burn_values=[1.0],
            options=['ALL_TOUCHED=TRUE'],
        )
        band = out = ref = shp_ds = None

    _info(f'Perimeter rasterized → {os.path.basename(dst_path)}')


# ===========================================================================
# Per-fire processing
# ===========================================================================

def process_fire(
    row,
    viirs_gdf:       gpd.GeoDataFrame,
    raster_path:     str,
    raster_crs:      str,
    raster_gt,
    raster_W:        int,
    raster_H:        int,
    output_root:     str,
    crop_buffer_px:       int,
    cli_script:      str,
    cli_pass_args:   list,
    viirs_shp_dir:   str,
):
    """
    Process one fire polygon end-to-end.

    Returns the path to the fire output directory on success, or None.
    """
    fire_numbe = str(row.get('FIRE_NUMBE', row.name))
    fire_date_str = str(row.get('FIRE_DATE', ''))

    _box(
        f'Processing  {fire_numbe}',
        [
            f'FIRE_DATE : {fire_date_str}',
            f'Geometry  : {row.geometry.geom_type}',
        ],
        char='-',
    )

    # -----------------------------------------------------------------------
    # 1. Parse FIRE_DATE
    # Geopandas may already have parsed the column as datetime64, so
    # str(value) can be '2025-08-25 00:00:00' rather than '2025-08-25'.
    # Accept both formats by trying the full datetime string first.
    # -----------------------------------------------------------------------
    try:
        raw = row.get('FIRE_DATE', '')
        # If geopandas gave us a Timestamp/datetime, convert directly
        if hasattr(raw, 'date'):
            fire_date = datetime.datetime(raw.year, raw.month, raw.day)
        else:
            # Strip any trailing time component before parsing
            fire_date = datetime.datetime.strptime(
                str(raw).split()[0], '%Y-%m-%d')
    except (ValueError, AttributeError):
        _skip(fire_numbe, f'Cannot parse FIRE_DATE: {fire_date_str!r}')
        return None

    acc_start = fire_date - datetime.timedelta(days=5)

    # -----------------------------------------------------------------------
    # 2. Compute pixel bounding box of the fire polygon
    # -----------------------------------------------------------------------
    bounds = row.geometry.bounds   # (minx, miny, maxx, maxy) in raster CRS

    gt = raster_gt
    # Convert geographic bounds to pixel indices
    px_lo = int((bounds[0] - gt[0]) / gt[1])
    px_hi = int((bounds[2] - gt[0]) / gt[1])
    py_lo = int((bounds[3] - gt[3]) / gt[5])   # maxy → top row
    py_hi = int((bounds[1] - gt[3]) / gt[5])   # miny → bottom row

    # Add buffer, clip to raster extent
    px_lo = max(0, px_lo - crop_buffer_px)
    px_hi = min(raster_W - 1, px_hi + crop_buffer_px)
    py_lo = max(0, py_lo - crop_buffer_px)
    py_hi = min(raster_H - 1, py_hi + crop_buffer_px)

    if px_lo >= px_hi or py_lo >= py_hi:
        _skip(fire_numbe, 'Polygon bbox is entirely outside the raster.')
        return None

    # Back to projected coordinates for GDAL Translate
    crop_xmin = gt[0] + px_lo * gt[1]
    crop_xmax = gt[0] + px_hi * gt[1]
    crop_ymax = gt[3] + py_lo * gt[5]   # row py_lo = top = ymax
    crop_ymin = gt[3] + py_hi * gt[5]   # row py_hi = bottom = ymin

    _info(f'Crop window: px [{px_lo}:{px_hi}] × py [{py_lo}:{py_hi}]  '
          f'({px_hi - px_lo} × {py_hi - py_lo} px)')

    # -----------------------------------------------------------------------
    # 3. Create fire output directory
    # -----------------------------------------------------------------------
    fire_dir = os.path.join(output_root, 'fire_mapping_results', fire_numbe)
    if os.path.isdir(fire_dir):
        _info(f'Removing existing output folder: {fire_dir}')
        shutil.rmtree(fire_dir)
    os.makedirs(fire_dir)

    # -----------------------------------------------------------------------
    # 4. Crop the main Sentinel-2 raster
    # -----------------------------------------------------------------------
    crop_name = f'{fire_numbe}_crop'
    crop_bin  = os.path.join(fire_dir, f'{crop_name}.bin')

    _info(f'Writing cropped raster → {os.path.basename(crop_bin)}')
    if not crop_raster(raster_path, crop_bin, crop_xmin, crop_ymin,
                       crop_xmax, crop_ymax):
        _skip(fire_numbe, 'GDAL Translate failed — no overlap with raster?')
        return None

    # -----------------------------------------------------------------------
    # 5. Find VIIRS pixels inside the fire polygon
    # -----------------------------------------------------------------------
    if viirs_gdf.empty:
        _warn('No VIIRS data loaded — cannot determine accumulation end date.')
        acc_end        = fire_date
        plot_start     = acc_start.date()
        plot_end       = acc_end.date()
    else:
        inside = viirs_gdf[viirs_gdf.geometry.within(row.geometry)]

        if inside.empty:
            _warn(f'No VIIRS pixels inside polygon {fire_numbe}.')
            _warn('Using FIRE_DATE as accumulation end date.')
            acc_end    = fire_date
            plot_start = acc_start.date()
            plot_end   = fire_date.date()
        else:
            acc_end = datetime.datetime.combine(
                inside['detection_date'].max(), datetime.time.min)

            # Plot start = earliest pixel inside polygon from acc_start onward
            inside_window = inside[
                inside['detection_datetime'] >= acc_start]
            if not inside_window.empty:
                plot_start = inside_window['detection_date'].min()
            else:
                plot_start = acc_start.date()

            plot_end = acc_end.date()
            _info(f'VIIRS pixels inside polygon: {len(inside)}')
            _info(f'Accumulation: {acc_start.date()} → {acc_end.date()}')
            _info(f'Plot dates  : {plot_start} → {plot_end}')

    # -----------------------------------------------------------------------
    # 6. Accumulate VIIRS for this fire's date range
    # -----------------------------------------------------------------------
    _info('Running VIIRS accumulation ...')
    tmp_acc_dir = tempfile.mkdtemp(prefix=f'acc_{fire_numbe}_')

    try:
        acc_paths = accumulate(
            shp_dir          = viirs_shp_dir,
            start_str        = acc_start.strftime('%Y%m%d'),
            end_str          = acc_end.strftime('%Y%m%d'),
            reference_raster = crop_bin,
            output_dir       = tmp_acc_dir,
        )
    except Exception as exc:
        _warn(f'Accumulation failed: {exc}')
        shutil.rmtree(tmp_acc_dir, ignore_errors=True)
        acc_paths = []

    if not acc_paths:
        _warn('No accumulated shapefiles produced.')
        shutil.rmtree(tmp_acc_dir, ignore_errors=True)
        # Cannot run without a VIIRS hint — skip this fire
        _skip(fire_numbe, 'Accumulation produced no output.')
        return None

    # Keep only the final (most complete) accumulated shapefile
    final_acc_shp = sorted(acc_paths)[-1]
    final_stem    = Path(final_acc_shp).stem

    _info(f'Final accumulated shapefile: {final_stem}')

    # Copy the shapefile sidecar bundle to the fire output dir
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        src = Path(final_acc_shp).with_suffix(ext)
        if src.exists():
            shutil.copy2(src, os.path.join(fire_dir, src.name))

    shutil.rmtree(tmp_acc_dir, ignore_errors=True)

    acc_shp_in_fire_dir = os.path.join(fire_dir, Path(final_acc_shp).name)

    # -----------------------------------------------------------------------
    # 7. Rasterize accumulated VIIRS onto the cropped raster
    # -----------------------------------------------------------------------
    _info('Rasterizing accumulated VIIRS ...')
    viirs_bin = rasterize_shapefile(
        shp_path   = acc_shp_in_fire_dir,
        ref_image  = crop_bin,
        output_dir = fire_dir,
        buffer_m   = 375.0,
    )

    if viirs_bin is None:
        _skip(fire_numbe, 'VIIRS rasterization failed.')
        return None

    _info(f'VIIRS binary → {os.path.basename(viirs_bin)}')

    # -----------------------------------------------------------------------
    # 8. Rasterize the traditional fire perimeter
    # -----------------------------------------------------------------------
    perim_bin = os.path.join(fire_dir, f'{fire_numbe}_perimeter.bin')
    try:
        rasterize_polygon(row.geometry, raster_crs, crop_bin, perim_bin)
    except Exception as exc:
        _warn(f'Perimeter rasterization failed: {exc}')
        perim_bin = None

    # -----------------------------------------------------------------------
    # 9. Call fire_mapping_cli.py
    # -----------------------------------------------------------------------
    cmd = [
        sys.executable, cli_script,
        crop_bin, viirs_bin,
        '--fire_numbe',  fire_numbe,
        '--start_date',  str(plot_start),
        '--end_date',    str(plot_end),
    ]
    if perim_bin and os.path.exists(perim_bin):
        cmd += ['--perimeter', perim_bin]
    cmd += cli_pass_args

    _info(f'Calling fire_mapping_cli.py ...')
    _info(f'Command: {" ".join(os.path.basename(c) if i < 3 else c for i, c in enumerate(cmd))}')

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)

    if result.returncode != 0:
        _warn(f'fire_mapping_cli.py exited with code {result.returncode}')
    else:
        _info('fire_mapping_cli.py completed successfully.')

    return fire_dir


# ===========================================================================
# Polygon loading and filtering
# ===========================================================================

def load_and_filter_polygons(
    polygon_file: str,
    raster_path:  str,
    year:         int = None,
) -> gpd.GeoDataFrame:
    """
    Load the fire polygon shapefile, reproject to the raster's CRS, and keep
    only polygons that intersect the raster extent.
    If *year* is given, also filter to that FIRE_YEAR.
    """
    _info(f'Loading polygon file: {polygon_file}')
    gdf = gpd.read_file(polygon_file)
    _info(f'Total features: {len(gdf)}')

    if year is not None:
        if 'FIRE_YEAR' in gdf.columns:
            gdf = gdf[pd.to_numeric(gdf['FIRE_YEAR'], errors='coerce') == year].copy()
            _info(f'After --year={year} filter: {len(gdf)} features')
        else:
            _warn('FIRE_YEAR column not found — ignoring --year filter.')

    if gdf.empty:
        return gdf

    # Reproject to raster CRS
    crs_wkt, gt, W, H = get_raster_info(raster_path)
    gdf = gdf.to_crs(crs_wkt)

    # Spatial filter: keep only polygons overlapping the raster extent
    from shapely.geometry import box as shapely_box
    xmin, ymin, xmax, ymax = raster_native_extent(gt, W, H)
    rbox = shapely_box(xmin, ymin, xmax, ymax)

    gdf = gdf[gdf.geometry.intersects(rbox)].copy()
    _info(f'After spatial filter (intersects raster): {len(gdf)} features')

    return gdf


# ===========================================================================
# Argument parsing
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='run_fire_mapping.py',
        description='Batch fire mapping: VIIRS download → crop → accumulate → classify.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example
-------
  python batch_fire_mapping/run_fire_mapping.py          \\
      IN_HISTORICAL_FIRE_POLYGONS_SVW.shp                \\
      C11659/S2C_MSIL1C_20251014T192401_...20m.bin       \\
      --crop_buffer_px 100
        """,
    )

    # ---- Required ----
    p.add_argument('polygon_file',
                   help='Historical fire perimeters shapefile (.shp).  '
                        'Must have FIRE_NUMBE and FIRE_DATE columns.')
    p.add_argument('raster_file',
                   help='Sentinel-2 ENVI .bin raster.')

    # ---- Filtering ----
    p.add_argument('--year', type=int, default=None,
                   help='Only process fires from this FIRE_YEAR '
                        '(default: all years in the shapefile)')

    # ---- Output ----
    p.add_argument('--output_dir', default=None,
                   help='Root output directory (default: same directory as RASTER)')
    p.add_argument('--crop_buffer_px', type=int, default=100,
                   help='Extra pixels added on each side of the fire bounding '
                        'box when cropping the raster (default: 100)')

    # ---- VIIRS ----
    p.add_argument('--skip_download', action='store_true',
                   help='Skip downloading and shapifying VIIRS — go straight to mapping')
    p.add_argument('--shapify_workers', type=int, default=8,
                   help='Workers for shapify step (default: 8)')

    # ---- Pass-through to fire_mapping_cli.py ----
    p.add_argument('--sample_size',         type=int,   default=10_000)
    p.add_argument('--seed',                type=int,   default=123)
    p.add_argument('--embed_bands',         default=None,
                   help='1-indexed comma-separated band list for T-SNE '
                        '(default: all bands)')
    p.add_argument('--rf_n_estimators',     type=int,   default=100)
    p.add_argument('--rf_max_depth',        type=int,   default=15)
    p.add_argument('--rf_max_features',     default='sqrt')
    p.add_argument('--rf_random_state',     type=int,   default=42)
    p.add_argument('--controlled_ratio',    type=float, default=0.5)
    p.add_argument('--hdbscan_min_samples', type=int,   default=20)
    p.add_argument('--tsne_perplexity',     type=float, default=60.0)
    p.add_argument('--tsne_learning_rate',  type=float, default=200.0)
    p.add_argument('--tsne_max_iter',       type=int,   default=2000)
    p.add_argument('--tsne_init',           default='pca',
                   choices=['pca', 'random'])
    p.add_argument('--tsne_n_components',   type=int,   default=2)
    p.add_argument('--tsne_random_state',   type=int,   default=42)
    p.add_argument('--plot_downsample',     type=int,   default=2)

    return p


# ===========================================================================
# main
# ===========================================================================

def main(argv=None):
    args = _build_parser().parse_args(argv)

    raster_path  = os.path.abspath(args.raster_file)
    polygon_file = os.path.abspath(args.polygon_file)
    raster_dir   = os.path.dirname(raster_path)
    output_root  = os.path.abspath(args.output_dir) if args.output_dir else raster_dir

    if not os.path.exists(raster_path):
        sys.exit(f'ERROR: Raster not found: {raster_path}')
    if not os.path.exists(polygon_file):
        sys.exit(f'ERROR: Polygon file not found: {polygon_file}')

    os.makedirs(output_root, exist_ok=True)

    _box(
        'BATCH FIRE MAPPING',
        [
            f'Raster   : {raster_path}',
            f'Polygons : {polygon_file}',
            f'Buffer   : {args.crop_buffer_px} px',
            f'Output   : {output_root}',
        ],
    )

    # -----------------------------------------------------------------------
    # VIIRS save directory  (next to the raster, named <raster_name>_VIIRS)
    # -----------------------------------------------------------------------
    raster_basename = os.path.splitext(os.path.basename(raster_path))[0]
    viirs_save_dir  = os.path.join(raster_dir, f'{raster_basename}_VIIRS')
    viirs_shp_dir   = os.path.join(viirs_save_dir, 'VNP14IMG')
    os.makedirs(viirs_shp_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load polygons and determine download date range
    # -----------------------------------------------------------------------
    _box('Step 1 — Load and filter polygons')
    gdf = load_and_filter_polygons(
        polygon_file=polygon_file,
        raster_path=raster_path,
        year=args.year,
    )

    if gdf.empty:
        _warn('No matching polygons found.  Exiting.')
        sys.exit(0)

    _info(f'{len(gdf)} fire(s) to process.')

    # Download date range:
    #   start = Jan 1 of the earliest FIRE_YEAR in the filtered set
    #   end   = Dec 31 of the latest FIRE_YEAR
    fire_years_series = pd.to_numeric(gdf.get('FIRE_YEAR', pd.Series(dtype=int)),
                                      errors='coerce').dropna()
    if fire_years_series.empty:
        # Fallback: derive years from FIRE_DATE
        fire_dates = pd.to_datetime(gdf['FIRE_DATE'], errors='coerce')
        fire_years_series = fire_dates.dt.year.dropna()

    if fire_years_series.empty:
        sys.exit('ERROR: Cannot determine fire years from shapefile — '
                 'need FIRE_YEAR or parseable FIRE_DATE column.')

    min_year = int(fire_years_series.min())
    max_year = int(fire_years_series.max())
    dl_start = datetime.datetime(min_year, 1, 1)
    dl_end   = datetime.datetime(max_year, 12, 31)
    _info(f'Fire years in dataset: {min_year} – {max_year}')

    _info(f'Download range: {dl_start.date()} → {dl_end.date()}')

    # -----------------------------------------------------------------------
    # Download and shapify VIIRS
    # -----------------------------------------------------------------------
    if not args.skip_download:
        _box('Step 2 — Download VIIRS VNP14IMG')
        token = load_token()
        download_viirs(
            raster_path    = raster_path,
            start_dt       = dl_start,
            end_dt         = dl_end,
            token          = token,
            viirs_save_dir = viirs_save_dir,
        )

        _box('Step 3 — Shapify (nc → shp)')
        shapify_viirs(
            viirs_save_dir = viirs_save_dir,
            raster_path    = raster_path,
            workers        = args.shapify_workers,
        )
    else:
        _info('--skip_download set: skipping download and shapify steps.')

    # -----------------------------------------------------------------------
    # Load all VIIRS shapified data once (for spatial queries per fire)
    # -----------------------------------------------------------------------
    _box('Step 4 — Load VIIRS shapefiles for spatial queries')
    crs_wkt, gt, W, H = get_raster_info(raster_path)
    viirs_gdf = load_all_viirs(viirs_shp_dir, crs_wkt)

    # -----------------------------------------------------------------------
    # Build pass-through CLI args list
    # -----------------------------------------------------------------------
    cli_pass_args = [
        '--sample_size',         str(args.sample_size),
        '--seed',                str(args.seed),
        '--rf_n_estimators',     str(args.rf_n_estimators),
        '--rf_max_depth',        str(args.rf_max_depth),
        '--rf_max_features',     args.rf_max_features,
        '--rf_random_state',     str(args.rf_random_state),
        '--controlled_ratio',    str(args.controlled_ratio),
        '--hdbscan_min_samples', str(args.hdbscan_min_samples),
        '--tsne_perplexity',     str(args.tsne_perplexity),
        '--tsne_learning_rate',  str(args.tsne_learning_rate),
        '--tsne_max_iter',       str(args.tsne_max_iter),
        '--tsne_init',           args.tsne_init,
        '--tsne_n_components',   str(args.tsne_n_components),
        '--tsne_random_state',   str(args.tsne_random_state),
        '--plot_downsample',     str(args.plot_downsample),
    ]
    # Only forward --embed_bands if the user explicitly specified it;
    # otherwise the CLI defaults to all bands automatically.
    if args.embed_bands is not None:
        cli_pass_args += ['--embed_bands', args.embed_bands]

    cli_script = os.path.join(
        _REPO_ROOT, 'py', 'fire_mapping', 'fire_mapping_cli.py')

    if not os.path.isfile(cli_script):
        sys.exit(f'ERROR: fire_mapping_cli.py not found at {cli_script}')

    # -----------------------------------------------------------------------
    # Main loop — one fire at a time  (no parallelism: GPU is busy)
    # -----------------------------------------------------------------------
    _box(f'Step 5 — Processing {len(gdf)} fire(s)  (sequential)')

    results = {}
    for idx, row in gdf.iterrows():
        fire_numbe = str(row.get('FIRE_NUMBE', idx))
        out = process_fire(
            row            = row,
            viirs_gdf      = viirs_gdf,
            raster_path    = raster_path,
            raster_crs     = crs_wkt,
            raster_gt      = gt,
            raster_W       = W,
            raster_H       = H,
            output_root    = output_root,
            crop_buffer_px      = args.crop_buffer_px,
            cli_script     = cli_script,
            cli_pass_args  = cli_pass_args,
            viirs_shp_dir  = viirs_shp_dir,
        )
        results[fire_numbe] = out

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n_ok   = sum(1 for v in results.values() if v is not None)
    n_fail = len(results) - n_ok
    _box(
        'BATCH COMPLETE',
        [
            f'Processed : {n_ok} / {len(results)} fire(s)',
            f'Failed    : {n_fail}',
            f'Results   : {os.path.join(output_root, "fire_mapping_results")}',
        ],
    )

    if n_fail:
        for fire_numbe, v in results.items():
            if v is None:
                _warn(f'Failed: {fire_numbe}')


if __name__ == '__main__':
    main()
