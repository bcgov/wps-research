"""Year-wide VIIRS data bootstrap.

Downloads VNP14IMG .nc files for each year's full raster footprint over
the default seasonal window (year-03-01 .. year-10-30) and shapifies them
once at server boot. Per-fire prepare then only has to ``accumulate`` from
the shared shapefile dir — no per-fire LAADS calls, no per-fire shapify.

The shared dir lives at::

    <output_root_for_year>/_year_viirs/VNP14IMG/<YYYY>/<DDD>/

Each ``.nc`` granule is shapified to a ``.shp`` next to it, matching the
sibling polygon-driven package's layout so ``viirs.utils.accumulate`` can
glob both pre-existing and freshly-downloaded files identically.
"""

import datetime
import glob
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from osgeo import gdal, osr

gdal.UseExceptions()


def year_viirs_dir(state, year: int) -> str:
    """Return the year-wide VIIRS data dir for *year*."""
    out_dir = state.outdirs_by_year.get(year) or state.output_root
    return os.path.join(out_dir, '_year_viirs')


def year_shp_dir(state, year: int) -> str:
    """Return the dir to scan with ``viirs.utils.accumulate``."""
    return os.path.join(year_viirs_dir(state, year), 'VNP14IMG')


def default_window(year: int) -> tuple:
    """Return (start, end) datetimes for the year's default seasonal window,
    clamped so the upper bound never exceeds today (LAADS only has past data).
    """
    start = datetime.datetime(year, 1, 1)
    end = datetime.datetime(year, 12, 31)
    today = datetime.datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0)
    if end > today:
        end = today
    return start, end


def _wgs84_extent_of(raster_path: str):
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Cannot open raster: {raster_path}')
    try:
        gt = ds.GetGeoTransform()
        W, H = ds.RasterXSize, ds.RasterYSize
        crs_wkt = ds.GetProjection() or ''
    finally:
        ds = None

    xs = [gt[0], gt[0] + W * gt[1]]
    ys = [gt[3], gt[3] + H * gt[5]]
    xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)

    src = osr.SpatialReference()
    src.ImportFromWkt(crs_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    lons, lats = [], []
    for x, y in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
        lon, lat, _ = ct.TransformPoint(x, y)
        lons.append(lon)
        lats.append(lat)
    return min(lons), min(lats), max(lons), max(lats)


def _download_day(day: datetime.datetime, save_dir: str,
                  bbox_wgs84: tuple, token: str) -> int:
    """Download one day's granules into save_dir/VNP14IMG/<year>/<jday>/.
    Returns the count of .nc files in that day's dir after the call.
    Skips if the dir already has .nc files."""
    from viirs.utils.laads_data_download_v2 import sync as _sync

    west, south, east, north = bbox_wgs84
    jday = day.timetuple().tm_yday
    year = day.year
    target_dir = os.path.join(
        save_dir, 'VNP14IMG', f'{year:04d}', f'{jday:03d}')
    os.makedirs(target_dir, exist_ok=True)

    existing = glob.glob(os.path.join(target_dir, '*.nc'))
    if existing:
        return len(existing)

    url = (
        f'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?'
        f'products=VNP14IMG&'
        f'temporalRanges={year}-{jday}&'
        f'regions=%5BBBOX%5DN{north:.6f}%20S{south:.6f}'
        f'%20E{east:.6f}%20W{west:.6f}'
    )
    try:
        _sync(url, target_dir, token)
    except Exception as exc:
        sys.stderr.write(
            f'[year_viirs] download {day.date()}: {exc}\n')
    return len(glob.glob(os.path.join(target_dir, '*.nc')))


def download_year(year: int, raster_path: str, save_dir: str,
                  token: str, workers: int = 16,
                  start: datetime.datetime = None,
                  end: datetime.datetime = None) -> int:
    """Download .nc granules for the raster's full bbox + year window.
    Idempotent (skips days that already have .nc files). Returns total .nc count."""
    if start is None or end is None:
        ds, de = default_window(year)
        start = start or ds
        end = end or de
    if end < start:
        return 0

    bbox_wgs84 = _wgs84_extent_of(raster_path)

    days = []
    d = start
    while d <= end:
        days.append(d)
        d += datetime.timedelta(days=1)

    pending = []
    for day in days:
        jday = day.timetuple().tm_yday
        day_dir = os.path.join(
            save_dir, 'VNP14IMG', f'{day.year:04d}', f'{jday:03d}')
        if not glob.glob(os.path.join(day_dir, '*.nc')):
            pending.append(day)

    if not pending:
        return sum(
            len(glob.glob(os.path.join(
                save_dir, 'VNP14IMG', f'{d.year:04d}',
                f'{d.timetuple().tm_yday:03d}', '*.nc')))
            for d in days)

    completed = {'n': 0}
    print(f'      [{os.path.basename(raster_path)}] '
          f'{len(pending)} day(s) of VIIRS to download '
          f'(workers={workers}) ...')
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_download_day, d, save_dir, bbox_wgs84, token): d
            for d in pending
        }
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:
                sys.stderr.write(
                    f'[year_viirs] download error: {exc}\n')
            completed['n'] += 1
            if completed['n'] % 10 == 0 or completed['n'] == len(pending):
                print(f'        downloaded {completed["n"]}/{len(pending)}',
                      flush=True)
    return sum(
        len(glob.glob(os.path.join(
            save_dir, 'VNP14IMG', f'{d.year:04d}',
            f'{d.timetuple().tm_yday:03d}', '*.nc')))
        for d in days)


def shapify_year(save_dir: str, raster_path: str, workers: int = 8) -> int:
    """Shapify any .nc inside save_dir that doesn't yet have a sibling .shp.
    Idempotent. Returns count of .shp files now on disk."""
    from batch_fire_mapping.run_fire_mapping import shapify_viirs

    nc_files = sorted(glob.glob(
        os.path.join(save_dir, 'VNP14IMG', '**', '*.nc'), recursive=True))
    pending_nc = []
    for nc in nc_files:
        nc_dir = os.path.dirname(nc)
        if not glob.glob(os.path.join(nc_dir, '*.shp')):
            pending_nc.append(nc)
    if pending_nc:
        print(f'      [{os.path.basename(raster_path)}] '
              f'shapifying {len(pending_nc)} .nc granule(s) '
              f'(workers={workers}) ...')
        shapify_viirs(save_dir, raster_path, workers=workers)
    return len(glob.glob(os.path.join(
        save_dir, 'VNP14IMG', '**', '*.shp'), recursive=True))


# ---------------------------------------------------------------------------
# Year-wide spatial index (single GeoPackage)
# ---------------------------------------------------------------------------

def year_index_path(save_dir: str) -> str:
    """Path to the consolidated year-wide GeoPackage index."""
    return os.path.join(save_dir, 'year_index.gpkg')


def _index_is_fresh(index_path: str, shp_files: list) -> bool:
    if not os.path.isfile(index_path):
        return False
    manifest = index_path + '.manifest'
    try:
        with open(manifest) as fh:
            recorded = int(fh.read().strip())
    except (OSError, ValueError):
        return False
    if recorded != len(shp_files):
        return False
    try:
        idx_mtime = os.path.getmtime(index_path)
    except OSError:
        return False
    for shp in shp_files:
        try:
            if os.path.getmtime(shp) > idx_mtime:
                return False
        except OSError:
            return False
    return True


def build_year_index(save_dir: str, raster_path: str) -> str:
    """Consolidate every per-granule shapefile under *save_dir* into a single
    GeoPackage with a text ``det_dt`` column (compact YYYYMMDDHHMM, so date
    filters reduce to lexicographic comparisons regardless of GDAL/SQLite
    type coercion). The GPKG driver builds an R-tree on geometry so per-fire
    bbox queries skip irrelevant points without opening per-granule files.

    Idempotent: rebuilt only when the .shp count or any .shp's mtime
    diverges from the recorded manifest. Atomic write (tmp → rename).
    Returns the index path, or '' when there are no shapefiles to index.
    """
    import geopandas as gpd
    import pandas as pd
    from pathlib import Path
    from viirs.utils.accumulate import extract_datetime_from_filename

    shp_files = sorted(glob.glob(
        os.path.join(save_dir, 'VNP14IMG', '**', '*.shp'), recursive=True))
    index_path = year_index_path(save_dir)
    manifest_path = index_path + '.manifest'

    if not shp_files:
        for p in (index_path, manifest_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        return ''

    if _index_is_fresh(index_path, shp_files):
        return index_path

    print(f'      [{os.path.basename(raster_path)}] '
          f'building year index from {len(shp_files)} shapefiles ...')

    frames = []
    crs = None
    for shp in shp_files:
        dt = extract_datetime_from_filename(Path(shp).stem)
        if dt is None:
            continue
        try:
            gdf = gpd.read_file(shp)
        except Exception as exc:
            sys.stderr.write(f'[year_viirs] index read {shp}: {exc}\n')
            continue
        if gdf.empty:
            continue
        if crs is None and gdf.crs is not None:
            crs = gdf.crs
        gdf['det_dt'] = dt.strftime('%Y%m%d%H%M')
        frames.append(gdf)

    if not frames:
        for p in (index_path, manifest_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        return ''

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=crs)
    # GPKG reserves the lowercase ``fid`` column as the primary key; the
    # ``FID`` column geopandas pulls off shapefiles case-folds onto it on
    # write and collides across concatenated granules. Drop it.
    drop_cols = [c for c in combined.columns if c.lower() == 'fid']
    if drop_cols:
        combined = combined.drop(columns=drop_cols)

    # Keep the .gpkg suffix on the temp path so pyogrio's extension sniff
    # doesn't second-guess the explicit driver= argument.
    tmp_path = index_path + '.tmp.gpkg'
    if os.path.isfile(tmp_path):
        os.remove(tmp_path)
    combined.to_file(tmp_path, driver='GPKG', layer='viirs')
    os.replace(tmp_path, index_path)
    with open(manifest_path, 'w') as fh:
        fh.write(str(len(shp_files)))
    print(f'      [{os.path.basename(raster_path)}] '
          f'year index: {len(combined)} features → {index_path}')
    return index_path


def bootstrap_year(state, year: int, raster_path: str,
                   save_dir: str = None,
                   download_workers: int = 16,
                   shapify_workers: int = 8) -> dict:
    """Run download + shapify + index for a year's full raster footprint and
    default seasonal window. Idempotent. Returns counts."""
    if save_dir is None:
        save_dir = year_viirs_dir(state, year)
    os.makedirs(save_dir, exist_ok=True)

    n_nc = download_year(
        year, raster_path, save_dir, state.laads_token,
        workers=download_workers)
    n_shp = shapify_year(save_dir, raster_path, workers=shapify_workers)
    index_path = build_year_index(save_dir, raster_path)
    return {'n_nc': n_nc, 'n_shp': n_shp, 'save_dir': save_dir,
            'index_path': index_path}


def bootstrap_all_years(state) -> dict:
    """Run bootstrap_year for every (year, raster) in state.rasters_by_year.
    Sequential — printing progress is more useful than parallelism here.
    Returns {year: result_dict}."""
    out = {}
    rasters = state.rasters_by_year or {}
    for year in sorted(rasters):
        out[year] = bootstrap_year(
            state, year, rasters[year],
            download_workers=state.viirs_download_workers or 16,
            shapify_workers=state.viirs_shapify_workers or 8)
    return out
