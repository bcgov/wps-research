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

Retry policy — zero tolerance for missed data
-----------------------------------------------
Every unit of work here (a day's download, the shapify pass, the index
build) retries indefinitely until it succeeds. Nothing is ever silently
skipped or treated as "done" on failure. The schedule, per attempt
cycle, is:

    * Attempts 1-100  : retried immediately, no wait between them.
    * Attempts 101-125: exponential backoff starting at 5s, doubling
      each attempt (5, 10, 20, 40, 80, 160, 300, 300, ... seconds),
      capped at 300s (5 min), with up to +/-20% jitter so many
      concurrently-failing days don't retry in lockstep.
    * After attempt 125 the cycle repeats from attempt 1 (another 100
      immediate retries, then another 25 backoff-spaced ones),
      forever, until that unit of work succeeds.

This means a transient blip clears almost immediately (most of the
first 100 attempts), while a sustained outage (LAADS down for hours)
backs off to a gentle 5-minute cadence instead of hammering the API,
but NEVER gives up.

Each retrying unit checks ``shutdown_event`` between attempts/waits so
a server shutdown (Ctrl-C) doesn't hang forever waiting on retries.
"""

import datetime
import glob
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from osgeo import gdal, osr

gdal.UseExceptions()

# Set by main() at startup so indefinitely-retrying loops can be told
# to stop promptly on Ctrl-C / shutdown rather than blocking process
# exit. threading.Event(), checked between attempts and during waits.
shutdown_event = threading.Event()

# Retry schedule constants (see module docstring for the full policy).
_IMMEDIATE_ATTEMPTS = 100
_BACKOFF_ATTEMPTS = 25
_BACKOFF_BASE_S = 5.0
_BACKOFF_CAP_S = 300.0
_BACKOFF_JITTER = 0.20
_CYCLE_LENGTH = _IMMEDIATE_ATTEMPTS + _BACKOFF_ATTEMPTS


def _wait_seconds_for_attempt(attempt_in_cycle: int) -> float:
    """attempt_in_cycle is 1-based within one 125-attempt cycle.
    Returns the wait BEFORE this attempt (0 for the first 100)."""
    if attempt_in_cycle <= _IMMEDIATE_ATTEMPTS:
        return 0.0
    backoff_step = attempt_in_cycle - _IMMEDIATE_ATTEMPTS  # 1..25
    base = min(_BACKOFF_BASE_S * (2 ** (backoff_step - 1)), _BACKOFF_CAP_S)
    jitter = base * _BACKOFF_JITTER
    return max(0.0, base + random.uniform(-jitter, jitter))


def _interruptible_sleep(seconds: float) -> bool:
    """Sleep up to *seconds*, waking early if shutdown_event is set.
    Returns False if shutdown was requested (caller should bail out)."""
    if seconds <= 0:
        return not shutdown_event.is_set()
    return not shutdown_event.wait(timeout=seconds)


def retry_forever(fn, description: str, on_attempt=None):
    """Call fn() until it returns truthy / doesn't raise, retrying
    indefinitely on the schedule documented at module level. fn should
    return a truthy "done" value, or raise/return falsy to indicate
    "not yet done, try again". Returns fn()'s truthy result, or None
    if shutdown was requested mid-retry.

    on_attempt(attempt_number, exc_or_None) is called after every
    attempt (success or failure) for progress reporting / persisted
    retry-state bookkeeping.
    """
    attempt = 0
    while True:
        if shutdown_event.is_set():
            return None
        attempt += 1
        attempt_in_cycle = ((attempt - 1) % _CYCLE_LENGTH) + 1
        exc = None
        result = None
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 - intentionally broad: retry forever
            exc = e
        if on_attempt is not None:
            try:
                on_attempt(attempt, exc)
            except Exception:
                pass  # progress reporting must never break the retry loop
        if exc is None and result:
            return result
        if exc is not None:
            sys.stderr.write(
                f'[year_viirs] {description}: attempt {attempt} failed: '
                f'{exc}\n')
        wait_s = _wait_seconds_for_attempt(attempt_in_cycle)
        if wait_s > 0:
            sys.stderr.write(
                f'[year_viirs] {description}: retrying in '
                f'{wait_s:.0f}s (attempt {attempt}) ...\n')
        if not _interruptible_sleep(wait_s):
            return None


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


def _retry_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, 'retry_state.json')


def _load_retry_state(save_dir: str) -> dict:
    path = _retry_state_path(save_dir)
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


_retry_state_lock = threading.Lock()


def _update_retry_state(save_dir: str, day_key: str, attempt: int,
                         exc) -> None:
    """Record this attempt's outcome for day_key (yyyy-jday) into
    retry_state.json. Best-effort -- a failure here must never break
    the actual retry loop, so all errors are swallowed."""
    try:
        with _retry_state_lock:
            state_dict = _load_retry_state(save_dir)
            if exc is None:
                state_dict.pop(day_key, None)
            else:
                state_dict[day_key] = {
                    'attempts': attempt,
                    'last_error': str(exc),
                    'last_attempt_at': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                }
            tmp = _retry_state_path(save_dir) + f'.{os.getpid()}.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2, sort_keys=True)
            os.replace(tmp, _retry_state_path(save_dir))
    except Exception:
        pass


def _download_day(day: datetime.datetime, save_dir: str,
                  bbox_wgs84: tuple, token: str) -> int:
    """Download one day's granules into save_dir/VNP14IMG/<year>/<jday>/.
    Retries indefinitely (see module docstring) until at least one
    .nc file exists for that day, or shutdown is requested. Returns
    the count of .nc files in that day's dir, or 0 only if shutdown
    interrupted the retry loop before success."""
    from viirs.utils.laads_data_download_v2 import sync as _sync

    west, south, east, north = bbox_wgs84
    jday = day.timetuple().tm_yday
    year = day.year
    day_key = f'{year:04d}-{jday:03d}'
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

    def _attempt():
        _sync(url, target_dir, token)
        found = glob.glob(os.path.join(target_dir, '*.nc'))
        # _sync() may "succeed" (no exception) yet still have fetched
        # nothing for a day with genuinely zero granules available --
        # but LAADS always has *something* for a covered bbox/date
        # within this app's supported range, so an empty result after
        # a clean call is treated as not-yet-done and retried, same as
        # an exception. This is the zero-tolerance behaviour: silence
        # is not success.
        return len(found)

    def _on_attempt(attempt, exc):
        _update_retry_state(save_dir, day_key, attempt, exc)

    n = retry_forever(_attempt, f'download {day.date()}',
                       on_attempt=_on_attempt)
    if n is None:
        # Shutdown requested mid-retry -- report whatever's on disk.
        return len(glob.glob(os.path.join(target_dir, '*.nc')))
    return n


def download_year(year: int, raster_path: str, save_dir: str,
                  token: str, workers: int = 16,
                  start: datetime.datetime = None,
                  end: datetime.datetime = None,
                  progress_cb=None) -> int:
    """Download .nc granules for the raster's full bbox + year window.
    Idempotent (skips days that already have .nc files). Every day
    retries indefinitely until it succeeds or shutdown is requested
    (see module docstring) -- this never gives up on a day and never
    returns having silently skipped one. Returns total .nc count.

    progress_cb(done, total), if given, is called after each day
    completes (success or shutdown-interrupted)."""
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
        if progress_cb:
            progress_cb(len(days), len(days))
        return sum(
            len(glob.glob(os.path.join(
                save_dir, 'VNP14IMG', f'{d.year:04d}',
                f'{d.timetuple().tm_yday:03d}', '*.nc')))
            for d in days)

    completed = {'n': 0}
    n_already_done = len(days) - len(pending)
    print(f'      [{os.path.basename(raster_path)}] '
          f'{len(pending)} day(s) of VIIRS to download '
          f'(workers={workers}) ...')
    if progress_cb:
        progress_cb(n_already_done, len(days))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_download_day, d, save_dir, bbox_wgs84, token): d
            for d in pending
        }
        for fut in as_completed(futs):
            fut.result()  # _download_day no longer raises -- it
                          # retries forever internally; this just
                          # surfaces a genuine bug if one slips through
            completed['n'] += 1
            if completed['n'] % 10 == 0 or completed['n'] == len(pending):
                print(f'        downloaded {completed["n"]}/{len(pending)}',
                      flush=True)
            if progress_cb:
                progress_cb(n_already_done + completed['n'], len(days))
    return sum(
        len(glob.glob(os.path.join(
            save_dir, 'VNP14IMG', f'{d.year:04d}',
            f'{d.timetuple().tm_yday:03d}', '*.nc')))
        for d in days)


def shapify_year(save_dir: str, raster_path: str, workers: int = 8) -> int:
    """Shapify any .nc inside save_dir that doesn't yet have a sibling .shp.
    Idempotent. Retries indefinitely on failure (see module docstring)
    rather than aborting the bootstrap. Returns count of .shp files
    now on disk."""
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

        def _attempt():
            shapify_viirs(save_dir, raster_path, workers=workers)
            still_pending = [
                nc for nc in pending_nc
                if not glob.glob(os.path.join(os.path.dirname(nc), '*.shp'))
            ]
            return not still_pending  # True only once every .nc has a .shp

        retry_forever(_attempt, 'shapify_year')
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
    Retries indefinitely on failure (see module docstring) rather than
    aborting the bootstrap. Returns the index path, or '' when there
    are no shapefiles to index.
    """
    result = retry_forever(
        lambda: _build_year_index_once(save_dir, raster_path) or True,
        'build_year_index')
    # _build_year_index_once returns '' (falsy) when there's nothing to
    # index, which is a legitimate outcome, not a failure -- retry_forever
    # would otherwise keep retrying it forever. The `or True` above makes
    # an empty-but-successful result count as "done"; recompute the real
    # path/'' to return to the caller.
    return _index_or_empty(save_dir)


def _index_or_empty(save_dir: str) -> str:
    index_path = year_index_path(save_dir)
    return index_path if os.path.isfile(index_path) else ''


def _build_year_index_once(save_dir: str, raster_path: str) -> str:
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
    default seasonal window. Idempotent. Every step retries indefinitely
    on failure (see module docstring) -- this function only returns
    once everything has genuinely succeeded, or shutdown was requested.
    Updates state.startup_progress as it goes. Returns counts."""
    if save_dir is None:
        save_dir = year_viirs_dir(state, year)
    os.makedirs(save_dir, exist_ok=True)

    def _dl_progress(done, total):
        state.startup_progress = {
            'stage': 'downloading',
            'year': year,
            'detail': f'{done}/{total} day(s) of VIIRS data',
        }

    n_nc = download_year(
        year, raster_path, save_dir, state.laads_token,
        workers=download_workers, progress_cb=_dl_progress)

    if shutdown_event.is_set():
        return {'n_nc': n_nc, 'n_shp': 0, 'save_dir': save_dir,
                'index_path': ''}

    state.startup_progress = {
        'stage': 'shapifying',
        'year': year,
        'detail': 'converting downloaded granules',
    }
    n_shp = shapify_year(save_dir, raster_path, workers=shapify_workers)

    if shutdown_event.is_set():
        return {'n_nc': n_nc, 'n_shp': n_shp, 'save_dir': save_dir,
                'index_path': ''}

    state.startup_progress = {
        'stage': 'indexing',
        'year': year,
        'detail': 'building spatial index',
    }
    index_path = build_year_index(save_dir, raster_path)
    return {'n_nc': n_nc, 'n_shp': n_shp, 'save_dir': save_dir,
            'index_path': index_path}


def bootstrap_all_years(state) -> dict:
    """Run bootstrap_year for every (year, raster) in state.rasters_by_year.
    Sequential — printing progress is more useful than parallelism here.
    Stops between years if shutdown_event is set. Returns
    {year: result_dict}."""
    out = {}
    rasters = state.rasters_by_year or {}
    for year in sorted(rasters):
        if shutdown_event.is_set():
            break
        out[year] = bootstrap_year(
            state, year, rasters[year],
            download_workers=state.viirs_download_workers or 16,
            shapify_workers=state.viirs_shapify_workers or 8)
    return out
