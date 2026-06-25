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
import json
import os
import shutil
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


def migrate_stale_viirs_data(out_root: str, active_dirs: set) -> dict:
    """Move already-downloaded VIIRS day-folders from OTHER
    <stem>_mapping_results/_year_viirs/VNP14IMG/<year>/<jday>/
    directories under out_root into the corresponding location inside
    whichever of *active_dirs* is in use today.

    Each daily stack file (e.g. 20260622_stack.bin -> 20260623_stack.bin
    tomorrow) gets its own freshly-named _mapping_results directory --
    without this, every day's already-downloaded .nc granules would be
    silently orphaned in yesterday's folder and re-downloaded from
    scratch on every restart, for no reason (the VNP14IMG data for a
    given calendar day doesn't change).

    Deliberately simple, NOT footprint-aware: a day-folder is moved if
    the active directory doesn't already have one for that
    year/jday. No attempt is made to verify the source and
    destination rasters cover the same extent -- per the explicit
    instruction this was built against, this is left for a future
    pass. Existing data at the destination is never overwritten.

    active_dirs may contain more than one directory (multi-year
    setups) -- a moved day only needs to land in ANY one of them; in
    practice each year's stem is independent so this rarely matters,
    but the check guards against ever leaving migratable data behind
    just because it happened to check against the wrong active year's
    directory first.

    Returns {'moved': N, 'skipped_existing': N, 'errors': [...]}.
    """
    result = {'moved': 0, 'skipped_existing': 0, 'errors': []}

    active_dirs = {os.path.abspath(d) for d in active_dirs}
    if not os.path.isdir(out_root):
        return result

    candidate_roots = []
    for name in sorted(os.listdir(out_root)):
        if not name.endswith('_mapping_results'):
            continue
        full = os.path.join(out_root, name)
        if os.path.abspath(full) in active_dirs or not os.path.isdir(full):
            continue
        candidate_roots.append(full)

    if not candidate_roots:
        return result

    for stale_root in candidate_roots:
        stale_viirs = os.path.join(stale_root, '_year_viirs', 'VNP14IMG')
        if not os.path.isdir(stale_viirs):
            continue
        for year_name in sorted(os.listdir(stale_viirs)):
            year_dir = os.path.join(stale_viirs, year_name)
            if not os.path.isdir(year_dir):
                continue
            for jday_name in sorted(os.listdir(year_dir)):
                src_day_dir = os.path.join(year_dir, jday_name)
                if not os.path.isdir(src_day_dir):
                    continue
                src_nc = glob.glob(os.path.join(src_day_dir, '*.nc'))
                if not src_nc:
                    continue  # nothing worth moving for this day

                moved_this_day = False
                for active_dir in active_dirs:
                    dst_day_dir = os.path.join(
                        active_dir, '_year_viirs', 'VNP14IMG',
                        year_name, jday_name)
                    dst_nc = glob.glob(os.path.join(dst_day_dir, '*.nc'))
                    if dst_nc:
                        result['skipped_existing'] += 1
                        moved_this_day = True  # already covered, don't
                                               # also move it elsewhere
                        break
                    try:
                        os.makedirs(dst_day_dir, exist_ok=True)
                        for f in os.listdir(src_day_dir):
                            shutil.move(
                                os.path.join(src_day_dir, f),
                                os.path.join(dst_day_dir, f))
                        result['moved'] += 1
                        moved_this_day = True
                        break
                    except OSError as exc:
                        result['errors'].append(
                            f'{src_day_dir} -> {dst_day_dir}: {exc}')
                if not moved_this_day:
                    result['errors'].append(
                        f'{src_day_dir}: no active directory accepted it')
    return result


def check_laads_credentials(token: str, timeout_s: float = 15.0) -> dict:
    """One small, fast request to LAADS's discovery API to find out
    WHY VIIRS downloads might be failing, before running the full
    (slow, many-day) bootstrap. Distinguishes:

        'ok'           - got a real response; token is valid
        'bad_token'    - HTTP 401/403: token is invalid/expired
        'http_error'   - some other non-2xx HTTP status from the
                          server -- server-side issue, not the token
        'unreachable'  - timeout / DNS / connection failure -- network
                          issue, not the token, possibly not even
                          LAADS's fault (e.g. local network down)
        'unknown'      - response didn't look like valid JSON; can't
                          tell, but at least the server answered

    Returns {'status': ..., 'detail': str, 'http_code': int|None}.
    This does NOT replace per-day error handling in download_year --
    it's a one-shot "is the credential/connection even viable" check
    so a startup log line can say something more useful than "VIIRS
    bootstrap failed" with no indication of which of these it was.
    """
    import json
    import urllib.error
    import urllib.request

    # A small, cheap, near-certain-to-exist query: a single recent day,
    # a tiny bbox -- this just needs *a* response, not real data.
    probe_url = (
        'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?'
        'products=VNP14IMG&temporalRanges=2024-1&'
        'regions=%5BBBOX%5DN50.0%20S49.9%20E-122.9%20W-123.0'
    )
    headers = {'Authorization': f'Bearer {token}'} if token else {}

    try:
        req = urllib.request.Request(probe_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            code = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        code = exc.code
        if code in (401, 403):
            return {'status': 'bad_token',
                    'detail': f'HTTP {code} from LAADS -- token is '
                              f'missing, invalid, or expired.',
                    'http_code': code}
        return {'status': 'http_error',
                'detail': f'HTTP {code} from LAADS (not a token '
                          f'problem -- likely a server-side issue).',
                'http_code': code}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {'status': 'unreachable',
                'detail': f'Could not reach LAADS at all: {exc} -- '
                          f'check network connectivity, not the token.',
                'http_code': None}

    try:
        json.loads(raw)
    except (ValueError, TypeError):
        return {'status': 'unknown',
                'detail': f'Got HTTP {code} but the response was not '
                          f'valid JSON -- LAADS answered, but something '
                          f'unexpected came back.',
                'http_code': code}

    return {'status': 'ok',
            'detail': f'HTTP {code}, valid JSON response -- token and '
                      f'connectivity both appear fine.',
            'http_code': code}


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


_FOOTPRINT_TOLERANCE_DEG = 1e-4  # ~10m at these latitudes; generous
                                  # enough to absorb reprojection
                                  # float noise between two computations
                                  # of the "same" extent, tight enough
                                  # to reject a genuinely different one.


def _footprint_marker_path(save_dir: str) -> str:
    return os.path.join(save_dir, '_footprint.json')


def _write_footprint_marker(save_dir: str, wgs84_bbox: tuple) -> None:
    """Best-effort -- a failure here must never block bootstrap."""
    try:
        with open(_footprint_marker_path(save_dir), 'w', encoding='utf-8') as f:
            json.dump({'wgs84_bbox': list(wgs84_bbox)}, f)
    except OSError:
        pass


def _read_footprint_marker(save_dir: str) -> tuple | None:
    path = _footprint_marker_path(save_dir)
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        bbox = data.get('wgs84_bbox')
        if bbox and len(bbox) == 4:
            return tuple(bbox)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _footprints_match(a: tuple, b: tuple,
                      tol: float = _FOOTPRINT_TOLERANCE_DEG) -> bool:
    return all(abs(a[i] - b[i]) <= tol for i in range(4))


def migrate_viirs_from_previous_stacks(state, year: int,
                                       raster_path: str) -> dict:
    """Look for VIIRS data already downloaded under a DIFFERENT
    stack-date's output directory (e.g. yesterday's
    20260622_stack_mapping_results/_year_viirs/...) whose footprint
    matches the CURRENT raster, and move (not copy) any day-folders
    not already present in the current stack's _year_viirs/VNP14IMG/
    tree -- so a fresh daily stack file doesn't force a full
    redownload of a whole year's VIIRS history it already has.

    Each stack's _mapping_results dir gets a per-output-dir
    _year_viirs/_footprint.json the first time it's bootstrapped here,
    recording its WGS84 extent -- comparing against that marker
    (rather than re-opening the OLD raster file) is necessary because
    old /ram/<date>_stack.bin files are routinely deleted once a new
    stack is built (see build_and_serve_stack.py's cleanup step), so
    the old raster usually no longer exists to re-derive its extent
    from. A directory with no marker (predates this feature, or the
    marker write failed) is skipped -- conservative: no migration
    rather than guessing.

    Returns {'migrated_days': N, 'skipped_existing': N, 'sources': [...]}
    for startup logging. Best-effort throughout: any single source
    directory or day-folder that fails to process is skipped with a
    warning rather than aborting the whole migration.
    """
    result = {'migrated_days': 0, 'skipped_existing': 0, 'sources': []}

    save_dir = year_viirs_dir(state, year)
    os.makedirs(save_dir, exist_ok=True)

    try:
        current_bbox = _wgs84_extent_of(raster_path)
    except Exception as exc:
        sys.stderr.write(
            f'[viirs-migrate] could not compute footprint for '
            f'{raster_path}: {exc} -- skipping migration\n')
        return result

    # Stamp our own marker now (idempotent -- overwrites with the
    # same value on every run) so a FUTURE stack date can find us.
    _write_footprint_marker(save_dir, current_bbox)

    out_root = state.output_root
    if not out_root or not os.path.isdir(out_root):
        return result

    current_shp_dir = os.path.join(save_dir, 'VNP14IMG')

    for entry in sorted(os.listdir(out_root)):
        candidate_dir = os.path.join(out_root, entry, '_year_viirs')
        if not os.path.isdir(candidate_dir):
            continue
        if os.path.abspath(candidate_dir) == os.path.abspath(save_dir):
            continue

        other_bbox = _read_footprint_marker(candidate_dir)
        if other_bbox is None:
            sys.stderr.write(
                f'[viirs-migrate] {entry}: no footprint marker, '
                f'skipping (predates this feature, or raster no '
                f'longer available to confirm a match)\n')
            continue
        if not _footprints_match(current_bbox, other_bbox):
            continue

        other_shp_dir = os.path.join(candidate_dir, 'VNP14IMG')
        if not os.path.isdir(other_shp_dir):
            continue

        sys.stderr.write(
            f'[viirs-migrate] {entry}: footprint matches current stack '
            f'-- checking for VIIRS data to reuse ...\n')
        moved_here = 0
        for yyyy in sorted(os.listdir(other_shp_dir)):
            src_year_dir = os.path.join(other_shp_dir, yyyy)
            if not os.path.isdir(src_year_dir):
                continue
            for jday in sorted(os.listdir(src_year_dir)):
                src_day_dir = os.path.join(src_year_dir, jday)
                if not os.path.isdir(src_day_dir):
                    continue
                dst_day_dir = os.path.join(current_shp_dir, yyyy, jday)
                if glob.glob(os.path.join(dst_day_dir, '*.nc')):
                    result['skipped_existing'] += 1
                    continue
                try:
                    os.makedirs(os.path.dirname(dst_day_dir), exist_ok=True)
                    if os.path.isdir(dst_day_dir):
                        # Empty dir from a prior makedirs elsewhere --
                        # remove so shutil.move can place the real one.
                        if not os.listdir(dst_day_dir):
                            os.rmdir(dst_day_dir)
                        else:
                            result['skipped_existing'] += 1
                            continue
                    shutil.move(src_day_dir, dst_day_dir)
                    moved_here += 1
                    result['migrated_days'] += 1
                except OSError as exc:
                    sys.stderr.write(
                        f'[viirs-migrate] {entry}/{yyyy}/{jday}: '
                        f'move failed: {exc}\n')
        if moved_here:
            sys.stderr.write(
                f'[viirs-migrate] {entry}: reused {moved_here} '
                f'day(s) of VIIRS data -- will not be redownloaded\n')
            result['sources'].append(entry)

    return result


def bootstrap_year(state, year: int, raster_path: str,
                   save_dir: str = None,
                   download_workers: int = 16,
                   shapify_workers: int = 8) -> dict:
    """Run download + shapify + index for a year's full raster footprint and
    default seasonal window. Idempotent. Returns counts."""
    if save_dir is None:
        save_dir = year_viirs_dir(state, year)
    os.makedirs(save_dir, exist_ok=True)

    migration = migrate_viirs_from_previous_stacks(state, year, raster_path)
    if migration['migrated_days']:
        print(f'      [{os.path.basename(raster_path)}] reused '
              f"{migration['migrated_days']} VIIRS day(s) already "
              f"downloaded under a previous stack date with a matching "
              f"footprint ({', '.join(migration['sources'])})")

    n_nc = download_year(
        year, raster_path, save_dir, state.laads_token,
        workers=download_workers)
    n_shp = shapify_year(save_dir, raster_path, workers=shapify_workers)
    index_path = build_year_index(save_dir, raster_path)
    return {'n_nc': n_nc, 'n_shp': n_shp, 'save_dir': save_dir,
            'index_path': index_path, 'migration': migration}


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
