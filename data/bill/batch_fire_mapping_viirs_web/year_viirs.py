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
    """Blindly move every .nc file from OTHER
    <stem>_mapping_results/_year_viirs/VNP14IMG/<year>/<jday>/
    directories under out_root into the corresponding location inside
    whichever of *active_dirs* is in use today. Shapefiles (.shp and
    siblings) are deliberately left behind in the old folder -- see
    purge_active_shapefiles(), which deletes and regenerates them for
    the active dir instead, since this migration alone doesn't
    guarantee every .nc has been (re)shapified yet.

    Each daily stack file (e.g. 20260622_stack.bin -> 20260623_stack.bin
    tomorrow) gets its own freshly-named _mapping_results directory --
    without this, every day's already-downloaded .nc granules would be
    silently orphaned in yesterday's folder and re-downloaded from
    scratch on every restart, for no reason (the VNP14IMG data for a
    given calendar day doesn't change).

    Blind overwrite: if a destination .nc with the same filename
    already exists, it's replaced unconditionally (no footprint
    checking, per instruction) -- but if the two files' contents
    differ, a warning is printed first so a real discrepancy doesn't
    pass silently.

    Returns {'moved': N, 'overwritten': N, 'overwritten_mismatched':
    N, 'errors': [...]}.
    """
    result = {'moved': 0, 'overwritten': 0, 'overwritten_mismatched': 0,
              'errors': []}

    active_dirs = sorted({os.path.abspath(d) for d in active_dirs})
    if not active_dirs or not os.path.isdir(out_root):
        return result
    # Single destination root for .nc migration -- with multiple
    # active dirs (multi-year), use the first; in practice each
    # year's stem is independent so this is just a stable choice.
    dest_root = active_dirs[0]

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
        for nc_path in glob.glob(
                os.path.join(stale_viirs, '**', '*.nc'), recursive=True):
            rel = os.path.relpath(nc_path, stale_viirs)
            dst_path = os.path.join(
                dest_root, '_year_viirs', 'VNP14IMG', rel)
            try:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                if os.path.isfile(dst_path):
                    same = _files_identical(nc_path, dst_path)
                    if not same:
                        sys.stderr.write(
                            f'[year_viirs] WARNING: overwriting '
                            f'{dst_path} with different content from '
                            f'{nc_path} (sizes/bytes differ) -- '
                            f'stomping as instructed.\n')
                        result['overwritten_mismatched'] += 1
                    result['overwritten'] += 1
                    os.remove(dst_path)
                shutil.move(nc_path, dst_path)
                result['moved'] += 1
            except OSError as exc:
                result['errors'].append(f'{nc_path} -> {dst_path}: {exc}')
    return result


def _files_identical(path_a: str, path_b: str) -> bool:
    """Cheap content comparison: size first, then bytes if sizes
    match. Good enough to decide whether an overwrite is silent
    (truly identical) or worth a warning (different)."""
    try:
        if os.path.getsize(path_a) != os.path.getsize(path_b):
            return False
        with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
            while True:
                ca = fa.read(1 << 20)
                cb = fb.read(1 << 20)
                if ca != cb:
                    return False
                if not ca:
                    return True
    except OSError:
        return False  # can't confirm identical -- treat as mismatch


def purge_active_shapefiles(active_dirs: set) -> int:
    """Delete every VNP*.shp and its sidecar components (.cpg, .dbf,
    .prj, .shx) under each active dir's _year_viirs/VNP14IMG tree, so
    shapify_year() re-derives shapefiles from .nc on this run -- this
    is what makes freshly-migrated/newly-downloaded .nc granules
    actually get (re)converted, since shapify_year() only shapifies a
    .nc that doesn't already have a sibling .shp. Returns count of
    files deleted.
    """
    removed = 0
    for active_dir in {os.path.abspath(d) for d in active_dirs}:
        viirs_dir = os.path.join(active_dir, '_year_viirs', 'VNP14IMG')
        if not os.path.isdir(viirs_dir):
            continue
        for ext in ('shp', 'cpg', 'dbf', 'prj', 'shx'):
            for f in glob.glob(os.path.join(
                    viirs_dir, '**', f'VNP*.{ext}'), recursive=True):
                try:
                    os.remove(f)
                    removed += 1
                except OSError as exc:
                    sys.stderr.write(
                        f'[year_viirs] WARNING: could not remove '
                        f'{f}: {exc}\n')
    return removed


def check_laads_credentials(token: str, timeout_s: float = 15.0,
                            log_dir: str = None) -> dict:
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

    If log_dir is given, the request/response is logged there as a
    .http_request / .http_response file pair (this check isn't tied
    to a specific day, so it belongs one level up from the day
    folders -- the year dir).
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
    request_text = f'GET {probe_url}\n' + '\n'.join(
        f'{k}: {v}' for k, v in headers.items())

    try:
        req = urllib.request.Request(probe_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            code = resp.status
            raw = resp.read()
        if log_dir:
            response_text = (
                f'HTTP {code}\n'
                + '\n'.join(f'{k}: {v}' for k, v in resp.headers.items())
                + '\n\n' + raw.decode('utf-8', errors='replace'))
            _log_http_pair(log_dir, request_text, response_text)
    except urllib.error.HTTPError as exc:
        code = exc.code
        if log_dir:
            try:
                body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                body = ''
            _log_http_pair(log_dir, request_text,
                           f'HTTP {code}\n\n{body}')
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
        if log_dir:
            _log_http_pair(log_dir, request_text, f'EXCEPTION: {exc}')
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


def _log_http_pair(log_dir: str, request_text: str,
                   response_text: str) -> None:
    """Write one matched .http_request / .http_response file pair into
    log_dir. Filenames: yyyymmddhhmmss_<uuid>.http_request/.http_response
    -- timestamp for sorting, uuid guarantees no collision between
    concurrent requests."""
    import uuid
    try:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        uid = uuid.uuid4().hex[:12]
        base = os.path.join(log_dir, f'{ts}_{uid}')
        with open(base + '.http_request', 'w', encoding='utf-8') as f:
            f.write(request_text)
        with open(base + '.http_response', 'w', encoding='utf-8') as f:
            f.write(response_text)
    except OSError as exc:
        sys.stderr.write(f'[year_viirs] WARNING: could not write HTTP '
                         f'log pair in {log_dir}: {exc}\n')


def _log_curl_pair(log_dir: str, request_text: str,
                   response_text: str) -> None:
    """Write one matched .curl_request / .curl_response file pair into
    log_dir — mirrors _log_http_pair but uses different extensions so
    curl traffic is distinguishable from urllib traffic at a glance."""
    import uuid
    try:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        uid = uuid.uuid4().hex[:12]
        base = os.path.join(log_dir, f'{ts}_{uid}')
        with open(base + '.curl_request', 'w', encoding='utf-8') as f:
            f.write(request_text)
        with open(base + '.curl_response', 'w', encoding='utf-8') as f:
            f.write(response_text)
    except OSError as exc:
        sys.stderr.write(f'[year_viirs] WARNING: could not write curl '
                         f'log pair in {log_dir}: {exc}\n')


def _install_http_logging(log_dir: str) -> None:
    """Monkey-patch urllib.request.urlopen (used by
    viirs.utils.laads_data_download_v2.geturl) so every HTTP
    request/response it makes gets logged as a file pair in log_dir.
    Only safe to call inside the forked subprocess in
    _sync_in_subprocess -- it permanently patches urlopen in whatever
    process calls it."""
    import urllib.request

    _orig_urlopen = urllib.request.urlopen

    def _logging_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        headers = (dict(req.headers) if hasattr(req, 'headers') else {})
        request_text = f'GET {url}\n' + '\n'.join(
            f'{k}: {v}' for k, v in headers.items())
        try:
            resp = _orig_urlopen(req, *args, **kwargs)
            body = resp.read()
            response_text = (
                f'HTTP {resp.status}\n'
                + '\n'.join(f'{k}: {v}' for k, v in resp.headers.items())
                + '\n\n' + body.decode('utf-8', errors='replace'))
            _log_http_pair(log_dir, request_text, response_text)
            resp_replay = type(resp)
            import io
            replay = io.BytesIO(body)
            replay.status = resp.status
            replay.headers = resp.headers
            return replay
        except Exception as exc:
            response_text = f'EXCEPTION: {exc}'
            _log_http_pair(log_dir, request_text, response_text)
            raise

    urllib.request.urlopen = _logging_urlopen


def _install_curl_logging(log_dir: str) -> None:
    """Monkey-patch the third-party module's getcURL so every curl
    request/response is logged as .curl_request / .curl_response.
    Only safe inside the forked subprocess."""
    try:
        import viirs.utils.laads_data_download_v2 as _ldl
    except ImportError:
        return
    if not hasattr(_ldl, 'getcURL'):
        return
    _orig_getcURL = _ldl.getcURL

    def _logging_getcURL(url, tok, *args, **kwargs):
        request_text = f'CURL {url}\nAuthorization: Bearer {tok[:20]}...'
        try:
            result = _orig_getcURL(url, tok, *args, **kwargs)
            if result is None:
                response_text = 'CURL RETURNED None (failed)'
            else:
                body_preview = str(result)[:4000]
                response_text = f'CURL OK (len={len(str(result))})\n\n{body_preview}'
            _log_curl_pair(log_dir, request_text, response_text)
            return result
        except Exception as exc:
            _log_curl_pair(log_dir, request_text, f'CURL EXCEPTION: {exc}')
            raise

    _ldl.getcURL = _logging_getcURL


def _install_curl_primary_order() -> None:
    """Monkey-patch the third-party module's geturl() to try curl
    first and urllib as fallback (opposite of the default order).
    Only safe inside the forked subprocess."""
    try:
        import viirs.utils.laads_data_download_v2 as _ldl
    except ImportError:
        return
    if not hasattr(_ldl, 'geturl') or not hasattr(_ldl, 'getcURL'):
        return
    _orig_geturl = _ldl.geturl
    _orig_getcURL = _ldl.getcURL  # may already be the logging wrapper

    def _curl_first_geturl(url, tok, *args, **kwargs):
        """Try curl first; fall back to urllib on failure."""
        import urllib.request
        import urllib.error
        # Try curl first
        try:
            result = _orig_getcURL(url, tok)
            if result is not None:
                return result
        except Exception:
            pass
        # Fall back to urllib
        try:
            headers = {'Authorization': f'Bearer {tok}'} if tok else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception:
            pass
        return None

    _ldl.geturl = _curl_first_geturl


def _sync_in_subprocess(url: str, target_dir: str, token: str,
                        timeout_s: int = 600,
                        curl_primary: bool = True) -> tuple:
    """Run sync() in a child process. netCDF4's corrupt-file
    validation (inside sync(), checking previously-downloaded .nc
    files) can segfault the whole interpreter on a sufficiently
    mangled/truncated file -- a plain try/except cannot catch that,
    since it's a C-level crash, not a Python exception. Running it
    in a subprocess means a crash there only kills that subprocess;
    this function detects that and returns a normal failure instead
    of taking the server down with it.

    Every HTTP request sync() makes is logged as a .http_request /
    .http_response file pair into target_dir (the day's own folder).
    Curl requests are logged as .curl_request / .curl_response.

    If curl_primary is True (default), the download order is
    curl-first with urllib fallback; otherwise urllib-first with
    curl fallback (the original third-party module's default order).

    Returns (ok: bool, error: str|None).
    """
    import multiprocessing

    def _target(q):
        try:
            _install_http_logging(target_dir)
            _install_curl_logging(target_dir)
            if curl_primary:
                _install_curl_primary_order()
            from viirs.utils.laads_data_download_v2 import sync as _sync
            _sync(url, target_dir, token)
            q.put((True, None))
        except Exception as exc:
            q.put((False, str(exc)))

    ctx = multiprocessing.get_context('fork')
    q = ctx.Queue()
    proc = ctx.Process(target=_target, args=(q,))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return False, f'timed out after {timeout_s}s'
    if proc.exitcode != 0:
        return False, (f'subprocess crashed (exit code {proc.exitcode}) -- '
                       f'likely a corrupted .nc file segfaulting the '
                       f'netCDF4/HDF5 library during validation')
    try:
        return q.get_nowait()
    except Exception:
        return False, 'subprocess exited with no result'


def _download_day(day: datetime.datetime, save_dir: str,
                  bbox_wgs84: tuple, token: str,
                  curl_primary: bool = True) -> dict:
    """Download one day's granules into save_dir/VNP14IMG/<year>/<jday>/.

    Always calls sync() -- does NOT skip just because the day-folder
    already has some .nc files. sync() lists what LAADS has for this
    day/bbox and only fetches files it doesn't already have locally
    (it recurses the remote listing and skips by filename), so this
    is the correct way to pick up any additional granules for a day
    that was only partially downloaded before, without re-fetching
    what's already there.

    Returns {'before': N, 'after': N, 'error': str|None} so the
    caller can report exactly how many were already present vs newly
    downloaded vs still missing for this day.
    """
    west, south, east, north = bbox_wgs84
    jday = day.timetuple().tm_yday
    year = day.year
    target_dir = os.path.join(
        save_dir, 'VNP14IMG', f'{year:04d}', f'{jday:03d}')
    os.makedirs(target_dir, exist_ok=True)

    before_files = set(glob.glob(os.path.join(target_dir, '*.nc')))
    before = len(before_files)

    url = (
        f'https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?'
        f'products=VNP14IMG&'
        f'temporalRanges={year}-{jday}&'
        f'regions=%5BBBOX%5DN{north:.6f}%20S{south:.6f}'
        f'%20E{east:.6f}%20W{west:.6f}'
    )
    ok, error = _sync_in_subprocess(url, target_dir, token,
                                     curl_primary=curl_primary)
    if not ok:
        sys.stderr.write(
            f'[year_viirs] download {day.date()}: {error}\n')

    after_files = set(glob.glob(os.path.join(target_dir, '*.nc')))
    new_files = sorted(after_files - before_files)
    for f in new_files:
        print(f'        DOWNLOAD SUCCESS: {os.path.basename(f)}',
              flush=True)
    after = len(after_files)
    return {'before': before, 'after': after, 'error': error}


def download_year(year: int, raster_path: str, save_dir: str,
                  token: str, workers: int = 16,
                  start: datetime.datetime = None,
                  end: datetime.datetime = None,
                  curl_primary: bool = True) -> dict:
    """Download .nc granules for the raster's full bbox + year window.

    Every day in the window is checked against LAADS's listing (via
    sync(), see _download_day) -- a day-folder already having some
    .nc files no longer skips that day outright, since that could
    mean only some of the day's granules were ever fetched. sync()
    itself avoids re-fetching files it already has locally.

    Returns {'already_present': N, 'newly_downloaded': N,
    'still_missing': N, 'missing_days': [...], 'total_nc': N} -- the
    counts requested for the end-of-run summary.
    """
    if start is None or end is None:
        ds, de = default_window(year)
        start = start or ds
        end = end or de
    if end < start:
        return {'already_present': 0, 'newly_downloaded': 0,
                'still_missing': 0, 'missing_days': [], 'total_nc': 0}

    bbox_wgs84 = _wgs84_extent_of(raster_path)

    days = []
    d = start
    while d <= end:
        days.append(d)
        d += datetime.timedelta(days=1)

    print(f'      [{os.path.basename(raster_path)}] '
          f'checking {len(days)} day(s) of VIIRS data '
          f'(workers={workers}) ...')

    per_day = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_download_day, d, save_dir, bbox_wgs84, token,
                        curl_primary): d
            for d in days
        }
        completed = 0
        for fut in as_completed(futs):
            day = futs[fut]
            try:
                per_day[day] = fut.result()
            except Exception as exc:
                sys.stderr.write(
                    f'[year_viirs] download error for {day.date()}: '
                    f'{exc}\n')
                per_day[day] = {'before': 0, 'after': 0, 'error': str(exc)}
            completed += 1
            info = per_day[day]
            gained = max(0, info['after'] - info['before'])
            if gained > 0:
                print(f"        CONFIRMED: {gained} new .nc file(s) for "
                      f"{day.date()} ({info['after']} total on disk now)",
                      flush=True)
            if completed % 10 == 0 or completed == len(days):
                print(f'        checked {completed}/{len(days)}',
                      flush=True)

    already_present = 0
    newly_downloaded = 0
    still_missing = 0
    missing_days = []
    total_nc = 0
    for day, info in per_day.items():
        before, after = info['before'], info['after']
        total_nc += after
        gained = max(0, after - before)
        newly_downloaded += gained
        already_present += min(before, after)
        if after == 0:
            still_missing += 1
            reason = info['error'] or 'no granules returned for this day/bbox'
            missing_days.append((day.date().isoformat(), reason))

    return {
        'already_present': already_present,
        'newly_downloaded': newly_downloaded,
        'still_missing': still_missing,
        'missing_days': missing_days,
        'total_nc': total_nc,
    }


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
                   shapify_workers: int = 8,
                   curl_primary: bool = True) -> dict:
    """Run download + shapify + index for a year's full raster footprint and
    default seasonal window. Idempotent. Returns counts.

    Migration of VIIRS data from previous stack-dated directories
    happens once, centrally, in __main__.py (migrate_stale_viirs_data
    + purge_active_shapefiles) before this is called -- not here.
    """
    if save_dir is None:
        save_dir = year_viirs_dir(state, year)
    os.makedirs(save_dir, exist_ok=True)

    dl_result = download_year(
        year, raster_path, save_dir, state.laads_token,
        workers=download_workers, curl_primary=curl_primary)
    print(f"      [{os.path.basename(raster_path)}] VIIRS summary: "
          f"{dl_result['already_present']} already present, "
          f"{dl_result['newly_downloaded']} newly downloaded, "
          f"{dl_result['still_missing']} day(s) still missing "
          f"(total {dl_result['total_nc']} .nc file(s) on disk).")
    for day_iso, reason in dl_result['missing_days']:
        sys.stderr.write(
            f"      [{os.path.basename(raster_path)}] no data for "
            f"{day_iso}: {reason}\n")

    n_shp = shapify_year(save_dir, raster_path, workers=shapify_workers)
    index_path = build_year_index(save_dir, raster_path)
    return {'n_nc': dl_result['total_nc'], 'n_shp': n_shp,
            'save_dir': save_dir, 'index_path': index_path,
            'download_summary': dl_result}


def bootstrap_all_years(state, curl_primary: bool = True) -> dict:
    """Run bootstrap_year for every (year, raster) in state.rasters_by_year.
    Sequential — printing progress is more useful than parallelism here.
    Returns {year: result_dict}."""
    out = {}
    rasters = state.rasters_by_year or {}
    for year in sorted(rasters):
        out[year] = bootstrap_year(
            state, year, rasters[year],
            download_workers=state.viirs_download_workers or 16,
            shapify_workers=state.viirs_shapify_workers or 8,
            curl_primary=curl_primary)
    return out
