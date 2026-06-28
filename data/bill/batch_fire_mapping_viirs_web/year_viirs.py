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
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from osgeo import gdal, osr

gdal.UseExceptions()


def _ts() -> str:
    """Current timestamp as [YYYY-MM-DD HH:MM:SS], matching the
    prefix used everywhere else in startup output."""
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')


def _log(msg: str = '') -> None:
    """print() with a timestamp prefix on every line."""
    ts = _ts()
    for line in str(msg).split('\n'):
        print(f'{ts} {line}', flush=True)


def _elog(msg: str = '') -> None:
    """sys.stderr.write() with a timestamp prefix, matching _log()."""
    ts = _ts()
    for line in str(msg).rstrip('\n').split('\n'):
        sys.stderr.write(f'{ts} {line}\n')
    sys.stderr.flush()


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
                        _elog(
                            f'[year_viirs] WARNING: overwriting '
                            f'{dst_path} with different content from '
                            f'{nc_path} (sizes/bytes differ) -- '
                            f'stomping as instructed.')
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
                    _elog(
                        f'[year_viirs] WARNING: could not remove '
                        f'{f}: {exc}')
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
        _elog(f'[year_viirs] WARNING: could not write HTTP '
              f'log pair in {log_dir}: {exc}')


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
        _elog(f'[year_viirs] WARNING: could not write curl '
              f'log pair in {log_dir}: {exc}')


def _classify_status(method: str, status, exc: Exception = None) -> tuple:
    """Returns (return_code_str, meaning_str) for a request outcome,
    used to build the JSON event entries requested for every
    listing/download attempt."""
    if exc is not None:
        msg = str(exc)
        if '401' in msg or 'Unauthorized' in msg:
            return ('401', f'{method}: server rejected credentials for '
                    f'this specific request (token works elsewhere in '
                    f'this run -- likely an auth-header/redirect '
                    f'handling difference, not a bad token)')
        if '500' in msg:
            return ('500', f'{method}: LAADS server-side error')
        if 'items' in msg and 'str' in msg:
            return ('EXCEPTION', f'{method}: third-party wrapper bug '
                    f'(not a network/auth failure) -- {msg}')
        return ('EXCEPTION', f'{method}: request raised {type(exc).__name__}'
                f': {msg}')
    if status == 200:
        return ('200', f'{method}: request succeeded')
    if status == 401:
        return ('401', f'{method}: unauthorized')
    if status == 500:
        return ('500', f'{method}: server error')
    if status is None:
        return ('NONE', f'{method}: no response received')
    return (str(status), f'{method}: HTTP {status}')


_FILENAME_RE_GLOBAL = None


def _extract_filenames(text: str) -> list:
    """Best-effort extraction of VNP14IMG granule filenames from a
    listing response body, for the 'files' field of listing JSON
    events."""
    import re
    global _FILENAME_RE_GLOBAL
    if _FILENAME_RE_GLOBAL is None:
        _FILENAME_RE_GLOBAL = re.compile(
            r'VNP14IMG\.A\d{7}\.\d{4}\.\d{3}\.\d{13}\.nc')
    try:
        return sorted(set(_FILENAME_RE_GLOBAL.findall(text)))
    except Exception:
        return []


def _log_json_event(events_path: str, event: dict) -> None:
    """Print one JSON-encoded line to stdout for this listing/download
    attempt (as requested), and also append it to a shared JSONL file
    so the parent process can build the post-run summary section
    (each day's sync() runs in its own subprocess, so module-level
    counters here wouldn't be visible to the parent otherwise)."""
    import json as _json
    line = _json.dumps(event, default=str)
    print(line, flush=True)
    if events_path:
        try:
            with open(events_path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except OSError:
            pass


def _install_http_logging(log_dir: str, events_path: str = None) -> None:
    """Monkey-patch urllib.request.urlopen (used by
    viirs.utils.laads_data_download_v2.geturl) so every HTTP
    request/response it makes gets logged as a file pair in log_dir,
    plus a JSON event (listing or download) to stdout/events_path.
    Only safe to call inside the forked subprocess in
    _sync_in_subprocess -- it permanently patches urlopen in whatever
    process calls it."""
    import urllib.request

    _orig_urlopen = urllib.request.urlopen

    def _is_listing_url(url: str) -> bool:
        return '/content/details' in url

    def _logging_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        headers = (dict(req.headers) if hasattr(req, 'headers') else {})
        request_text = f'GET {url}\n' + '\n'.join(
            f'{k}: {v}' for k, v in headers.items())
        listing = _is_listing_url(url)
        filename = url.rsplit('/', 1)[-1] if not listing else None
        _log(f'        [http] ENTER  GET {url}')
        # Same rationale as the curl path's retry loop below: only
        # retry on a 5xx (transient server-side) response, not on 4xx
        # or connection-level failures. The inner try/except here only
        # decides retry-vs-not; any exception that ends up propagating
        # past it (a non-5xx HTTPError, or a 5xx with retries
        # exhausted) falls through to the existing outer except below,
        # so every failure path still gets logged exactly as before --
        # this only adds a delay-and-retry step in front of that.
        try:
            _RETRY_DELAYS_S = (2, 5, 10)
            attempt = 0
            while True:
                try:
                    resp = _orig_urlopen(req, *args, **kwargs)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code >= 500 and attempt < len(_RETRY_DELAYS_S):
                        delay = _RETRY_DELAYS_S[attempt]
                        _log(f'        [http] RETRY  GET {url} -- '
                             f'HTTP {exc.code}, retrying in {delay}s '
                             f'({attempt + 1}/{len(_RETRY_DELAYS_S)}) ...')
                        time.sleep(delay)
                        attempt += 1
                        continue
                    raise
            body = resp.read()
            body_text = body.decode('utf-8', errors='replace')
            response_text = (
                f'HTTP {resp.status}\n'
                + '\n'.join(f'{k}: {v}' for k, v in resp.headers.items())
                + '\n\n' + body_text)
            _log_http_pair(log_dir, request_text, response_text)
            _log(f'        [http] REQUEST  GET {url}')
            _log(f'        [http] RESPONSE HTTP {resp.status} '
                 f'({len(body)} bytes) for {url}')
            rc, meaning = _classify_status('urllib', resp.status)
            if listing:
                _log_json_event(events_path, {
                    'kind': 'listing', 'method': 'urllib', 'url': url,
                    'files': _extract_filenames(body_text),
                    'return_code': rc, 'meaning': meaning,
                    'request': request_text[:2000],
                    'response': response_text[:4000],
                })
            else:
                _log_json_event(events_path, {
                    'kind': 'download', 'method': 'urllib',
                    'filename': filename, 'size_bytes': len(body),
                    'return_code': rc, 'meaning': meaning,
                    'request': request_text[:2000],
                    'response': response_text[:2000],
                })
            import io
            replay = io.BytesIO(body)
            replay.status = resp.status
            replay.headers = resp.headers
            _log(f'        [http] EXIT   GET {url} -- OK '
                 f'(HTTP {resp.status})')
            return replay
        except Exception as exc:
            response_text = f'EXCEPTION: {exc}'
            _log_http_pair(log_dir, request_text, response_text)
            _log(f'        [http] REQUEST  GET {url}')
            _log(f'        [http] RESPONSE EXCEPTION for {url}: {exc}')
            rc, meaning = _classify_status('urllib', None, exc)
            if listing:
                _log_json_event(events_path, {
                    'kind': 'listing', 'method': 'urllib', 'url': url,
                    'files': [], 'return_code': rc, 'meaning': meaning,
                    'request': request_text[:2000],
                    'response': response_text[:2000],
                })
            else:
                _log_json_event(events_path, {
                    'kind': 'download', 'method': 'urllib',
                    'filename': filename, 'size_bytes': 'N/A',
                    'return_code': rc, 'meaning': meaning,
                    'request': request_text[:2000],
                    'response': response_text[:2000],
                })
            _log(f'        [http] EXIT   GET {url} -- FAILED: {exc}')
            raise

    urllib.request.urlopen = _logging_urlopen


def _curl_cli_request(url: str, token: str, timeout: int = 120) -> tuple:
    """Issue the request using the real system `curl` binary via
    subprocess, bypassing the third-party module's getcURL() wrapper
    entirely (that wrapper has a bug -- 'str' object has no attribute
    'items' -- that fires on every call regardless of outcome).

    Returns (body_bytes_or_None, http_status_or_None, request_text,
    response_text).
    """
    request_text = f'CURL {url}\nAuthorization: Bearer {token[:20]}...'
    cmd = [
        'curl', '-s', '-L', '--max-time', str(timeout),
        '-D', '-',  # dump headers to stdout, ahead of the body
        '-H', f'Authorization: Bearer {token}',
        url,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout + 10)
    except Exception as exc:
        return None, None, request_text, f'CURL CLI EXCEPTION: {exc}'

    raw = proc.stdout
    # With -L, redirects mean multiple header blocks are concatenated;
    # the LAST one (immediately preceding the final body) is the one
    # that matters. Split on the blank-line CRLF/LF boundary between
    # headers and body, repeatedly, to walk past intermediate redirects.
    status = None
    body = raw
    remaining = raw
    while True:
        sep_idx = None
        for sep in (b'\r\n\r\n', b'\n\n'):
            idx = remaining.find(sep)
            if idx != -1:
                sep_idx = (idx, len(sep))
                break
        if sep_idx is None:
            break
        idx, sep_len = sep_idx
        header_block = remaining[:idx]
        first_line = header_block.split(b'\n', 1)[0]
        m = re.match(rb'HTTP/\S+\s+(\d+)', first_line)
        if not m:
            break
        status = int(m.group(1))
        body = remaining[idx + sep_len:]
        # If this header block's body looks like it's just another
        # header block (next redirect hop), keep peeling; otherwise stop.
        if body[:5] == b'HTTP/':
            remaining = body
            continue
        break

    if status is None and proc.returncode != 0:
        stderr_text = proc.stderr.decode('utf-8', errors='replace')
        return None, None, request_text, f'CURL CLI FAILED (exit {proc.returncode}): {stderr_text}'

    response_text = (f'HTTP {status}\n\n'
                     + body[:4000].decode('utf-8', errors='replace'))
    return body, status, request_text, response_text


def _install_native_curl(log_dir: str, events_path: str = None) -> None:
    """Replace the third-party module's getcURL() entirely with a call
    to the real system curl binary, since the wrapper has a bug that
    raises on every call regardless of whether the transfer actually
    succeeded. Logs a .curl_request/.curl_response file pair and a
    JSON listing/download event for every call, same as
    _install_http_logging does for urllib."""
    try:
        import viirs.utils.laads_data_download_v2 as _ldl
    except ImportError:
        return
    if not hasattr(_ldl, 'getcURL'):
        return

    def _is_listing_url(url: str) -> bool:
        return '/content/details' in url

    def _native_getcURL(url, tok, *args, **kwargs):
        listing = _is_listing_url(url)
        filename = url.rsplit('/', 1)[-1] if not listing else None
        _log(f'        [curl] ENTER  {url}')
        # Retry a handful of times on a 5xx (server-side) response.
        # LAADS occasionally returns transient 500s -- the LAADS
        # preflight check at startup logs exactly this -- and without
        # a retry here, a transient server hiccup on the *listing*
        # call gets indistinguishable from a genuine "no fire
        # detected this day" and silently drops real detections from
        # the season's accumulation. 4xx responses and outright
        # connection failures are deliberately NOT retried: those
        # aren't transient server load, so retrying just delays an
        # outcome that won't change.
        _RETRY_DELAYS_S = (2, 5, 10)
        body, status, request_text, response_text = _curl_cli_request(
            url, tok)
        attempt = 0
        while (status is not None and status >= 500
               and attempt < len(_RETRY_DELAYS_S)):
            delay = _RETRY_DELAYS_S[attempt]
            _log(f'        [curl] RETRY  {url} -- HTTP {status}, '
                 f'retrying in {delay}s ({attempt + 1}/'
                 f'{len(_RETRY_DELAYS_S)}) ...')
            time.sleep(delay)
            body, status, request_text, response_text = _curl_cli_request(
                url, tok)
            attempt += 1
        _log_curl_pair(log_dir, request_text, response_text)
        _log(f'        [curl] REQUEST  {url}')
        if body is None:
            _log(f'        [curl] RESPONSE FAILED for {url}: '
                 f'{response_text[:200]}')
            _log(f'        [curl] EXIT   {url} -- FAILED')
            rc, meaning = _classify_status(
                'curl(native)', status,
                Exception(response_text[:300]))
            if listing:
                _log_json_event(events_path, {
                    'kind': 'listing', 'method': 'curl', 'url': url,
                    'files': [], 'return_code': rc, 'meaning': meaning,
                    'request': request_text[:2000],
                    'response': response_text[:2000],
                })
            else:
                _log_json_event(events_path, {
                    'kind': 'download', 'method': 'curl',
                    'filename': filename, 'size_bytes': 'N/A',
                    'return_code': rc, 'meaning': meaning,
                    'request': request_text[:2000],
                    'response': response_text[:2000],
                })
            return None
        _log(f'        [curl] RESPONSE HTTP {status} '
             f'({len(body)} bytes) for {url}')
        _log(f'        [curl] EXIT   {url} -- '
             f'{"OK" if status == 200 else "FAILED"} (HTTP {status})')
        rc, meaning = _classify_status('curl(native)', status)
        if listing:
            body_text = body.decode('utf-8', errors='replace')
            _log_json_event(events_path, {
                'kind': 'listing', 'method': 'curl', 'url': url,
                'files': _extract_filenames(body_text),
                'return_code': rc, 'meaning': meaning,
                'request': request_text[:2000],
                'response': response_text[:4000],
            })
        else:
            _log_json_event(events_path, {
                'kind': 'download', 'method': 'curl',
                'filename': filename, 'size_bytes': len(body),
                'return_code': rc, 'meaning': meaning,
                'request': request_text[:2000],
                'response': response_text[:2000],
            })
        if status != 200:
            return None
        return body

    _ldl.getcURL = _native_getcURL


def _install_curl_primary_order() -> None:
    """Monkey-patch the third-party module's geturl() to try curl
    first and urllib as fallback (opposite of the default order).
    Only safe inside the forked subprocess. Assumes getcURL has
    already been replaced (by _install_native_curl) with the native
    CLI implementation, or the original wrapper otherwise."""
    try:
        import viirs.utils.laads_data_download_v2 as _ldl
    except ImportError:
        return
    if not hasattr(_ldl, 'geturl') or not hasattr(_ldl, 'getcURL'):
        return
    _orig_getcURL = _ldl.getcURL  # already the logging/native wrapper

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
                        curl_primary: bool = True,
                        events_path: str = None) -> tuple:
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
    Curl requests (now issued via the native system curl binary, not
    the buggy third-party getcURL wrapper) are logged as
    .curl_request / .curl_response. A JSON event is also emitted to
    stdout (and appended to events_path) for every listing/download
    attempt, on either path.

    If curl_primary is True (default), the download order is
    curl-first with urllib fallback; otherwise urllib-first with
    curl fallback (the original third-party module's default order).

    Returns (ok: bool, error: str|None).
    """
    import multiprocessing

    def _target(q):
        try:
            _install_http_logging(target_dir, events_path)
            _install_native_curl(target_dir, events_path)
            if curl_primary:
                _install_curl_primary_order()
            from viirs.utils.laads_data_download_v2 import sync as _sync
            _log(f'      [sync] ENTER  {url}')
            _sync(url, target_dir, token)
            _log(f'      [sync] EXIT   {url} -- OK')
            q.put((True, None))
        except Exception as exc:
            _log(f'      [sync] EXIT   {url} -- FAILED: {exc}')
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
                  curl_primary: bool = True,
                  events_path: str = None) -> dict:
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
    _log(f'        [download] {day.date()} (jday {jday}): starting '
         f'({before} .nc already on disk) ...')
    ok, error = _sync_in_subprocess(url, target_dir, token,
                                     curl_primary=curl_primary,
                                     events_path=events_path)
    if not ok:
        _elog(
            f'[year_viirs] download {day.date()}: {error}')

    after_files = set(glob.glob(os.path.join(target_dir, '*.nc')))
    new_files = sorted(after_files - before_files)
    for f in new_files:
        _log(f'        DOWNLOAD SUCCESS: {os.path.basename(f)}')
    after = len(after_files)
    _log(f'        [download] {day.date()} (jday {jday}): done '
         f'({after} .nc on disk now, {len(new_files)} new'
         f'{"" if ok else ", FAILED: " + str(error)}).')
    return {'before': before, 'after': after, 'error': error}


def download_year(year: int, raster_path: str, save_dir: str,
                  token: str, workers: int = 16,
                  start: datetime.datetime = None,
                  end: datetime.datetime = None,
                  curl_primary: bool = True,
                  parallel: bool = False) -> dict:
    """Download .nc granules for the raster's full bbox + year window.

    Every day in the window is checked against LAADS's listing (via
    sync(), see _download_day) -- a day-folder already having some
    .nc files no longer skips that day outright, since that could
    mean only some of the day's granules were ever fetched. sync()
    itself avoids re-fetching files it already has locally.

    Returns {'already_present': N, 'newly_downloaded': N,
    'still_missing': N, 'missing_days': [...], 'total_nc': N} -- the
    counts requested for the end-of-run summary.

    If parallel is False (default), days are downloaded one at a
    time, with no thread pool -- entry/exit and request/response
    messages for every curl/http call print directly to stdout in
    the order they happen, since there's no concurrent worker output
    to untangle. If parallel is True, days run concurrently across
    up to `workers` threads (each still in its own crash-safe
    subprocess), matching the original behaviour; per-call stdout
    lines still print but may interleave across days.
    """
    if start is None or end is None:
        ds, de = default_window(year)
        start = start or ds
        end = end or de
    if end < start:
        return {'already_present': 0, 'newly_downloaded': 0,
                'still_missing': 0, 'missing_days': [], 'total_nc': 0}

    bbox_wgs84 = _wgs84_extent_of(raster_path)

    events_path = os.path.join(save_dir, '_viirs_events.jsonl')
    try:
        os.makedirs(save_dir, exist_ok=True)
        if os.path.isfile(events_path):
            os.remove(events_path)
    except OSError:
        pass

    days = []
    d = start
    while d <= end:
        days.append(d)
        d += datetime.timedelta(days=1)

    _mode_label = (f'parallel, workers={workers}' if parallel
                  else 'serial, no thread pool')
    _log(f'      [{os.path.basename(raster_path)}] '
         f'VIIRS download: starting -- checking {len(days)} day(s) '
         f'({_mode_label}) ...')

    per_day = {}
    if parallel:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {
                pool.submit(_download_day, d, save_dir, bbox_wgs84, token,
                            curl_primary, events_path): d
                for d in days
            }
            completed = 0
            for fut in as_completed(futs):
                day = futs[fut]
                try:
                    per_day[day] = fut.result()
                except Exception as exc:
                    _elog(
                        f'[year_viirs] download error for {day.date()}: '
                        f'{exc}')
                    per_day[day] = {'before': 0, 'after': 0,
                                    'error': str(exc)}
                completed += 1
                info = per_day[day]
                gained = max(0, info['after'] - info['before'])
                if gained > 0:
                    _log(f"        CONFIRMED: {gained} new .nc file(s) for "
                         f"{day.date()} ({info['after']} total on disk now)")
                if completed % 10 == 0 or completed == len(days):
                    _log(f'        checked {completed}/{len(days)}')
    else:
        # Serial: one day at a time, no thread pool. Every curl/http
        # entry/exit + request/response message (see
        # _logging_urlopen / _logging_getcURL above) prints to stdout
        # in strict chronological order, since nothing else is
        # running concurrently to interleave with it.
        completed = 0
        for day in days:
            try:
                per_day[day] = _download_day(
                    day, save_dir, bbox_wgs84, token, curl_primary,
                    events_path)
            except Exception as exc:
                _elog(
                    f'[year_viirs] download error for {day.date()}: '
                    f'{exc}')
                per_day[day] = {'before': 0, 'after': 0, 'error': str(exc)}
            completed += 1
            info = per_day[day]
            gained = max(0, info['after'] - info['before'])
            if gained > 0:
                _log(f"        CONFIRMED: {gained} new .nc file(s) for "
                     f"{day.date()} ({info['after']} total on disk now)")
            if completed % 10 == 0 or completed == len(days):
                _log(f'        checked {completed}/{len(days)}')

    _log(f'      [{os.path.basename(raster_path)}] '
         f'VIIRS download: done ({len(days)} day(s) checked).')

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
        'events_path': events_path,
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
        _log(f'      [{os.path.basename(raster_path)}] '
             f'shapify: starting -- {len(pending_nc)} .nc granule(s) '
             f'(workers={workers}) ...')
        shapify_viirs(save_dir, raster_path, workers=workers)
        _log(f'      [{os.path.basename(raster_path)}] shapify: done.')
    n_shp = len(glob.glob(os.path.join(
        save_dir, 'VNP14IMG', '**', '*.shp'), recursive=True))
    return n_shp


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

    _log(f'      [{os.path.basename(raster_path)}] '
         f'year index: starting -- building from {len(shp_files)} '
         f'shapefiles ...')

    frames = []
    crs = None
    for shp in shp_files:
        dt = extract_datetime_from_filename(Path(shp).stem)
        if dt is None:
            continue
        try:
            gdf = gpd.read_file(shp)
        except Exception as exc:
            _elog(f'[year_viirs] index read {shp}: {exc}')
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
        _log(f'      [{os.path.basename(raster_path)}] year index: '
             f'done (no usable shapefiles, index cleared).')
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
    _log(f'      [{os.path.basename(raster_path)}] '
         f'year index: done -- {len(combined)} features → {index_path}')
    return index_path




def print_viirs_events_summary(events_path: str) -> dict:
    """Read the JSONL events file written during this run's listing
    and download attempts, and print the requested 'VIIRS listing and
    VIIRS data summary' section -- counts by kind/method/outcome,
    after the run finishes. Returns the summary dict (also useful for
    tests)."""
    summary = {
        'listing': {'curl': {'ok': 0, 'failed': 0},
                    'urllib': {'ok': 0, 'failed': 0}},
        'download': {'curl': {'ok': 0, 'failed': 0},
                     'urllib': {'ok': 0, 'failed': 0}},
        'files_seen': set(),
        'files_downloaded': set(),
        'total_events': 0,
    }
    if not events_path or not os.path.isfile(events_path):
        _log('[VIIRS SUMMARY] No events recorded for this run.')
        return summary
    with open(events_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            summary['total_events'] += 1
            kind = ev.get('kind')
            method = ev.get('method', 'unknown')
            ok = (str(ev.get('return_code')) == '200')
            if kind == 'listing':
                bucket = summary['listing'].setdefault(
                    method, {'ok': 0, 'failed': 0})
                bucket['ok' if ok else 'failed'] += 1
                for fn in ev.get('files', []):
                    summary['files_seen'].add(fn)
            elif kind == 'download':
                bucket = summary['download'].setdefault(
                    method, {'ok': 0, 'failed': 0})
                bucket['ok' if ok else 'failed'] += 1
                if ok and ev.get('filename'):
                    summary['files_downloaded'].add(ev['filename'])

    _log('')
    _log('=' * 60)
    _log('  VIIRS LISTING AND DATA SUMMARY')
    _log('=' * 60)
    _log(f"  Total request events recorded: {summary['total_events']}")
    _log('  Listing requests:')
    for method, counts in summary['listing'].items():
        _log(f"    {method:8s}: {counts['ok']} succeeded, "
             f"{counts['failed']} failed")
    _log('  Download requests:')
    for method, counts in summary['download'].items():
        _log(f"    {method:8s}: {counts['ok']} succeeded, "
             f"{counts['failed']} failed")
    _log(f"  Distinct granule filenames seen in listings: "
         f"{len(summary['files_seen'])}")
    _log(f"  Distinct granule filenames actually downloaded: "
         f"{len(summary['files_downloaded'])}")
    _log('=' * 60)
    _log('')
    return summary


def build_viirs_accumulated_buffer(save_dir: str, raster_path: str) -> dict:
    """Accumulate every VIIRS detection across all per-granule
    shapefiles under save_dir into a single deduplicated buffer:

    - Initialize an empty buffer.
    - For each detection record, add it to the buffer if no record
      already exists at that geo-coordinate.
    - If a record already exists at that coordinate, keep whichever
      of the two (existing vs candidate) has the most recent
      detection date.
    - All coordinates are reprojected to the active raster's own CRS
      first, so the buffer (and the resulting shapefile) use one
      consistent projection regardless of which CRS individual
      granule shapefiles came in.

    Writes the deduplicated result to <save_dir>/VIIRS.shp, and a
    lightweight JSON cache (<save_dir>/viirs_overlay.json) with the
    point coordinates + detection dates + nominal VIIRS pixel
    resolution, for the client-side overlay.

    Returns a summary dict: {'n_input_records', 'n_unique', 'shp_path',
    'overlay_path'}.
    """
    import geopandas as gpd
    import pandas as pd
    from osgeo import gdal as _gdal, osr as _osr

    index_path = year_index_path(save_dir)
    overlay_path = os.path.join(save_dir, 'viirs_overlay.json')
    shp_path = os.path.join(save_dir, 'VIIRS.shp')

    if not os.path.isfile(index_path):
        _log('      [accumulate] No year_index.gpkg found -- nothing to '
             'accumulate yet.')
        return {'n_input_records': 0, 'n_unique': 0, 'shp_path': None,
                'overlay_path': None}

    _log(f'      [accumulate] Building deduplicated VIIRS buffer from '
         f'{index_path}: starting ...')

    try:
        gdf = gpd.read_file(index_path)
    except Exception as exc:
        _elog(f'[year_viirs] accumulate: could not read {index_path}: {exc}')
        return {'n_input_records': 0, 'n_unique': 0, 'shp_path': None,
                'overlay_path': None}

    n_input = len(gdf)
    if n_input == 0:
        _log('      [accumulate] year index is empty -- nothing to '
             'accumulate.')
        return {'n_input_records': 0, 'n_unique': 0, 'shp_path': None,
                'overlay_path': None}

    # Consistent CRS: reproject everything to the active raster's own
    # projection (same convention used elsewhere in this app, e.g.
    # bcws.py), rather than trusting whatever CRS happened to be on
    # the first granule shapefile read into the index.
    ds = _gdal.Open(raster_path)
    raster_wkt = ds.GetProjection()
    ds = None
    if raster_wkt and gdf.crs is not None:
        try:
            gdf = gdf.to_crs(raster_wkt)
        except Exception as exc:
            _elog(f'[year_viirs] accumulate: reprojection failed, '
                 f'keeping original CRS: {exc}')

    # Dedup by geo-coordinate (rounded to the nearest metre -- VIIRS
    # detections at the same physical pixel reproject to effectively
    # identical coordinates across different overpasses/days, modulo
    # floating-point noise), keeping the record with the latest
    # det_dt at each coordinate.
    xs = gdf.geometry.x.round(0)
    ys = gdf.geometry.y.round(0)
    gdf = gdf.assign(_xr=xs, _yr=ys)
    gdf = gdf.sort_values('det_dt')
    deduped = gdf.drop_duplicates(subset=['_xr', '_yr'], keep='last')
    deduped = deduped.drop(columns=['_xr', '_yr'])
    n_unique = len(deduped)

    deduped = gpd.GeoDataFrame(deduped, crs=gdf.crs)
    tmp_path = shp_path + '.tmp.shp'
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        try:
            os.remove(tmp_path[:-4] + ext)
        except FileNotFoundError:
            pass
    deduped.to_file(tmp_path, driver='ESRI Shapefile')
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        src = tmp_path[:-4] + ext
        dst = shp_path[:-4] + ext
        if os.path.isfile(src):
            try:
                os.remove(dst)
            except FileNotFoundError:
                pass
            os.replace(src, dst)

    points = [[round(float(pt.x), 2), round(float(pt.y), 2)]
              for pt in deduped.geometry]
    det_dts = [str(v) for v in deduped['det_dt']]
    overlay = {
        'points': points,
        'det_dts': det_dts,
        'n_points': len(points),
        'native_resolution_m': 375.0,  # VIIRS I-band nadir pixel size
        'crs_wkt': raster_wkt,
        'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(overlay_path, 'w', encoding='utf-8') as f:
        json.dump(overlay, f)

    _log(f'      [accumulate] Building deduplicated VIIRS buffer: done '
         f'-- {n_input} input record(s) -> {n_unique} unique '
         f'location(s) -> {shp_path}')
    return {'n_input_records': n_input, 'n_unique': n_unique,
            'shp_path': shp_path, 'overlay_path': overlay_path}


def load_viirs_overlay(state) -> dict:
    """Load the cached viirs_overlay.json for the active year, or an
    empty overlay if it doesn't exist yet."""
    year = state.active_year
    save_dir = year_viirs_dir(state, year)
    overlay_path = os.path.join(save_dir, 'viirs_overlay.json')
    if not os.path.isfile(overlay_path):
        return {'points': [], 'det_dts': [], 'n_points': 0,
                'native_resolution_m': 375.0}
    with open(overlay_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def bootstrap_year(state, year: int, raster_path: str,
                   save_dir: str = None,
                   download_workers: int = 16,
                   shapify_workers: int = 8,
                   curl_primary: bool = True,
                   parallel_viirs_downloading: bool = False) -> dict:
    """Run download + shapify + index for a year's full raster footprint and
    default seasonal window. Idempotent. Returns counts.

    Migration of VIIRS data from previous stack-dated directories
    happens once, centrally, in __main__.py (migrate_stale_viirs_data
    + purge_active_shapefiles) before this is called -- not here.
    """
    if save_dir is None:
        save_dir = year_viirs_dir(state, year)
    os.makedirs(save_dir, exist_ok=True)

    _log(f'      [{os.path.basename(raster_path)}] year {year}: '
         f'bootstrap starting ...')

    dl_result = download_year(
        year, raster_path, save_dir, state.laads_token,
        workers=download_workers, curl_primary=curl_primary,
        parallel=parallel_viirs_downloading)
    _log(f"      [{os.path.basename(raster_path)}] VIIRS summary: "
         f"{dl_result['already_present']} already present, "
         f"{dl_result['newly_downloaded']} newly downloaded, "
         f"{dl_result['still_missing']} day(s) still missing "
         f"(total {dl_result['total_nc']} .nc file(s) on disk).")
    for day_iso, reason in dl_result['missing_days']:
        _elog(
            f"      [{os.path.basename(raster_path)}] no data for "
            f"{day_iso}: {reason}")

    n_shp = shapify_year(save_dir, raster_path, workers=shapify_workers)
    index_path = build_year_index(save_dir, raster_path)

    print_viirs_events_summary(dl_result.get('events_path'))

    acc_result = build_viirs_accumulated_buffer(save_dir, raster_path)

    _log(f'      [{os.path.basename(raster_path)}] year {year}: '
         f'bootstrap done.')
    return {'n_nc': dl_result['total_nc'], 'n_shp': n_shp,
            'save_dir': save_dir, 'index_path': index_path,
            'download_summary': dl_result,
            'accumulate_summary': acc_result}


def bootstrap_all_years(state, curl_primary: bool = True,
                        parallel_viirs_downloading: bool = False) -> dict:
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
            curl_primary=curl_primary,
            parallel_viirs_downloading=parallel_viirs_downloading)
    return out
