"""Sentinel-2 cloud cover percentages for an AOI's tiles.

An importable adaptation of ``sentinel2_extract_cloud_cover_tiles.py``:
same discovery over the Canadian mirror, same metadata extraction, same
per-product result cache. What changes is the shape -- no argument
parsing, no plotting, no printing -- and where the cache lives: under
the server's output root on real disk, so it survives restarts and the
ramdisk being cleared.

Only the L2A level is queried. The cloud percentage is a property of the
Sentinel-2 acquisition, not of what we later build from it, so the same
figure applies whether the operator is choosing an L2-recent start date
or a MRAP composite to clip -- both are asking "was it clear that day".

Everything here is best-effort. The network may be unreachable, the
mirror may be slow, a product's metadata may be malformed; none of that
should ever stop a date list from being shown. Failures are recorded as
"unknown" and retried on a later call.
"""

import json
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

BUCKET = 'sentinel-products-ca-mirror'
S3_BASE_URL = f'https://{BUCKET}.s3.amazonaws.com'
LEVEL_PREFIXES = {
    'L1C': 'Sentinel-2/S2MSI1C',
    'L2A': 'Sentinel-2/S2MSI2A',
}
LEVEL_METADATA = {
    'L1C': 'MTD_MSIL1C.xml',
    'L2A': 'MTD_MSIL2A.xml',
}
L2A_PREFIX = LEVEL_PREFIXES['L2A']

# Parallelism for the mirror. The original script defaults to 8 and
# accepts --workers; here it is a module constant because the caller is
# a web request, not a person at a terminal.
N_WORKERS = 8

# How long a "no products found" answer stays trusted. A date with no
# acquisition will not grow one, but the mirror can lag, so this is not
# cached forever.
_EMPTY_TTL_S = 6 * 3600

_lock = threading.Lock()
_inflight: dict = {}

# Live progress per key, so the dialog can say what is happening rather
# than leaving the operator to guess whether anything is running at all.
_progress: dict = {}
# What the last completed run did. Kept AFTER the run finishes: a fast
# failure clears _progress within a second, so without this the dialog
# polls forever against an empty progress dict and reports nothing --
# which is indistinguishable from a feature that does not work.
_last_run: dict = {}


def last_run(key: str) -> dict:
    with _lock:
        return dict(_last_run.get(key) or {})


def key_for(tiles) -> str:
    """The progress/last-run key for a tile set.

    Callers must use THIS rather than joining their own list: the
    module canonicalises tile IDs ('10UFB' -> 'T10UFB') before keying,
    so an endpoint joining the raw shapefile spelling looked up a key
    that never existed -- progress and last-run came back empty for
    ever, which is why the dialog polled without end and the bars never
    resolved from "still retrieving" to "nothing on record".
    """
    return ','.join(sorted(set(canon_tile(t) for t in (tiles or []) if t)))


def progress(key: str) -> dict:
    """Current retrieval progress for *key*, or {} if none."""
    with _lock:
        p = dict(_progress.get(key) or {})
    if not p:
        return {}
    done, total = int(p.get('done', 0)), int(p.get('total', 0))
    started = float(p.get('started', 0) or 0)
    elapsed = max(0.0, time.time() - started) if started else 0.0
    # ETA from the rate actually achieved. Withheld until a few days
    # have completed, because the first lookups include DNS, TLS and
    # the mirror warming up and are not representative of the rest.
    eta = None
    if done >= 3 and total > done and elapsed > 0:
        eta = (elapsed / done) * (total - done)
    p['elapsed_s'] = round(elapsed, 1)
    p['eta_s'] = None if eta is None else round(eta, 1)
    return p


# ---------------------------------------------------------------- cache

def cache_path(cache_root: str) -> str:
    """Where the per-tile cloud figures live, on real disk."""
    return os.path.join(cache_root, 'cloud_cover.json')


def _load(cache_root: str) -> dict:
    try:
        with open(cache_path(cache_root), encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        if int(data.get('_version', 1)) < CACHE_VERSION:
            sys.stderr.write(
                f'[cloud] discarding cache written by an older version '
                f'({len(data)} entries): its tile matching was wrong, '
                f'so its "no products" answers cannot be trusted\n')
            return {}
        return data
    except (OSError, ValueError):
        return {}


def _save(cache_root: str, data: dict) -> None:
    p = cache_path(cache_root)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = dict(data)
        data['_version'] = CACHE_VERSION
        tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except OSError as exc:
        sys.stderr.write(f'[cloud] could not save cache: {exc}\n')


# Cache format. Bumped when a change would make older entries wrong
# rather than merely incomplete -- entries from an earlier version are
# then ignored instead of being trusted.
#
# v2: tile IDs are canonicalised. v1 compared the shapefile's '10UFB'
# against the 'T10UFB' in product filenames, so nothing ever matched
# and every tile-day was recorded as "no products". Those entries have
# to be discarded, not kept: they say the mirror has nothing, which is
# false.
CACHE_VERSION = 2


def canon_tile(tile: str) -> str:
    """'10UFB' or 't10ufb' -> 'T10UFB' (the form used in filenames).

    The app's own tile lists come from the shapefile WITHOUT the leading
    T; Sentinel-2 product names carry it. Both are correct in their own
    context, so everything here converts to one form before comparing.
    """
    t = str(tile or '').strip().upper()
    if re.fullmatch(r'[0-9]{2}[A-Z]{3}', t):
        return 'T' + t
    return t


def _key(tile: str, day: str) -> str:
    return f'{canon_tile(tile)}:{day}'


# ------------------------------------------------------- metadata parse

def _extract_cloud_from_xml(xml_text: str) -> float:
    root = ET.fromstring(xml_text)
    for path in ('.//CLOUDY_PIXEL_PERCENTAGE',
                 './/{*}CLOUDY_PIXEL_PERCENTAGE',
                 './/Cloud_Coverage_Assessment',
                 './/{*}Cloud_Coverage_Assessment'):
        el = root.find(path)
        if el is not None and el.text:
            return float(el.text)
    raise RuntimeError('CLOUDY_PIXEL_PERCENTAGE not found')


def _safe_folder(zip_filename: str) -> str:
    if zip_filename.endswith('.zip'):
        return zip_filename[:-4] + '.SAFE'
    return zip_filename + '.SAFE'


def _read_vsi(vsi_path: str) -> bytes:
    from osgeo import gdal
    f = gdal.VSIFOpenL(vsi_path, 'rb')
    if f is None:
        raise RuntimeError(f'cannot open {vsi_path}')
    try:
        gdal.VSIFSeekL(f, 0, 2)
        size = gdal.VSIFTellL(f)
        gdal.VSIFSeekL(f, 0, 0)
        return gdal.VSIFReadL(1, size, f)
    finally:
        gdal.VSIFCloseL(f)


def tile_matches_utm_zone(tile_id: str, utm_zone) -> bool:
    """As in the original: prefix match, case-insensitive, '' = keep."""
    if not utm_zone:
        return True
    return (tile_id or '').upper().startswith(str(utm_zone).upper())


def _cloud_for_product(s3_path: str, level: str = 'L2A') -> float:
    """Read one product's cloud percentage straight out of the ZIP.

    Same trick as the original: GDAL's /vsizip//vsicurl/ reads the one
    metadata file over HTTP without downloading the archive, which is
    the difference between a second and several hundred megabytes.
    """
    zip_name = s3_path.split('/')[-1]
    path_no_bucket = s3_path[len(BUCKET) + 1:]
    url = f'{S3_BASE_URL}/{path_no_bucket}'
    meta = LEVEL_METADATA.get(level, LEVEL_METADATA['L2A'])
    vsi = f'/vsizip//vsicurl/{url}/{_safe_folder(zip_name)}/{meta}'
    return _extract_cloud_from_xml(_read_vsi(vsi).decode('utf-8'))


# ------------------------------------------------------------ discovery

def _tile_of(product_path: str) -> str:
    for part in product_path.split('/')[-1].split('_'):
        if part.startswith('T') and len(part) == 6:
            return part
    return ''


def _list_via_http(prefix: str) -> list:
    """List one prefix using the bucket's public REST API.

    The original script uses s3fs. Depending on it here would make a
    missing package silently disable the whole feature on a server that
    is otherwise fine, so this speaks to the same bucket over plain
    HTTP and keeps s3fs as a fallback rather than a requirement.
    """
    import urllib.parse
    import urllib.request
    keys = []
    token = ''
    for _ in range(20):                      # bounded: ~20k objects
        q = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if token:
            q['continuation-token'] = token
        url = f'{S3_BASE_URL}/?{urllib.parse.urlencode(q)}'
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read()
        root = ET.fromstring(body)
        ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        for c in root.findall('.//s3:Contents/s3:Key', ns):
            if c.text:
                keys.append(c.text)
        trunc = root.find('.//s3:IsTruncated', ns)
        nxt = root.find('.//s3:NextContinuationToken', ns)
        if (trunc is None or (trunc.text or '').lower() != 'true'
                or nxt is None or not nxt.text):
            break
        token = nxt.text
    return keys


def _list_via_s3fs(prefix: str) -> list:
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    return [k[len(BUCKET) + 1:] if k.startswith(BUCKET + '/') else k
            for k in fs.ls(f'{BUCKET}/{prefix}')]


def _products_for_day(day: str, tiles, level: str = 'L2A',
                      utm_zone=None) -> list:
    """Products on the mirror for one day, restricted to *tiles*."""
    root = LEVEL_PREFIXES.get(level, L2A_PREFIX)
    prefix = f'{root}/{day[:4]}/{day[4:6]}/{day[6:8]}/'
    keys = None
    first_err = None
    for lister in (_list_via_http, _list_via_s3fs):
        try:
            keys = lister(prefix)
            break
        except Exception as exc:
            if first_err is None:
                first_err = f'{type(exc).__name__}: {exc}'
    if keys is None:
        raise RuntimeError(f'listing {day}: {first_err}')
    out = []
    for key in keys:
        if not key.endswith('.zip'):
            continue
        t = _tile_of(key)
        if t and t in tiles and tile_matches_utm_zone(t, utm_zone):
            out.append(f'{BUCKET}/{key}')
    return out


# ----------------------------------------------------------- public API

def cached_coverage(cache_root: str, tiles, days) -> dict:
    """``{day: (mean, n_with_data, n_tiles)}`` from the cache alone.

    Averaged over the tiles that HAVE a figure, not only over days
    where every tile does. Sentinel-2 does not image every tile on
    every pass, so an AOI spanning four tiles routinely has two or
    three of them on a given day -- and requiring all four meant those
    days never resolved, were re-queried on every run, and sat striped
    for ever.

    The count travels with the mean so the caller can say what the
    average is over rather than implying full coverage.
    """
    data = _load(cache_root)
    tiles = sorted(set(canon_tile(t) for t in (tiles or []) if t))
    out = {}
    for day in days or []:
        vals = []
        for t in tiles:
            v = data.get(_key(t, day))
            if isinstance(v, dict):
                v = v.get('pct')
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[day] = (sum(vals) / len(vals), len(vals), len(tiles))
    return out


def cached_percentages(cache_root: str, tiles, days) -> dict:
    """``{day: percent}`` -- the mean only, for callers that want it."""
    return {d: v[0]
            for d, v in cached_coverage(cache_root, tiles, days).items()}


def pending_days(cache_root: str, tiles, days) -> list:
    """Days that would actually be looked up, newest first.

    The caller needs this to decide whether to wait: a day whose tiles
    are all recorded as "no product" is answered, even though it has no
    number, and treating it as outstanding is what made the dialog poll
    for ever.
    """
    tiles = sorted(set(canon_tile(t) for t in (tiles or []) if t))
    data = _load(cache_root)
    now = time.time()
    out = []
    for day in sorted(set(d for d in (days or []) if d), reverse=True):
        for t in tiles:
            v = data.get(_key(t, day))
            if isinstance(v, dict):
                if v.get('pct') is not None:
                    continue
                if (v.get('empty_at')
                        and now - float(v['empty_at']) < _EMPTY_TTL_S):
                    continue
            elif isinstance(v, (int, float)):
                continue
            out.append(day)
            break
    return out


def fetch_percentages(cache_root: str, tiles, days,
                      log=None, level: str = 'L2A',
                      workers: int = None, single_thread: bool = False,
                      use_cache: bool = True, utm_zone=None) -> dict:
    """Fill in whatever is missing, then return everything known.

    Incremental by construction: a (tile, day) already in the cache is
    never fetched again, so opening the dialog a second time costs
    nothing and adding one new date costs only that date.
    """
    tiles = sorted(set(canon_tile(t) for t in (tiles or []) if t))
    tiles = [t for t in tiles if tile_matches_utm_zone(t, utm_zone)]
    days = sorted(set(d for d in (days or []) if d), reverse=True)
    if not tiles or not days:
        return {}

    data = _load(cache_root) if use_cache else {}
    now = time.time()

    # What still needs looking up -- the same test the caller uses, so
    # "is there work?" and "do the work" can never disagree.
    todo = (pending_days(cache_root, tiles, days) if use_cache
            else list(days))

    if not todo:
        # Nothing to look up, yet the caller asked. That means every
        # requested day is already answered -- including days answered
        # as "no products". Saying so is the difference between a
        # feature that is finished and one that appears hung.
        with _lock:
            _last_run[key_for(tiles)] = {
                'finished_at': time.time(), 'attempted': 0,
                'with_data': 0, 'no_products': len(days),
                'failed': 0, 'error': '',
                'note': 'all requested dates already answered',
            }
        if log:
            log(f'[cloud] nothing to look up: all {len(days)} date(s) '
                f'already answered for {len(tiles)} tile(s)')
        return cached_percentages(cache_root, tiles, days)

    if log:
        log(f'[cloud] {len(todo)} day(s) to look up for '
            f'{len(tiles)} tile(s): {", ".join(tiles[:8])}'
            + (' ...' if len(tiles) > 8 else ''))
    pkey = key_for(tiles)
    with _lock:
        _progress[pkey] = {'done': 0, 'total': len(todo),
                           'started': time.time(), 'errors': 0,
                           'tiles': len(tiles), 'day': ''}

    def _one_day(day):
        found = {}
        try:
            products = _products_for_day(day, tiles, level=level,
                                         utm_zone=utm_zone)
        except Exception as exc:
            return day, None, str(exc)
        if not products:
            return day, {}, None
        for s3_path in products:
            t = _tile_of(s3_path)
            try:
                found[t] = _cloud_for_product(s3_path, level=level)
            except Exception as exc:
                sys.stderr.write(
                    f'[cloud] {os.path.basename(s3_path)}: {exc}\n')
        return day, found, None

    # Newest first. The operator is nearly always after recent
    # imagery, so the dates they will look at first are the ones that
    # resolve first; a long backfill fills in behind them.
    todo = sorted(todo, reverse=True)

    results = []
    n_workers = int(workers or N_WORKERS)
    if single_thread:
        # The original's --single-thread, kept for the same reason:
        # a failing mirror is far easier to diagnose without a pool
        # swallowing the order of events.
        for d in todo:
            try:
                r = _one_day(d)
                results.append(r)
                with _lock:
                    pr = _progress.get(pkey)
                    if pr is not None:
                        pr['done'] = int(pr.get('done', 0)) + 1
                        pr['day'] = r[0]
                        if r[2]:
                            pr['errors'] = int(pr.get('errors', 0)) + 1
            except Exception as exc:
                sys.stderr.write(f'[cloud] {d} failed: {exc}\n')
        n_workers = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
            if n_workers == 0:
                raise StopIteration
            futs = {ex.submit(_one_day, d): d for d in todo}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    results.append(r)
                    with _lock:
                        pr = _progress.get(pkey)
                        if pr is not None:
                            pr['done'] = int(pr.get('done', 0)) + 1
                            pr['day'] = r[0]
                            if r[2]:
                                pr['errors'] = int(pr.get('errors', 0)) + 1
                except Exception as exc:
                    sys.stderr.write(f'[cloud] worker failed: {exc}\n')
                    with _lock:
                        pr = _progress.get(pkey)
                        if pr is not None:
                            pr['done'] = int(pr.get('done', 0)) + 1
                            pr['errors'] = int(pr.get('errors', 0)) + 1
    except StopIteration:
        pass                       # single-threaded run already done
    except Exception as exc:
        sys.stderr.write(f'[cloud] pool failed: {exc}\n')

    changed = 0
    for day, found, err in results:
        if err:
            continue
        # The listing SUCCEEDED, so every tile without a product
        # genuinely has none for this day -- Sentinel-2 simply did not
        # image it. Recording that is what stops the day being looked
        # up again on every dialog open; leaving those tiles blank is
        # why partially-covered days were re-queried for ever.
        for t in tiles:
            if t in found:
                data[_key(t, day)] = {'pct': round(float(found[t]), 2),
                                      'at': now}
            else:
                data[_key(t, day)] = {'pct': None, 'empty_at': now}
            changed += 1

    if changed:
        _save(cache_root, data)
        if log:
            log(f'[cloud] cached {changed} tile-day value(s)')
    ok_days = sum(1 for _d, f, e in results if not e and f)
    empty_days = sum(1 for _d, f, e in results if not e and not f)
    err_days = [(d, e) for d, f, e in results if e]
    with _lock:
        _progress.pop(pkey, None)
        _last_run[pkey] = {
            'finished_at': time.time(),
            'attempted': len(todo),
            'with_data': ok_days,
            'no_products': empty_days,
            'failed': len(err_days),
            'error': (err_days[0][1] if err_days else ''),
        }
    if err_days and log:
        log(f'[cloud] {len(err_days)} of {len(todo)} day(s) failed; '
            f'first: {err_days[0][1][:160]}')
    if log:
        log(f'[cloud] done: {ok_days} day(s) with data, '
            f'{empty_days} with no products, {len(err_days)} failed')
    return cached_percentages(cache_root, tiles, days)


def fetch_in_background(cache_root: str, tiles, days, key: str,
                        log=None) -> bool:
    """Start one background fill for *key*; True if this call started it.

    Keyed so that reopening the dialog while a fill is running joins the
    existing work instead of starting a second one against the same
    mirror.
    """
    with _lock:
        t = _inflight.get(key)
        if t is not None and t.is_alive():
            return False

        def _run():
            try:
                fetch_percentages(cache_root, tiles, days, log=log)
            except Exception as exc:
                sys.stderr.write(f'[cloud] background fill failed: '
                                 f'{exc}\n')

        th = threading.Thread(target=_run, daemon=True,
                              name=f'cloud-{key[:24]}')
        _inflight[key] = th
        th.start()
        return True


def is_fetching(key: str) -> bool:
    with _lock:
        t = _inflight.get(key)
        return bool(t is not None and t.is_alive())
