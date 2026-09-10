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
L2A_PREFIX = 'Sentinel-2/S2MSI2A'

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
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(cache_root: str, data: dict) -> None:
    p = cache_path(cache_root)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except OSError as exc:
        sys.stderr.write(f'[cloud] could not save cache: {exc}\n')


def _key(tile: str, day: str) -> str:
    return f'{tile}:{day}'


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


def _cloud_for_product(s3_path: str) -> float:
    """Read one product's cloud percentage straight out of the ZIP.

    Same trick as the original: GDAL's /vsizip//vsicurl/ reads the one
    metadata file over HTTP without downloading the archive, which is
    the difference between a second and several hundred megabytes.
    """
    zip_name = s3_path.split('/')[-1]
    path_no_bucket = s3_path[len(BUCKET) + 1:]
    url = f'{S3_BASE_URL}/{path_no_bucket}'
    vsi = f'/vsizip//vsicurl/{url}/{_safe_folder(zip_name)}/MTD_MSIL2A.xml'
    return _extract_cloud_from_xml(_read_vsi(vsi).decode('utf-8'))


# ------------------------------------------------------------ discovery

def _tile_of(product_path: str) -> str:
    for part in product_path.split('/')[-1].split('_'):
        if part.startswith('T') and len(part) == 6:
            return part
    return ''


def _products_for_day(day: str, tiles) -> list:
    """L2A products on the mirror for one day, restricted to *tiles*."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    prefix = f'{BUCKET}/{L2A_PREFIX}/{day[:4]}/{day[4:6]}/{day[6:8]}/'
    out = []
    try:
        for obj in fs.ls(prefix):
            if not obj.endswith('.zip'):
                continue
            t = _tile_of(obj)
            if t and t in tiles:
                out.append(obj)
    except Exception as exc:
        raise RuntimeError(f'listing {day}: {exc}')
    return out


# ----------------------------------------------------------- public API

def cached_percentages(cache_root: str, tiles, days) -> dict:
    """Cloud cover already known, without touching the network.

    Returns ``{day: percent}`` averaged over the AOI's tiles, including
    only days where every tile is known -- a partial average would move
    as the rest arrived, and a figure that changes under the operator is
    worse than one that is briefly absent.
    """
    data = _load(cache_root)
    tiles = list(tiles or [])
    out = {}
    for day in days or []:
        vals = []
        for t in tiles:
            v = data.get(_key(t, day))
            if isinstance(v, dict):
                v = v.get('pct')
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals and len(vals) == len(tiles):
            out[day] = sum(vals) / len(vals)
    return out


def fetch_percentages(cache_root: str, tiles, days,
                      log=None) -> dict:
    """Fill in whatever is missing, then return everything known.

    Incremental by construction: a (tile, day) already in the cache is
    never fetched again, so opening the dialog a second time costs
    nothing and adding one new date costs only that date.
    """
    tiles = sorted(set(t for t in (tiles or []) if t))
    days = sorted(set(d for d in (days or []) if d), reverse=True)
    if not tiles or not days:
        return {}

    data = _load(cache_root)
    now = time.time()

    # What still needs looking up.
    todo = []
    for day in days:
        for t in tiles:
            v = data.get(_key(t, day))
            if isinstance(v, dict):
                if v.get('pct') is not None:
                    continue
                # A previous "none found" answer, still within its
                # lifetime: leave it alone rather than re-scanning a day
                # the mirror has nothing for.
                if (v.get('empty_at')
                        and now - float(v['empty_at']) < _EMPTY_TTL_S):
                    continue
            elif isinstance(v, (int, float)):
                continue
            todo.append(day)
            break

    if not todo:
        return cached_percentages(cache_root, tiles, days)

    if log:
        log(f'[cloud] {len(todo)} day(s) to look up for '
            f'{len(tiles)} tile(s)')
    pkey = ','.join(tiles)
    with _lock:
        _progress[pkey] = {'done': 0, 'total': len(todo),
                           'started': time.time(), 'errors': 0,
                           'tiles': len(tiles), 'day': ''}

    def _one_day(day):
        found = {}
        try:
            products = _products_for_day(day, tiles)
        except Exception as exc:
            return day, None, str(exc)
        if not products:
            return day, {}, None
        for s3_path in products:
            t = _tile_of(s3_path)
            try:
                found[t] = _cloud_for_product(s3_path)
            except Exception as exc:
                sys.stderr.write(
                    f'[cloud] {os.path.basename(s3_path)}: {exc}\n')
        return day, found, None

    results = []
    try:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
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
    except Exception as exc:
        sys.stderr.write(f'[cloud] pool failed: {exc}\n')

    changed = 0
    for day, found, err in results:
        if err:
            continue
        if not found:
            # Nothing on the mirror for this day. Recorded so the same
            # empty answer is not re-derived on every dialog open.
            for t in tiles:
                data[_key(t, day)] = {'pct': None, 'empty_at': now}
                changed += 1
            continue
        for t, pct in found.items():
            data[_key(t, day)] = {'pct': round(float(pct), 2),
                                  'at': now}
            changed += 1

    if changed:
        _save(cache_root, data)
        if log:
            log(f'[cloud] cached {changed} tile-day value(s)')
    with _lock:
        _progress.pop(pkey, None)
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
