"""Sentinel-2 acquisition plans: fetch, parse, cache, and query.

ESA publishes, per satellite (S2A/S2B/S2C), a KML of PLANNED datatakes
covering roughly the next 18 days. Each datatake carries a UTC time
span and a swath footprint, so intersecting an AOI against them answers
"when is this fire next imaged, and which part of it first?".

Cached on the ramdisk and refreshed on startup and once a day. The
plans extend well beyond a day, so a daily refresh is ample; a failed
refresh keeps serving the previous cache rather than losing the
feature.

Deliberately conservative about what it promises:

* ESA states the KML footprints are SIMPLIFIED -- the four corners of
  the strip linked up -- and may not match product boundaries exactly.
  Everything here is therefore an expectation, not a guarantee, and is
  labelled that way in the UI.
* Only NOBS (nominal observation) datatakes yield distributed
  products. RAW/VIC/DARK/TEST/MSMOON folders are parsed but excluded
  from predictions.
* Plans are revised and republished weekly, so a cached plan can go
  stale in content even while still being "fresh" by age.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET

KML_NS = '{http://www.opengis.net/kml/2.2}'

PLANS_INDEX = ('https://sentinels.copernicus.eu/copernicus/sentinel-2/'
               'acquisition-plans')
PLANS_DIR = '/ram/s2_acq_plans'
PLANS_JSON = os.path.join(PLANS_DIR, 'plans.json')

# Modes whose datatakes become products users can download.
DISTRIBUTED_MODES = ('NOBS',)

REFRESH_INTERVAL_S = 24 * 3600
_HTTP_TIMEOUT = 60

_lock = threading.Lock()
_cache = None          # parsed plans, in memory
_refresh_thread = None

# Live progress, so the UI can show what is happening instead of a bare
# "not downloaded yet". The refresh runs in the background at startup,
# so a page opened in the first seconds legitimately finds no cache --
# a state worth reporting, not an error.
_status = {
    'state': 'idle',        # idle | running | ok | error
    'message': 'Acquisition plans have not been fetched yet.',
    'started_at': None,
    'finished_at': None,
    'satellites': {},       # sat -> {state, bytes, datatakes, error}
    'total': 0,
    'done': 0,
    'detail': '',           # URL / proxy / remedy, shown in the UI
}


def status() -> dict:
    """Snapshot of the current/last refresh, plus cache facts."""
    with _lock:
        st = json.loads(json.dumps(_status))
    c = load_cache()
    st['has_cache'] = bool(c and c.get('datatakes'))
    st['cache_age_s'] = cache_age_s() if st['has_cache'] else None
    st['datatake_count'] = len(c.get('datatakes', [])) if c else 0
    return st


def _set_status(**kw):
    with _lock:
        _status.update(kw)


def _set_sat(sat, **kw):
    with _lock:
        cur = _status['satellites'].setdefault(sat, {})
        cur.update(kw)
        _status['done'] = sum(
            1 for v in _status['satellites'].values()
            if v.get('state') in ('ok', 'error'))


# ---------------------------------------------------------------- fetch

def describe_exc(exc) -> str:
    """A message that is never empty.

    A bare TimeoutError (and OSError, and Exception) stringifies to the
    empty string, so f'...: {exc}' produced 'Could not read the plans
    index:' with nothing after the colon -- an error report carrying no
    information. Always include the type, and unwrap the reasons urllib
    hides inside URLError.
    """
    parts = [type(exc).__name__]
    txt = str(exc).strip()
    if txt:
        parts.append(txt)
    reason = getattr(exc, 'reason', None)
    if reason is not None and str(reason).strip() and \
            str(reason).strip() != txt:
        parts.append(f'reason={reason}')
    code = getattr(exc, 'code', None)
    if code is not None:
        parts.append(f'HTTP {code}')
    if type(exc).__name__ in ('TimeoutError', 'timeout') and not txt:
        parts.append(f'no response within {_HTTP_TIMEOUT}s -- the '
                     f'server may have no outbound HTTPS route to '
                     f'sentinels.copernicus.eu, or needs a proxy')
    return ': '.join(parts)


def proxy_info() -> str:
    """Proxy environment, since that is the usual cause of timeouts."""
    got = {k: os.environ.get(k) for k in
           ('https_proxy', 'HTTPS_PROXY', 'http_proxy', 'HTTP_PROXY',
            'no_proxy', 'NO_PROXY')}
    got = {k: v for k, v in got.items() if v}
    return ', '.join(f'{k}={v}' for k, v in got.items()) or 'none set'


def _http_get(url: str, timeout: int = _HTTP_TIMEOUT,
              attempts: int = 3, log=None) -> bytes:
    """GET with retries and errors that say what actually happened."""
    last = None
    for i in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': ('Mozilla/5.0 (compatible; '
                               'wps-research/fire-mapping)'),
                'Accept': '*/*',
            })
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if not data:
                raise OSError('empty response body')
            return data
        except Exception as exc:
            last = exc
            msg = (f'      [acq] attempt {i}/{attempts} failed for '
                   f'{url}: {describe_exc(exc)}')
            sys.stderr.write(msg + '\n')
            if log:
                try:
                    log(msg)
                except Exception:
                    pass
            if i < attempts:
                time.sleep(min(5, 2 ** (i - 1)))
    raise last if last else OSError('unknown fetch failure')


def discover_kml_urls(index_html: str = None) -> dict:
    """Newest KML URL per satellite, from the plans index page.

    The page lists files newest-first per satellite, and the filename
    encodes the validity window, so the first match per satellite is
    the current plan.
    """
    if index_html is None:
        index_html = _http_get(PLANS_INDEX).decode('utf-8', 'replace')

    out = {}
    # e.g. .../documents/d/sentinel/s2a_mp_acq__kml_20260806t150000_20260824t180000
    pat = re.compile(
        r'href="([^"]*?/(s2[abc])_mp_acq__kml_(\d{8}t\d{6})_'
        r'(\d{8}t\d{6}))"', re.I)
    for m in pat.finditer(index_html):
        href, sat, t0, t1 = m.group(1), m.group(2).upper(), \
            m.group(3), m.group(4)
        if not href.startswith('http'):
            href = 'https://sentinels.copernicus.eu' + href
        if sat not in out:            # first = newest
            out[sat] = {'url': href, 'valid_from': t0, 'valid_to': t1}
    return out


# ---------------------------------------------------------------- parse

def _text(node, tag):
    el = node.find(KML_NS + tag)
    return el.text.strip() if el is not None and el.text else ''


def _extended(node) -> dict:
    """ExtendedData Data name=/value pairs as a dict."""
    out = {}
    ed = node.find(KML_NS + 'ExtendedData')
    if ed is None:
        return out
    for d in ed.findall(KML_NS + 'Data'):
        name = d.get('name') or ''
        val = d.find(KML_NS + 'value')
        if name:
            out[name] = (val.text or '').strip() if val is not None else ''
    return out


def _rings(node) -> list:
    """Outer ring coordinates as [[lon, lat], ...] lists."""
    rings = []
    for poly in node.iter(KML_NS + 'Polygon'):
        for ob in poly.iter(KML_NS + 'outerBoundaryIs'):
            for lr in ob.iter(KML_NS + 'LinearRing'):
                txt = _text(lr, 'coordinates')
                if not txt:
                    continue
                pts = []
                for tok in txt.replace('\n', ' ').split():
                    parts = tok.split(',')
                    if len(parts) >= 2:
                        try:
                            pts.append([float(parts[0]), float(parts[1])])
                        except ValueError:
                            pass
                if len(pts) >= 4:
                    rings.append(pts)
    return rings


def parse_kml(data) -> list:
    """Datatakes from one acquisition-plan KML.

    Returns dicts with satellite, id, mode, start/stop (ISO UTC),
    relative orbit, scene count and the footprint ring.
    """
    if isinstance(data, bytes):
        data = data.decode('utf-8', 'replace')
    root = ET.fromstring(data)

    out = []

    def walk(node, sat, mode):
        """Descend Document/Folder, collecting Placemarks.

        Satellite comes from the S2A/S2B/S2C folder and mode from the
        folder beneath it; both are inherited downward, because the
        real files nest satellite -> mode -> sub-mode -> Placemark.
        """
        for child in list(node):
            tag = child.tag
            if tag in (KML_NS + 'Document', KML_NS + 'Folder'):
                name = _text(child, 'name')
                c_sat, c_mode = sat, mode
                if re.fullmatch(r'S2[ABC]', name or '', re.I):
                    c_sat = name.upper()
                elif name and sat and not mode:
                    c_mode = name.upper()
                walk(child, c_sat, c_mode)
            elif tag == KML_NS + 'Placemark':
                ext = _extended(child)
                rings = _rings(child)
                if not rings:
                    continue
                ts = child.find(KML_NS + 'TimeSpan')
                begin = _text(ts, 'begin') if ts is not None else ''
                end = _text(ts, 'end') if ts is not None else ''
                out.append({
                    'sat': sat or '',
                    'id': ext.get('ID') or _text(child, 'name'),
                    'mode': (ext.get('Mode') or mode or '').upper(),
                    'start': ext.get('ObservationTimeStart') or begin,
                    'stop': ext.get('ObservationTimeStop') or end,
                    'orbit_rel': ext.get('OrbitRelative') or '',
                    'orbit_abs': ext.get('OrbitAbsolute') or '',
                    'scenes': ext.get('Scenes') or '',
                    'ring': rings[0],
                })

    walk(root, None, None)
    # Folder-level satellite may be absent on some files; fall back to
    # the document name, which states the mission.
    return out


# ------------------------------------------------------------ cache I/O

def _write_cache(payload: dict) -> None:
    os.makedirs(PLANS_DIR, exist_ok=True)
    tmp = PLANS_JSON + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    os.replace(tmp, PLANS_JSON)


def load_cache() -> dict:
    """Cached plans, from memory or the ramdisk."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(PLANS_JSON, encoding='utf-8') as f:
            _cache = json.load(f)
    except (OSError, ValueError):
        _cache = None
    return _cache


def cache_age_s() -> float:
    c = load_cache()
    if not c:
        return float('inf')
    return max(0.0, time.time() - float(c.get('fetched_at') or 0))


def _load_local_kmls(emit) -> dict:
    """Parse any *.kml an operator placed in PLANS_DIR.

    A deliberate escape hatch: if the host cannot reach ESA (no route,
    proxy, firewall), the plans can still be downloaded elsewhere and
    dropped in, and everything downstream works unchanged.
    """
    global _cache
    try:
        if not os.path.isdir(PLANS_DIR):
            return {}
        files = [f for f in sorted(os.listdir(PLANS_DIR))
                 if f.lower().endswith('.kml')]
        if not files:
            return {}
        datatakes, sources = [], {}
        for f in files:
            path = os.path.join(PLANS_DIR, f)
            try:
                with open(path, 'rb') as fh:
                    dts = parse_kml(fh.read())
            except Exception as exc:
                emit(f'      [acq] local {f}: parse failed: '
                     f'{describe_exc(exc)}')
                continue
            m = re.search(r's2([abc])', f, re.I)
            sat = ('S2' + m.group(1).upper()) if m else ''
            for d in dts:
                if not d.get('sat'):
                    d['sat'] = sat
            datatakes.extend(dts)
            sources[sat or f] = {'url': f'file://{path}'}
            emit(f'      [acq] local {f}: {len(dts)} datatake(s)')
        if not datatakes:
            return {}
        payload = {'fetched_at': time.time(), 'sources': sources,
                   'datatakes': datatakes, 'local': True,
                   'local_files': len(sources)}
        try:
            _write_cache(payload)
        except OSError:
            pass
        _cache = payload
        return payload
    except Exception as exc:
        emit(f'      [acq] local KML scan failed: {describe_exc(exc)}')
        return {}


def refresh(force: bool = False, log=None) -> dict:
    """Fetch and cache the current plan for every satellite.

    Failure is non-fatal: the previous cache keeps being served, since
    a stale plan is far more useful than none.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    global _cache
    if not force and cache_age_s() < REFRESH_INTERVAL_S:
        return load_cache() or {}

    _set_status(state='running', started_at=time.time(),
                finished_at=None, satellites={}, total=0, done=0,
                message='Reading ESA acquisition-plan index ...')
    try:
        urls = discover_kml_urls()
    except Exception as exc:
        detail = describe_exc(exc)
        emit(f'      [acq] could not read the plans index: {detail}')
        emit(f'      [acq] proxy env: {proxy_info()}')
        # Falling back to KMLs an operator dropped in by hand keeps the
        # feature usable on a host with no route to ESA.
        local = _load_local_kmls(emit)
        if local:
            _set_status(
                state='ok', finished_at=time.time(),
                message=f'Using {local.get("local_files", 0)} manually '
                        f'supplied KML file(s) from {PLANS_DIR} '
                        f'({len(local.get("datatakes", []))} '
                        f'datatakes); the ESA index was unreachable.',
                detail=f'Network error: {detail}')
            return local
        _set_status(
            state='error', finished_at=time.time(),
            message=f'Could not read the plans index -- {detail}',
            detail=(f'URL: {PLANS_INDEX}  |  proxy env: '
                    f'{proxy_info()}  |  drop KML files into '
                    f'{PLANS_DIR} to use them without network access'))
        return load_cache() or {}
    if not urls:
        emit('      [acq] plans index listed no KML files; '
             'keeping the previous cache.')
        _set_status(state='error', finished_at=time.time(),
                    message='The plans index listed no KML files.')
        return load_cache() or {}

    _set_status(total=len(urls),
                message=f'Downloading {len(urls)} acquisition plan(s) '
                        f'in parallel ...')
    for sat in urls:
        _set_sat(sat, state='pending', bytes=0, datatakes=0)

    def _one(item):
        """Fetch and parse one satellite's plan."""
        sat, meta = item
        _set_sat(sat, state='downloading')
        try:
            raw = _http_get(meta['url'])
            _set_sat(sat, state='parsing', bytes=len(raw))
            dts = parse_kml(raw)
            for d in dts:
                if not d.get('sat'):
                    d['sat'] = sat
            n_dist = sum(1 for d in dts if d['mode'] in DISTRIBUTED_MODES)
            _set_sat(sat, state='ok', datatakes=len(dts),
                     distributed=n_dist)
            emit(f'      [acq] {sat}: {len(dts)} datatake(s), '
                 f'{n_dist} distributed, {len(raw) / 1e6:.1f} MB, '
                 f'valid {meta["valid_from"]} - {meta["valid_to"]}')
            return sat, meta, dts, None
        except Exception as exc:
            _set_sat(sat, state='error', error=describe_exc(exc))
            emit(f'      [acq] {sat}: fetch/parse failed: '
                 f'{describe_exc(exc)}')
            return sat, meta, [], exc

    # The three satellite plans are independent files of a few MB, so
    # fetching them concurrently turns three serial round trips into
    # roughly one.
    datatakes, sources = [], {}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        for sat, meta, dts, err in pool.map(_one, sorted(urls.items())):
            if dts:
                datatakes.extend(dts)
                sources[sat] = meta

    if not datatakes:
        emit('      [acq] no datatakes parsed; keeping the '
             'previous cache.')
        _set_status(state='error', finished_at=time.time(),
                    message='All downloads failed. '
                            'The previous plan is still in use.')
        return load_cache() or {}

    payload = {'fetched_at': time.time(),
               'sources': sources,
               'datatakes': datatakes}
    with _lock:
        try:
            _write_cache(payload)
        except OSError as exc:
            emit(f'      [acq] could not write cache: {exc}')
        _cache = payload
    n_dist = sum(1 for d in datatakes if d['mode'] in DISTRIBUTED_MODES)
    emit(f'      [acq] cached {len(datatakes)} planned datatake(s) '
         f'({n_dist} distributed) from {len(sources)} satellite(s) '
         f'-> {PLANS_JSON}')
    _set_status(state='ok', finished_at=time.time(),
                message=f'{len(sources)} plan(s), {n_dist} distributed '
                        f'datatakes.')
    return payload


def start_background_refresh() -> None:
    """Refresh on startup, then daily."""
    global _refresh_thread
    if _refresh_thread is not None:
        return

    def _loop():
        while True:
            try:
                refresh(force=True)
            except Exception as exc:
                sys.stderr.write(f'[acq] refresh loop error: {exc}\n')
            time.sleep(REFRESH_INTERVAL_S)

    _refresh_thread = threading.Thread(target=_loop, daemon=True)
    _refresh_thread.start()


# ------------------------------------------------------------- querying

def next_coverage(aoi_ring_native, srs_wkt, geotransform, width, height,
                  horizon_days: float = 14.0, now_ts: float = None,
                  modes=DISTRIBUTED_MODES) -> dict:
    """Earliest planned coverage of an AOI, split by pass.

    Walks future datatakes in time order and, for each, takes the part
    of the AOI not already covered by an earlier pass. That mirrors the
    L2 coverage-by-acquisition plot: an AOI is often not re-imaged all
    at once, so what matters is which PART arrives WHEN.

    Geometry is returned in crop pixel coordinates so the client can
    draw it exactly like the existing plot.
    """
    from osgeo import ogr, osr

    plans = load_cache()
    if not plans or not plans.get('datatakes'):
        # Report the refresh state so the caller can show progress
        # rather than an unexplained absence.
        return {'error': 'no acquisition plans cached',
                'status': status(),
                'passes': [], 'width': width, 'height': height}

    now_ts = now_ts if now_ts is not None else time.time()

    def _iso_to_ts(s):
        if not s:
            return None
        try:
            from datetime import datetime, timezone
            s2 = s.replace('Z', '')
            fmt = '%Y-%m-%dT%H:%M:%S.%f' if '.' in s2 else \
                  '%Y-%m-%dT%H:%M:%S'
            return datetime.strptime(s2, fmt).replace(
                tzinfo=timezone.utc).timestamp()
        except Exception:
            return None

    # AOI polygon in its native CRS.
    aoi = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in aoi_ring_native:
        aoi.AddPoint_2D(float(x), float(y))
    aoi.CloseRings()
    aoi_poly = ogr.Geometry(ogr.wkbPolygon)
    aoi_poly.AddGeometry(aoi)
    aoi_area = aoi_poly.GetArea() or 1.0

    tgt = osr.SpatialReference()
    tgt.ImportFromWkt(srs_wkt)
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    try:
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass
    tx = osr.CoordinateTransformation(src, tgt)

    # Inverse geotransform: native -> pixel.
    gt = geotransform
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if not det:
        return {'error': 'degenerate geotransform', 'passes': []}

    def to_px(x, y):
        dx, dy = x - gt[0], y - gt[3]
        return ((dx * gt[5] - dy * gt[2]) / det,
                (-dx * gt[4] + dy * gt[1]) / det)

    horizon = now_ts + horizon_days * 86400.0
    cand = []
    for d in plans['datatakes']:
        if modes and d.get('mode') not in modes:
            continue
        t0 = _iso_to_ts(d.get('start'))
        if t0 is None or t0 < now_ts or t0 > horizon:
            continue
        cand.append((t0, d))
    cand.sort(key=lambda p: p[0])

    passes = []
    claimed = None
    for t0, d in cand:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for lon, lat in d['ring']:
            ring.AddPoint_2D(float(lon), float(lat))
        ring.CloseRings()
        swath = ogr.Geometry(ogr.wkbPolygon)
        swath.AddGeometry(ring)
        try:
            swath.Transform(tx)
        except Exception:
            continue
        if not swath.Intersects(aoi_poly):
            continue
        piece = swath.Intersection(aoi_poly)
        if piece is None or piece.IsEmpty():
            continue
        if claimed is not None:
            piece = piece.Difference(claimed)
            if piece is None or piece.IsEmpty():
                continue
        frac = (piece.GetArea() or 0.0) / aoi_area
        if frac < 0.002:                 # ignore slivers
            continue

        rings_px = []
        for i in range(piece.GetGeometryCount() or 1):
            g = piece.GetGeometryRef(i) if piece.GetGeometryCount() \
                else piece
            if g is None:
                continue
            for j in range(g.GetGeometryCount() or 1):
                r = g.GetGeometryRef(j) if g.GetGeometryCount() else g
                if r is None or r.GetPointCount() < 3:
                    continue
                rings_px.append([list(to_px(*r.GetPoint_2D(k)))
                                 for k in range(r.GetPointCount())])

        if not rings_px:
            continue
        passes.append({
            'sat': d.get('sat', ''),
            'id': d.get('id', ''),
            'start': d.get('start', ''),
            'stop': d.get('stop', ''),
            'orbit_rel': d.get('orbit_rel', ''),
            'start_ts': t0,
            'fraction': frac,
            'rings': rings_px,
        })
        claimed = piece if claimed is None else claimed.Union(piece)
        if claimed.GetArea() / aoi_area > 0.999:
            break

    covered = sum(p['fraction'] for p in passes)
    return {
        'width': width,
        'height': height,
        'passes': passes,
        'covered_fraction': covered,
        'horizon_days': horizon_days,
        'plans_fetched_at': plans.get('fetched_at'),
        'plans_age_s': cache_age_s(),
    }
