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

# Manually supplied KMLs are also looked for next to this module, which
# unlike the ramdisk survives a reboot -- otherwise the offline
# workaround has to be repeated after every restart.
PLANS_DIR_PERSISTENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 's2_acq_plans')


def _local_kml_dirs():
    seen, out = set(), []
    for d in (PLANS_DIR, PLANS_DIR_PERSISTENT):
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out

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
    'insecure': False,      # set when TLS could not be verified
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
    rtxt = str(reason).strip() if reason is not None else ''
    # urllib wraps the real error, so str(URLError) already CONTAINS
    # the reason. Appending it again doubled every message.
    if rtxt and rtxt != txt and rtxt not in txt:
        parts.append(f'reason={rtxt}')
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


# TLS to sentinels.copernicus.eu can fail with
# "unable to get issuer certificate": the server sends an incomplete
# chain, or a middlebox re-signs it, and Python's store cannot complete
# the path. curl usually carries a fuller CA bundle and often succeeds
# where urllib does not, so several transports are tried in order and
# the one that worked is logged.
#
# Verification is NEVER disabled implicitly. ACQ_PLANS_INSECURE=1 is an
# explicit operator choice for this one public, non-sensitive dataset,
# and it says loudly what it is doing.
def _strategies():
    yield 'urllib (system CA)', _get_urllib, {}
    try:
        import certifi
        yield ('urllib (certifi CA)', _get_urllib,
               {'cafile': certifi.where()})
    except Exception:
        pass
    for var in ('ACQ_PLANS_CAFILE', 'SSL_CERT_FILE',
                'REQUESTS_CA_BUNDLE'):
        ca = os.environ.get(var)
        if ca and os.path.isfile(ca):
            yield f'urllib (CA from ${var})', _get_urllib, {'cafile': ca}
    yield 'curl', _get_curl, {}
    # Last resort: unverified TLS.
    #
    # Some networks intercept HTTPS with a self-signed root whose
    # certificate has EXPIRED. Nothing on this host can repair that,
    # so a strictly-verifying client can never reach ESA -- the
    # feature simply would not work.
    #
    # This is allowed here, and ONLY here, because of what the data
    # is: three public, non-secret KML files of published satellite
    # timings. Nothing secret is sent (no credentials, no tokens), and
    # the worst case from a substituted response is a wrong
    # "next coverage" prediction -- it cannot affect fire mapping
    # output. Every verified transport is tried first, the fallback is
    # logged loudly, the UI marks the data as unverified, and
    # validate_plan_content() rejects anything that is not a plausible
    # S2 acquisition plan.
    #
    # Set ACQ_PLANS_STRICT_TLS=1 to forbid the fallback entirely.
    if os.environ.get('ACQ_PLANS_STRICT_TLS', '').strip() not in (
            '1', 'true', 'yes'):
        yield ('urllib (UNVERIFIED TLS)', _get_urllib,
               {'insecure': True})
        yield ('curl (UNVERIFIED TLS)', _get_curl, {'insecure': True})


def _get_urllib(url, timeout, cafile=None, insecure=False):
    import ssl
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    elif cafile:
        ctx = ssl.create_default_context(cafile=cafile)
    else:
        ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={
        'User-Agent': ('Mozilla/5.0 (compatible; '
                       'wps-research/fire-mapping)'),
        'Accept': '*/*',
    })
    with urllib.request.urlopen(req, timeout=timeout,
                                context=ctx) as r:
        return r.read()


def _get_curl(url, timeout, insecure=False):
    import subprocess
    cmd = ['curl', '-sS', '-L', '--fail',
           '--max-time', str(timeout),
           '-A', 'Mozilla/5.0 (compatible; wps-research/fire-mapping)']
    if insecure:
        cmd.append('-k')
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        err = (p.stderr or b'').decode('utf-8', 'replace').strip()
        raise OSError(f'curl exit {p.returncode}: {err or "no stderr"}')
    if not p.stdout:
        raise OSError('curl returned an empty body')
    return p.stdout


# Remembers the transport that worked, so subsequent fetches in the
# same run skip the ones already known to fail.
_working_strategy = None


def _http_get(url: str, timeout: int = _HTTP_TIMEOUT,
              attempts: int = 2, log=None) -> bytes:
    """GET over whichever transport can complete the TLS handshake."""
    global _working_strategy

    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    strategies = list(_strategies())
    if _working_strategy:
        strategies.sort(key=lambda s: s[0] != _working_strategy)

    errors = []
    for name, fn, kw in strategies:
        for i in range(1, attempts + 1):
            try:
                data = fn(url, timeout, **kw)
                if not data:
                    raise OSError('empty response body')
                if _working_strategy != name:
                    emit(f'      [acq] transport OK via {name}')
                    _working_strategy = name
                if 'UNVERIFIED' in name and not _status.get('insecure'):
                    _set_status(insecure=True)
                    emit('      [acq] NOTE: the TLS certificate could '
                         'not be verified (your network intercepts '
                         'HTTPS with an expired certificate). The '
                         'plan was fetched anyway because it is '
                         'public, non-sensitive data, and its content '
                         'is validated below. Set '
                         'ACQ_PLANS_STRICT_TLS=1 to forbid this.')
                return data
            except Exception as exc:
                detail = describe_exc(exc)
                errors.append(f'{name}: {detail}')
                emit(f'      [acq] {name} attempt {i}/{attempts} '
                     f'failed: {detail}')
                if i < attempts:
                    time.sleep(1)
    # De-duplicate: every transport reports the same handshake failure
    # once per attempt, which turned the message into four copies of
    # one fact.
    uniq, seen = [], set()
    for e in errors:
        key = e.split(':', 1)[-1].strip()[:120]
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    raise OSError('all transports failed -- ' + ' | '.join(uniq[:3]))


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


def validate_plan_content(datatakes, sat_hint='') -> tuple:
    """Sanity-check parsed datatakes before they are trusted.

    The fetch can fall back to unverified TLS, so the response is
    checked for being a plausible Sentinel-2 acquisition plan rather
    than taken on faith. This is not a substitute for TLS -- it cannot
    detect a subtly altered plan -- but it does stop a wholesale
    substitution (error page, wrong file, corrupted download) from
    being cached and displayed as fact.

    Returns (ok, reason).
    """
    if not datatakes:
        return False, 'no datatakes parsed'
    if len(datatakes) < 10:
        return False, (f'only {len(datatakes)} datatakes -- a real '
                       f'plan has thousands')

    n_dist = 0
    for d in datatakes:
        ring = d.get('ring') or []
        if len(ring) < 4:
            return False, f'datatake {d.get("id")} has a degenerate ring'
        for lon, lat in ring:
            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                return False, (f'datatake {d.get("id")} has an '
                               f'out-of-range coordinate '
                               f'({lon}, {lat})')
        if not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}',
                        d.get('start') or ''):
            return False, (f'datatake {d.get("id")} has an unparseable '
                           f'start time {d.get("start")!r}')
        if d.get('mode') in DISTRIBUTED_MODES:
            n_dist += 1
        sat = d.get('sat') or ''
        if sat and not re.fullmatch(r'S2[ABC]', sat):
            return False, f'unexpected satellite {sat!r}'

    if n_dist == 0:
        return False, 'no NOBS (distributed) datatakes present'
    return True, f'{len(datatakes)} datatakes, {n_dist} distributed'


def _load_local_kmls(emit) -> dict:
    """Parse any *.kml an operator placed in PLANS_DIR.

    A deliberate escape hatch: if the host cannot reach ESA (no route,
    proxy, firewall), the plans can still be downloaded elsewhere and
    dropped in, and everything downstream works unchanged.
    """
    global _cache
    try:
        pairs = []
        for d in _local_kml_dirs():
            for f in sorted(os.listdir(d)):
                if f.lower().endswith('.kml'):
                    pairs.append((d, f))
        if not pairs:
            return {}
        datatakes, sources = [], {}
        for d, f in pairs:
            path = os.path.join(d, f)
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
        tls = 'CERTIFICATE_VERIFY' in detail or 'SSL' in detail
        # A self-signed issuer in the chain means something is
        # re-signing the connection: a TLS-inspecting firewall or
        # proxy. Combined with "expired", that appliance's own
        # certificate is out of date. Neither is fixable at ESA's end
        # or in this code, so say so plainly rather than sending the
        # operator to chase CA bundles that are not the problem.
        intercepted = ('self-signed' in detail.lower()
                       or 'self signed' in detail.lower())
        expired = 'expired' in detail.lower()
        if intercepted or expired:
            what = []
            if intercepted:
                what.append('a self-signed issuer appears in the '
                            'chain, so HTTPS is being intercepted and '
                            're-signed on your network')
            if expired:
                what.append('the presented certificate has EXPIRED')
            remedy = (
                'TLS interception detected: ' + '; '.join(what) + '. '
                'ESA\'s own certificate is not the problem and no CA '
                'bundle here will help. Options: (1) ask whoever runs '
                'the firewall/proxy to renew its certificate, or to '
                'exempt sentinels.copernicus.eu from inspection; '
                '(2) install the appliance\'s root CA on this server '
                'and point ACQ_PLANS_CAFILE at it -- note this will '
                'still fail while that certificate is expired; '
                f'(3) download the three KMLs on a machine outside '
                f'this network and drop them into {PLANS_DIR} or '
                f'{PLANS_DIR_PERSISTENT}, which needs no network at '
                f'all; (4) run with --acq_plans_insecure (or '
                f'ACQ_PLANS_INSECURE=1) to skip verification for this '
                f'one public dataset -- acceptable here only because '
                f'the KMLs are public and non-sensitive, and it means '
                f'trusting whatever is intercepting.')
        elif tls:
            remedy = (
                'TLS chain could not be verified. Every transport was '
                'tried (system CA, certifi, $SSL_CERT_FILE, curl). '
                'Fixes, best first: (1) install the missing '
                'intermediate CA on the server, e.g. '
                '"sudo update-ca-certificates"; (2) point '
                'ACQ_PLANS_CAFILE at a bundle that includes it; '
                '(3) download the three KMLs elsewhere and drop them '
                f'into {PLANS_DIR}; (4) last resort, set '
                'ACQ_PLANS_INSECURE=1 to skip verification for this '
                'one public dataset.')
        else:
            remedy = (f'Drop KML files into {PLANS_DIR} to use them '
                      f'without network access.')
        _set_status(
            state='error', finished_at=time.time(),
            message=f'Could not read the plans index -- {detail}',
            detail=(f'URL: {PLANS_INDEX}  |  proxy env: '
                    f'{proxy_info()}  |  {remedy}'))
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
            ok, why = validate_plan_content(dts, sat)
            if not ok:
                raise OSError(f'content failed validation ({why}) -- '
                              f'refusing to cache it')
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
    with _lock:
        insecure = bool(_status.get('insecure'))
    _set_status(
        state='ok', finished_at=time.time(),
        message=f'{len(sources)} plan(s), {n_dist} distributed '
                f'datatakes.'
                + (' TLS could not be verified on this network; '
                   'content was validated instead.' if insecure else ''),
        detail=('Fetched over an intercepted HTTPS connection whose '
                'certificate is expired and self-signed. The content '
                'was checked for being a plausible S2 plan before use. '
                'Set ACQ_PLANS_STRICT_TLS=1 to refuse this.'
                if insecure else ''))
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
                  horizon_days: float = 21.0, now_ts: float = None,
                  modes=DISTRIBUTED_MODES, max_passes: int = 10) -> dict:
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
        # FULL footprint of this pass over the AOI. Earlier versions
        # subtracted what previous passes already covered, so a pass
        # that re-images ground shown above it vanished from the plot
        # entirely. Listing each pass's whole footprint means a
        # full-frame pass reads as one solid colour, and every date
        # still appears in the legend.
        frac = (piece.GetArea() or 0.0) / aoi_area
        if frac < 0.002:                 # ignore slivers
            continue
        # 'new_fraction' is what this pass adds beyond earlier ones --
        # the number that answers "when is the AOI fully re-imaged".
        if claimed is not None:
            extra = piece.Difference(claimed)
            new_frac = ((extra.GetArea() or 0.0) / aoi_area
                        if extra is not None and not extra.IsEmpty()
                        else 0.0)
        else:
            new_frac = frac

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
            'new_fraction': new_frac,
            'rings': rings_px,
        })
        claimed = piece if claimed is None else claimed.Union(piece)
        # Keep going past full coverage: the user wants the next N
        # opportunities, not merely the minimum set that tiles the AOI.
        if len(passes) >= max_passes:
            break

    covered = sum(p['new_fraction'] for p in passes)
    # When the AOI is first fully covered.
    full_ts, acc = None, 0.0
    for p in passes:
        acc += p['new_fraction']
        if acc > 0.999:
            full_ts = p['start_ts']
            break
    return {
        'width': width,
        'height': height,
        'passes': passes,
        'covered_fraction': covered,
        'full_coverage_ts': full_ts,
        'max_passes': max_passes,
        'horizon_days': horizon_days,
        'plans_fetched_at': plans.get('fetched_at'),
        'plans_age_s': cache_age_s(),
        'insecure': bool(_status.get('insecure')),
    }
