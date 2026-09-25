"""Memory and storage figures for the Memory panel, sampled in the background.

One process-wide set of sampler threads -- one per row -- each refreshing
its own figure about once a second and then sleeping. Requests never
query a device: /api/memory only reads the latest figures, so every page,
tab and user sees the same numbers and the number of viewers costs
nothing. The samplers start once (ensure_started is idempotent and
thread-safe) and run for the life of the server.

Rows, all in MB:
  RAM         physical memory (MemTotal / MemAvailable)
  swap        all swap devices and files together (SwapTotal / SwapFree)
  /ram        the ramdisk the AOI stacks live on
  SSD         the volume this application's own code is on
  /data       the magnetic storage holding the Sentinel-2 archive
  store, output, /tmp
              the durable store, the output root and the temporary
              directory -- only when one is on a volume of its own
  GPU n       each CUDA device's memory (via nvidia-smi, as KGC uses)
  GDAL cache  GDAL's raster block cache in this server (GDAL_CACHEMAX)
"""

import os
import shutil
import subprocess
import sys
import threading
import time

REFRESH_S = 1.0
_MB = 1048576.0

_lock = threading.Lock()
_rows = {}               # row name -> latest figures
_order = []              # display order of row names
_started = False


def _publish(name, total_b, used_b, free_b, where, extra=None):
    row = {'type': name,
           'capacity_mb': round(total_b / _MB, 1),
           'used_mb': round(used_b / _MB, 1),
           'free_mb': round(free_b / _MB, 1),
           'pct': (round(100.0 * used_b / total_b, 1) if total_b > 0
                   else None),
           'where': where, 'at': time.time()}
    if extra:
        row.update(extra)
    with _lock:
        _rows[name] = row
        if name not in _order:
            _order.append(name)


def _publish_error(name, where, why):
    with _lock:
        _rows[name] = {'type': name, 'capacity_mb': None, 'used_mb': None,
                       'free_mb': None, 'pct': None, 'where': where,
                       'error': why, 'at': time.time()}
        if name not in _order:
            _order.append(name)


def _loop(name, sample):
    """One sampler: sample, publish, sleep, forever. Never raises."""
    while True:
        try:
            sample()
        except Exception as exc:
            try:
                _publish_error(name, '', f'{type(exc).__name__}: {exc}')
            except Exception:
                pass
        time.sleep(REFRESH_S)


def _meminfo() -> dict:
    out = {}
    with open('/proc/meminfo', encoding='ascii') as fh:
        for line in fh:
            k, _, v = line.partition(':')
            parts = v.split()
            if parts:
                try:
                    out[k.strip()] = int(parts[0]) * 1024     # kB -> bytes
                except ValueError:
                    pass
    return out


def _sample_ram():
    m = _meminfo()
    total = m.get('MemTotal', 0)
    avail = m.get('MemAvailable', m.get('MemFree', 0))
    _publish('RAM', total, max(0, total - avail), avail,
             'physical memory; the ramdisk /ram is part of it')


def _sample_swap():
    m = _meminfo()
    total, free = m.get('SwapTotal', 0), m.get('SwapFree', 0)
    where = ('all swap devices and files together' if total
             else 'no swap is configured')
    _publish('swap', total, max(0, total - free), free, where)


def _mount_point(path: str) -> str:
    p = os.path.realpath(path)
    while not os.path.ismount(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def _disk_sampler(name, path, what):
    def sample():
        du = shutil.disk_usage(path)
        # used + free = capacity: blocks reserved for root count as used,
        # since the application cannot write to them either.
        _publish(name, du.total, du.total - du.free, du.free,
                 f'{what}: {path} (volume {_mount_point(path)})')
    return sample


def _gpu_sampler():
    exe = shutil.which('nvidia-smi')

    def sample():
        p = subprocess.run(
            [exe, '--query-gpu=index,name,memory.total,memory.used,'
                  'memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, timeout=10)
        if p.returncode != 0:
            raise RuntimeError('nvidia-smi failed')
        for line in (p.stdout or b'').decode().strip().splitlines():
            f = [x.strip() for x in line.split(',')]
            if len(f) < 5:
                continue
            idx, gname = f[0], f[1]
            total, used, free = (float(f[2]) * _MB, float(f[3]) * _MB,
                                 float(f[4]) * _MB)
            _publish(f'GPU {idx}', total, used, free,
                     f'{gname}; used by KGC clustering')
    return sample if exe else None


def _sample_gdal():
    from osgeo import gdal
    cap = int(gdal.GetCacheMax())
    used = int(gdal.GetCacheUsed())
    _publish('GDAL cache', cap, used, max(0, cap - used),
             'GDAL raster block cache in this server (GDAL_CACHEMAX)')


def _volumes(output_root: str = ''):
    """(name, path, what) for each storage row, one row per volume."""
    out, seen = [], set()

    def add(name, path, what, always=False):
        # /ram, SSD and /data always get their row, even on a machine
        # where two of them share a volume; the others only when they
        # are on a volume no earlier row already shows.
        try:
            if not path or not os.path.isdir(path):
                return
            dev = os.stat(path).st_dev
        except OSError:
            return
        if dev in seen and not always:
            return
        seen.add(dev)
        out.append((name, path, what))

    try:
        from .aoi_stack import RAM_DIR
    except Exception:
        RAM_DIR = '/ram'
    add('/ram', RAM_DIR, 'ramdisk for the AOI stacks', always=True)
    # The volume the application itself runs from -- found, not assumed.
    add('SSD', os.path.dirname(os.path.abspath(__file__)),
        'volume holding the application', always=True)
    add('/data', '/data', 'magnetic storage (Sentinel-2 archive)',
        always=True)
    try:
        from .durable import store_dir
        add('store', store_dir() or '', 'durable product store')
    except Exception:
        pass
    add('output', output_root or '', 'output root')
    try:
        import tempfile
        add('/tmp', tempfile.gettempdir(), 'temporary files')
    except Exception:
        pass
    return out


def ensure_started(output_root: str = '') -> None:
    """Start the samplers once for the whole server (idempotent)."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    samplers = [('RAM', _sample_ram), ('swap', _sample_swap)]
    for name, path, what in _volumes(output_root):
        samplers.append((name, _disk_sampler(name, path, what)))
    gs = _gpu_sampler()
    if gs:
        samplers.append(('GPU', gs))
    samplers.append(('GDAL cache', _sample_gdal))
    for name, fn in samplers:
        try:
            fn()                    # first figures before anyone asks
        except Exception as exc:
            _publish_error(name, '', f'{type(exc).__name__}: {exc}')
        threading.Thread(target=_loop, args=(name, fn), daemon=True,
                         name=f'memory-{name}').start()
    sys.stderr.write('[memory] %d sampler thread(s) started: %s\n'
                     % (len(samplers), ', '.join(n for n, _ in samplers)))


def snapshot(output_root: str = '') -> dict:
    """The latest figures, in display order."""
    ensure_started(output_root)
    now = time.time()
    with _lock:
        rows = [dict(_rows[n]) for n in _order if n in _rows]
    for r in rows:
        r['age_s'] = round(max(0.0, now - r.get('at', now)), 1)
    return {'rows': rows, 'refresh_s': REFRESH_S, 'at': now}
