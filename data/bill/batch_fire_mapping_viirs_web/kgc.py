"""KGC clustering: build, run, and collect results.

An alternative to the t-SNE + Random Forest + HDBSCAN pipeline. It
takes the same two inputs -- the AOI stack (with the same band
exclusions applied) and the selected hint mask -- and produces the same
output: a classified mask that is brushed, overlaid, scored and
accepted exactly like the existing pipeline's.

The point of this module is that everything AFTER the clustering is
shared. KGC writes a six-band product; band 1 is the binary selected
class, and once that is extracted as ``<fire>_classified.bin`` it is
indistinguishable from what the CLI produces. Brushing, agreement
scoring, the overlay, the results gallery, Accept, polygonisation and
export then all work unchanged rather than needing a parallel path.

Source lives in ``wps-research/cpp/kgc2``; the binary is built on
demand with the flags from its own ``run_all.sh``.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

from .state import AppState, FireStatus

state: AppState = None

# Defaults taken from the Args struct in kgc.cpp. Kept here so the UI
# can show them without parsing C++, and so a change there is a visible
# diff here rather than a silent behaviour change.
KGC_DEFAULTS = {
    'kgc_nskip': -1,             # -1 = choose from budget_points
    'kgc_kmax': 10000,
    'kgc_kstep': 5,
    'kgc_patience': 0,
    'kgc_min_class': -1,         # -1 = no minimum
    'kgc_budget_points': 20000,
    'kgc_threads': -1,           # -1 = one per CPU
    'kgc_no_cache': False,
    # Force the CPU build regardless of GPU availability. Not a kgc
    # flag -- it selects which binary runs, so it is deliberately not
    # in _KGC_FLAGS below.
    'kgc_cpu_only': False,
    # Run both builds back to back and report the difference. Neither
    # is a kgc flag: both select what the SERVER does, so they are
    # deliberately absent from _KGC_FLAGS.
    'kgc_compare': False,
}

# CLI flag for each parameter, and whether -1/0 means "let kgc decide"
# (in which case the flag is omitted rather than passed explicitly).
_KGC_FLAGS = [
    ('kgc_nskip', '--nskip', -1),
    ('kgc_kmax', '--kmax', None),
    ('kgc_kstep', '--kstep', None),
    ('kgc_patience', '--patience', None),
    ('kgc_min_class', '--min-class', -1),
    ('kgc_budget_points', '--budget-points', None),
    ('kgc_threads', '--threads', -1),
]


# Progress stages for a KGC run, in order. Deliberately separate from
# viirs_worker.STAGES, which lists the PREPARATION stages -- passing a
# mapping stage to _set_progress() yields stage_idx 0 of 3 and a
# progress bar that never moves.
KGC_STAGES = ('kgc_build', 'kgc_stack', 'kgc_cluster', 'kgc_classify',
              'kgc_brush', 'kgc_figure')


def set_kgc_progress(fire, stage: str, detail: str = '',
                     fraction=None) -> None:
    """Write a progress snapshot for a KGC run.

    Mirrors the shape viirs_worker._set_progress produces -- including
    the ETA and stall fields the fire list reads -- so the same UI
    renders it without a special case.
    """
    try:
        idx = KGC_STAGES.index(stage) + 1
    except ValueError:
        idx = 0
    now = time.time()
    snap = {
        'stage': stage,
        'stage_idx': idx,
        'total_stages': len(KGC_STAGES),
        'detail': detail,
        'updated_at': now,
        'kind': 'kgc',
    }
    if fraction is not None:
        try:
            snap['fraction'] = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            pass
    try:
        with state.lock:
            prev = getattr(fire, 'progress', None) or {}
            started = prev.get('started_at') or now
            if prev.get('stage') != stage:
                started = now
            snap['started_at'] = started
            changed = (prev.get('stage') != stage
                       or prev.get('detail') != detail)
            snap['last_change_at'] = (
                now if changed else (prev.get('last_change_at') or now))
            elapsed = max(0.0, now - started)
            snap['elapsed_s'] = elapsed
            f = snap.get('fraction')
            if f and f > 0.02 and elapsed > 2.0:
                snap['eta_s'] = max(0.0, elapsed / f - elapsed)
            fire.progress = snap
    except Exception:
        pass


def init(app_state: AppState):
    global state
    state = app_state


def kgc_dir() -> str:
    """Where the KGC sources live, derived from the CLI script path.

    The web app sits at ``<repo>/data/bill/batch_fire_mapping_viirs_web``
    and the sources at ``<repo>/cpp/kgc2``, so the repo root is found by
    walking up from this file rather than hard-coding a path that would
    break on a different checkout.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..', '..'))
    return os.path.join(repo, 'cpp', 'kgc2')


def estimate_memory(n_points: int, kmax: int, dim: int,
                    batch: int = 256) -> dict:
    """Bytes the run needs, on the host and on a GPU.

    The neighbour table dominates everything: n x kmax entries of a
    float distance plus a size_t index. The GPU additionally needs the
    points, one batch of the distance matrix, and CUB's sort scratch --
    but NOT the whole table, which is streamed back to the host as it is
    produced.

    This is what decides which build runs. The host has far more memory
    than the card, so the GPU is used only when its working set fits
    with margin; otherwise the CPU build is correct and merely slower.
    """
    n, k, d = max(1, int(n_points)), max(1, int(kmax)), max(1, int(dim))
    k = min(k, n)
    host_table = n * k * (4 + 8)          # float dist + size_t index
    host_points = n * d * 4
    gpu_points = n * d * 4
    # keys in/out + values in/out for one batch, plus CUB scratch which
    # is roughly the same size again.
    gpu_batch = batch * n * (4 * 2 + 4 * 2)
    gpu_total = gpu_points + gpu_batch * 2
    return {
        'n_points': n, 'kmax': k, 'dim': d, 'batch': batch,
        'host_bytes': host_table + host_points,
        'host_table_bytes': host_table,
        'gpu_bytes': gpu_total,
        'host_gb': (host_table + host_points) / 1e9,
        'gpu_gb': gpu_total / 1e9,
    }


def gpu_free_bytes() -> int:
    """Free memory on the first CUDA device, or 0 if there is none."""
    try:
        p = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, timeout=15)
        if p.returncode != 0:
            return 0
        first = (p.stdout or b'').decode().strip().splitlines()
        return int(float(first[0].strip())) * 1024 * 1024 if first else 0
    except Exception:
        return 0


def choose_kgc_build(params: dict, n_points: int, dim: int,
                     log=None) -> tuple:
    """(exe_path, which, note) -- pick the GPU build when it fits.

    Deliberately conservative: a run that exceeds device memory does not
    fail cleanly, it thrashes or is killed, and from the UI that is
    indistinguishable from slow. Requiring 1.5x headroom costs some
    speed on borderline cases and avoids that failure entirely.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    # An explicit request wins over the memory model. This exists so
    # the two builds can be compared on identical inputs -- the whole
    # point is to take the automatic choice out of the picture, so it
    # is checked before anything else and reported plainly.
    if (params or {}).get('kgc_cpu_only'):
        emit('  CPU only requested; the GPU build is not considered')
        return ensure_kgc_binary(log=log), 'cpu', 'forced by the user'

    kmax = int((params or {}).get('kgc_kmax')
               or KGC_DEFAULTS['kgc_kmax'])
    est = estimate_memory(n_points, kmax, dim)
    emit(f'  memory model: host {est["host_gb"]:.1f} GB '
         f'(table {est["host_table_bytes"] / 1e9:.1f} GB), '
         f'GPU working set {est["gpu_gb"]:.1f} GB '
         f'[{est["n_points"]:,} points x k_max {est["kmax"]:,} '
         f'x {est["dim"]} dims]')

    free = gpu_free_bytes()
    if free <= 0:
        emit('  no CUDA device visible; using the CPU build')
        return ensure_kgc_binary(log=log), 'cpu', 'no GPU'
    need = int(est['gpu_bytes'] * 1.5)
    if need > free:
        emit(f'  GPU has {free / 1e9:.1f} GB free but the run wants '
             f'~{need / 1e9:.1f} GB with headroom; using the CPU build')
        return ensure_kgc_binary(log=log), 'cpu', 'insufficient GPU memory'
    try:
        exe = ensure_kgc_gpu_binary(log=log)
        emit(f'  GPU build selected ({free / 1e9:.1f} GB free)')
        return exe, 'gpu', ''
    except Exception as exc:
        emit(f'  GPU build unavailable ({exc}); using the CPU build')
        return ensure_kgc_binary(log=log), 'cpu', str(exc)


def ensure_kgc_gpu_binary(log=None) -> str:
    """Path to a built ``kgc_gpu``, compiling the .cu if needed.

    Rebuilt whenever either source is newer than the binary: kgc.cu
    #includes kgc.cpp, so a change to the CPU file changes the GPU build
    too and checking only the .cu would silently ship a stale binary.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    d = kgc_dir()
    cu = os.path.join(d, 'kgc.cu')
    exe = os.path.join(d, 'kgc_gpu')
    if not os.path.isfile(cu):
        raise RuntimeError(f'no CUDA source at {cu}')

    # kgc.cu is SELF-CONTAINED: it no longer includes kgc.cpp, so the
    # CPU source is not a dependency of this build. The two are
    # deliberately independent -- kgc.cpp is the reference
    # implementation and nothing here may change how it behaves.
    fresh = False
    try:
        fresh = (os.path.isfile(exe)
                 and os.path.getmtime(exe) >= os.path.getmtime(cu))
    except OSError:
        fresh = False
    if fresh and os.access(exe, os.X_OK):
        return exe

    # Match the device rather than assuming. sm_89 is Ada (L40S).
    arch = 'sm_89'
    try:
        p = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap',
             '--format=csv,noheader'], capture_output=True, timeout=15)
        cc = (p.stdout or b'').decode().strip().splitlines()
        if cc:
            arch = 'sm_' + cc[0].strip().replace('.', '')
    except Exception:
        pass

    # -fmad=false is not optional. Contracting a*b+c into an FMA changes
    # the low bits of the squared distance, which reorders near-ties in
    # the neighbour list and therefore changes the classification. The
    # CPU build does not contract, so the GPU must not either.
    cmd = ['nvcc', '-O3', '-std=c++14', f'-arch={arch}', '-fmad=false',
           '-Xcompiler', '-pthread', 'kgc.cu', '-o', 'kgc_gpu']
    emit(f'  Building the GPU KGC: {" ".join(cmd)} (in {d})')
    t0 = time.time()
    p = subprocess.run(cmd, cwd=d, capture_output=True)
    if p.returncode != 0:
        err = (p.stderr or b'').decode('utf-8', 'replace').strip()
        raise RuntimeError(
            f'nvcc failed (exit {p.returncode}):\n{err[-2000:]}')
    emit(f'  GPU KGC built in {time.time() - t0:.1f}s -> {exe}')
    return exe


def ensure_kgc_binary(log=None) -> str:
    """Path to a built ``kgc``, compiling it if needed.

    Rebuilds when the binary is missing or older than the source. That
    matters here: ``class_brush.exe`` silently misparsed its arguments
    for months because a flag was added to the source and the binary
    was never rebuilt, and the failure mode was an unrelated-looking
    error. Checking mtime makes that impossible for this tool.

    Raises RuntimeError with the compiler output on failure, so the
    reason reaches the fire console instead of only the server log.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    d = kgc_dir()
    src = os.path.join(d, 'kgc.cpp')
    exe = os.path.join(d, 'kgc')
    if not os.path.isfile(src):
        raise RuntimeError(
            f'KGC source not found at {src}. Expected the repo to '
            f'contain cpp/kgc2/kgc.cpp.')

    fresh = False
    try:
        fresh = (os.path.isfile(exe)
                 and os.path.getmtime(exe) >= os.path.getmtime(src))
    except OSError:
        fresh = False
    if fresh and os.access(exe, os.X_OK):
        return exe

    cxx = os.environ.get('CXX', 'c++')
    cmd = [cxx, '-O3', '-std=c++11', '-pthread', 'kgc.cpp', '-o', 'kgc']
    emit(f'  Building KGC: {" ".join(cmd)} (in {d})')
    t0 = time.time()
    p = subprocess.run(cmd, cwd=d, capture_output=True)
    if p.returncode != 0:
        err = (p.stderr or b'').decode('utf-8', 'replace').strip()
        raise RuntimeError(
            f'KGC build failed (exit {p.returncode}):\n{err[-2000:]}')
    emit(f'  KGC built in {time.time() - t0:.1f}s -> {exe}')
    return exe


def purge_kgc_caches(image: str, why: str = '', log=None) -> int:
    """Delete kgc's memoised dedup and neighbour-table files for *image*.

    kgc writes ``<image>.kgc_dedup`` and ``<image>.kgc_knn_s<N>_k<K>``
    beside its INPUT, and the key is a checksum of that input plus
    n_skip and k_max. Nothing in the key identifies which build produced
    the table -- so a CPU run and a GPU run on the same stack share it,
    and whichever runs second silently reuses the other's neighbours.

    That makes a CPU-vs-GPU comparison meaningless and can carry a bad
    table across a backend switch. Purging from the server keeps
    kgc.cpp untouched, which matters because it is the reference
    implementation.
    """
    import glob as _glob
    n = 0
    base = os.path.splitext(image)[0]
    for pat in (image + '.kgc_dedup', image + '.kgc_knn_*',
                base + '.kgc_dedup', base + '.kgc_knn_*'):
        for f in _glob.glob(pat):
            try:
                os.remove(f)
                n += 1
            except OSError:
                pass
    if n:
        msg = (f'  cleared {n} kgc cache file(s) for '
               f'{os.path.basename(image)}'
               + (f' ({why})' if why else ''))
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
    return n


def _backend_stamp_path(image: str) -> str:
    return image + '.kgc_backend'


def enforce_backend_cache(image: str, which: str, log=None) -> None:
    """Purge the caches when the backend differs from the last run.

    A sidecar records which build last wrote the tables for this image.
    Same backend -> the cache is legitimately reusable and is kept, so
    repeat runs stay fast. Different backend -> purge, because the two
    may order equidistant neighbours differently and the key cannot tell
    them apart.
    """
    stamp = _backend_stamp_path(image)
    prev = ''
    try:
        with open(stamp, encoding='utf-8') as f:
            prev = f.read().strip()
    except OSError:
        prev = ''
    if prev and prev != which:
        purge_kgc_caches(image, f'backend changed {prev} -> {which}',
                         log=log)
    try:
        with open(stamp, 'w', encoding='utf-8') as f:
            f.write(which)
    except OSError:
        pass


def build_kgc_cmd(exe: str, image: str, hint: str, out_prefix: str,
                  params: dict) -> list:
    """Command line for one KGC run.

    Sentinel values are omitted rather than passed: ``--nskip -1`` is
    not the same as leaving it out, since kgc derives the stride from
    budget_points when nskip is unset.
    """
    cmd = [exe, image, '--hint', hint, '--out', out_prefix]
    for key, flag, sentinel in _KGC_FLAGS:
        raw = (params or {}).get(key, KGC_DEFAULTS.get(key))
        if raw is None or raw == '':
            continue
        try:
            val = int(raw)
        except (TypeError, ValueError):
            continue
        if sentinel is not None and val == sentinel:
            continue
        cmd += [flag, str(val)]
    if (params or {}).get('kgc_no_cache'):
        cmd.append('--no-cache')
    return cmd


class KgcEstimator:
    """Turns kgc's own output into a description and a time estimate.

    Two things make the naive version useless. First, "KGC cluster,
    estimating" says nothing about what is happening or how big the job
    is. Second, the sweep's levels are NOT equal cost: level j evaluates
    K = j * kstep neighbours, so cumulative work grows with the SQUARE
    of the level index. A linear "1050/2000" bar understates the
    remaining wait by about 4x at a quarter of the way through.

    An estimate is available immediately from the parameters, then
    replaced by measured rates as each phase reports progress -- so the
    first number is a guess and says so, and later ones are observed.
    """

    # Calibration from a measured run: 30,190 points x 11 bands,
    # k_max 10,000, 512 CPU workers. Only used before the run reports
    # anything of its own; every later figure is measured.
    CPU_NB_OPS = 3.3e9         # distance+select ops per second
    CPU_SW_OPS = 2.3e10        # sweep ops per second
    GPU_NB_GAIN = 8.0
    GPU_SW_GAIN = 12.0

    def __init__(self, params, dim, which):
        self.dim = max(1, int(dim or 1))
        self.gpu = (which == 'gpu')
        self.kmax = int((params or {}).get('kgc_kmax')
                        or KGC_DEFAULTS['kgc_kmax'])
        self.kstep = max(1, int((params or {}).get('kgc_kstep')
                                or KGC_DEFAULTS['kgc_kstep']))
        self.budget = int((params or {}).get('kgc_budget_points')
                          or KGC_DEFAULTS['kgc_budget_points'])
        self.n = None            # retained points, known from [params]
        self.levels = None
        self.t_start = time.time()
        self.phase = 'startup'
        self.phase_start = self.t_start
        self.nb_seconds = None   # measured, once the table finishes
        self.best = ''

    # ---- modelled totals -------------------------------------------
    def _ops(self):
        n = self.n or self.budget
        lv = self.levels or max(1, self.kmax // self.kstep)
        nb = n * n * self.dim
        sw = n * self.kstep * (lv * (lv + 1) / 2.0)
        return nb, sw

    def initial_estimate(self):
        nb, sw = self._ops()
        r_nb = self.CPU_NB_OPS * (self.GPU_NB_GAIN if self.gpu else 1.0)
        r_sw = self.CPU_SW_OPS * (self.GPU_SW_GAIN if self.gpu else 1.0)
        return nb / r_nb + sw / r_sw

    def sweep_estimate_from_table(self):
        """Once the table is timed, scale the sweep by the same machine.

        Both phases run on the same hardware, so the measured neighbour
        time is a far better calibration than any constant -- it folds
        in the actual core count, clocks and memory speed.
        """
        if not self.nb_seconds:
            return None
        nb, sw = self._ops()
        if nb <= 0:
            return None
        ratio = (self.CPU_SW_OPS
                 * (self.GPU_SW_GAIN if self.gpu else 1.0))
        base = (self.CPU_NB_OPS
                * (self.GPU_NB_GAIN if self.gpu else 1.0))
        # observed ops/sec for the table, transferred to the sweep by
        # the modelled ratio between the two rates
        obs = nb / max(1e-9, self.nb_seconds)
        return sw / max(1e-9, obs * (ratio / base))

    # ---- line handling ---------------------------------------------
    def feed(self, line):
        """Return (stage, detail, fraction) or None if the line is noise."""
        t = line.strip()

        m = re.search(r'n_skip=(\d+)\s*->\s*([\d,]+)\s*retained points'
                      r'.*?\((\d+)\s*levels\)', t)
        if m:
            self.n = int(m.group(2).replace(',', ''))
            self.levels = int(m.group(3))
            est = self.initial_estimate()
            return ('kgc_cluster',
                    f'{self.n:,} points x {self.dim} bands, k_max '
                    f'{self.kmax:,}, {self.levels:,} levels '
                    f'- first estimate {_fmt_secs(est)}', 0.10)

        m = re.match(r'\[neighbours\]\s*computing\s*([\d,]+)\s*x\s*([\d,]+)',
                     t)
        if m:
            self.phase = 'neighbours'
            self.phase_start = time.time()
            return ('kgc_cluster',
                    f'building the neighbour table '
                    f'({m.group(1)} x {m.group(2)})', 0.12)

        m = re.match(r'(?:\[gpu\])?\s*neighbours\s+([\d.]+)%', t)
        if m:
            pct = float(m.group(1)) / 100.0
            el = time.time() - self.phase_start
            eta = (el / pct - el) if (pct > 0.02 and el > 1) else None
            if pct >= 0.999:
                self.nb_seconds = max(el, 1e-6)
            det = f'neighbour table {pct * 100:.0f}%'
            if eta is not None:
                det += f' - {_fmt_secs(eta)} left in this phase'
            # the table occupies 0.12..0.35 of the whole run
            return ('kgc_cluster', det, 0.12 + 0.23 * pct)

        m = re.match(r'\[sweep\]\s*(\d[\d,]*)\s*levels', t)
        if m:
            self.phase = 'sweep'
            self.phase_start = time.time()
            est = self.sweep_estimate_from_table()
            det = 'sweeping K'
            if est:
                det += f' - estimated {_fmt_secs(est)}'
            return ('kgc_cluster', det, 0.36)

        m = re.match(r'(?:\[(?:cpu|gpu)\])?\s*(\d+)/(\d+)\s+levels(.*)', t)
        if m:
            done, tot = int(m.group(1)), int(m.group(2))
            tail = m.group(3) or ''
            b = re.search(r'best K=(\d+)\s*MI=([\d.]+)', tail)
            if b:
                self.best = f'best K={b.group(1)} MI={b.group(2)}'
            # Cost-weighted: level j costs ~ j, so work done ~ j^2.
            f = (done * (done + 1.0)) / float(tot * (tot + 1.0)) if tot else 0
            el = time.time() - self.phase_start
            eta = (el / f - el) if (f > 0.001 and el > 1) else None
            det = (f'K sweep {done:,}/{tot:,} levels '
                   f'({f * 100:.0f}% of the work)')
            if self.best:
                det += f' - {self.best}'
            if eta is not None:
                det += f' - {_fmt_secs(eta)} left'
            return ('kgc_cluster', det, 0.36 + 0.44 * f)

        if t.startswith('[select] best K'):
            return ('kgc_cluster', t[9:].strip(), 0.82)
        if t.startswith('[dedup]'):
            return ('kgc_cluster', t[8:].strip(), 0.08)
        if t.startswith('[out]'):
            return ('kgc_classify', 'writing the KGC product', 0.84)
        return None


def _fmt_secs(sec) -> str:
    """Compact duration for a status line."""
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return '?'
    if sec < 60:
        return f'{sec:.0f}s'
    if sec < 3600:
        m_ = int(sec // 60)
        r = int(round(sec - m_ * 60))
        return f'{m_}m {r}s' if r else f'{m_}m'
    h_ = int(sec // 3600)
    m_ = int(round((sec - h_ * 3600) / 60))
    return f'{h_}h {m_}m' if m_ else f'{h_}h'


def ensure_float32(path: str, work_dir: str, tag: str,
                   log=None) -> str:
    """Return *path*, or a float32 copy of it beside the work dir.

    KGC accepts ENVI data type 4 only, and says so by exiting 1 with
    "only ENVI data type 4 (32-bit float) is supported; got type N".
    Hint masks are the usual offenders: a 1/0 mask is naturally written
    as Byte, and every builder that does so breaks KGC.

    Converting here rather than only fixing the writers means KGC works
    against a hint from ANY source -- including files produced before
    this change and any future builder that reaches for Byte again.

    Returns the original path when it is already float32.
    """
    try:
        from osgeo import gdal
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            return path
        dt = ds.GetRasterBand(1).DataType
        if dt == gdal.GDT_Float32:
            ds = None
            return path

        name = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(work_dir, f'{name}_f32.bin')
        drv = gdal.GetDriverByName('ENVI')
        o = drv.Create(out, ds.RasterXSize, ds.RasterYSize,
                       ds.RasterCount, gdal.GDT_Float32,
                       options=['INTERLEAVE=BSQ'])
        o.SetGeoTransform(ds.GetGeoTransform())
        proj = ds.GetProjection()
        if proj:
            o.SetProjection(proj)
        for b in range(1, ds.RasterCount + 1):
            src_b = ds.GetRasterBand(b)
            arr = src_b.ReadAsArray()
            ob = o.GetRasterBand(b)
            ob.WriteArray(arr.astype('float32'))
            desc = src_b.GetDescription()
            if desc:
                ob.SetDescription(desc)
            ob = None
        o = None
        ds = None
        hdr = os.path.splitext(out)[0] + '.hdr'
        if not os.path.isfile(hdr) and os.path.isfile(out + '.hdr'):
            os.replace(out + '.hdr', hdr)
        msg = (f'  {tag} was ENVI data type {dt}, not 4 (float32); '
               f'converted -> {os.path.basename(out)}')
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
        return out
    except Exception as exc:
        sys.stderr.write(
            f'[kgc] float32 conversion of {path} failed: {exc}\n')
        return path


def _extract_class_band(selected_bin: str, ref_raster: str,
                        out_bin: str, log=None) -> str:
    """Write band 1 of the KGC product as a standalone classified mask.

    Band 1 is the binary selected class (1 where chosen at the best K).
    Everything downstream -- brushing, agreement, overlays, area,
    polygonisation -- expects a single-band mask on the AOI grid, so
    extracting it here is what lets the rest of the system treat a KGC
    result and a CLI result identically.

    Geometry is taken from *ref_raster* rather than from the KGC output
    so the mask is guaranteed to sit on the same grid as the previews,
    which is the invariant the split view depends on.
    """
    from osgeo import gdal
    import numpy as np

    ds = gdal.Open(selected_bin, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'cannot open KGC output {selected_bin}')
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    if arr is None:
        raise RuntimeError(f'KGC output {selected_bin} has no band 1')

    ref = gdal.Open(ref_raster, gdal.GA_ReadOnly)
    if ref is None:
        raise RuntimeError(f'cannot open reference raster {ref_raster}')
    gt, proj = ref.GetGeoTransform(), ref.GetProjection()
    rw, rh = ref.RasterXSize, ref.RasterYSize
    ref = None

    if arr.shape[0] != rh or arr.shape[1] != rw:
        raise RuntimeError(
            f'KGC output is {arr.shape[1]}x{arr.shape[0]} but the AOI '
            f'stack is {rw}x{rh}; refusing to write a mask that does '
            f'not match the grid')

    # Dimensions matching is necessary but not sufficient: two rasters
    # on DIFFERENT ground can share a pixel count. Compare the
    # geotransforms too, so a mask can never be written onto the wrong
    # extent and then look "geographically incorrect" later.
    try:
        sds = gdal.Open(selected_bin, gdal.GA_ReadOnly)
        sgt = sds.GetGeoTransform() if sds else None
        sds = None
        if sgt and gt and any(abs(a - b) > 1e-6 for a, b in zip(sgt, gt)):
            # KGC copies the input's header, so a mismatch means the
            # input was not the stack we think it was.
            msg = (f'  WARNING: KGC output geotransform {sgt} differs '
                   f'from the AOI stack {gt}; using the stack\'s, but '
                   f'check which image was passed in')
            sys.stderr.write(msg + '\n')
            if log:
                log(msg)
    except Exception:
        pass

    drv = gdal.GetDriverByName('ENVI')
    out = drv.Create(out_bin, rw, rh, 1, gdal.GDT_Float32,
                     options=['INTERLEAVE=BSQ'])
    out.SetGeoTransform(gt)
    if proj:
        out.SetProjection(proj)
    band = out.GetRasterBand(1)
    band.WriteArray((np.nan_to_num(arr) > 0.5).astype('float32'))
    band.SetDescription('KGC selected class')
    band = None
    out = None

    # Normalise the header name to <name>.hdr, matching every other
    # ENVI product the app writes.
    hdr = os.path.splitext(out_bin)[0] + '.hdr'
    if not os.path.isfile(hdr) and os.path.isfile(out_bin + '.hdr'):
        os.replace(out_bin + '.hdr', hdr)
    for junk in (out_bin + '.aux.xml',):
        try:
            os.remove(junk)
        except OSError:
            pass

    # Read the file back rather than trusting the write: this is the
    # artifact everything downstream reads, and a truncated or
    # unflushed write would otherwise surface much later as a wrong
    # picture.
    try:
        chk = gdal.Open(out_bin, gdal.GA_ReadOnly)
        if chk is None:
            raise RuntimeError('could not reopen the class mask')
        if (chk.RasterXSize, chk.RasterYSize) != (rw, rh):
            raise RuntimeError(
                f'class mask reopened as {chk.RasterXSize}x'
                f'{chk.RasterYSize}, expected {rw}x{rh}')
        cgt = chk.GetGeoTransform()
        cpr = chk.GetProjection()
        chk = None
        if any(abs(a - b) > 1e-6 for a, b in zip(cgt, gt)):
            raise RuntimeError('class mask lost its geotransform')
        if proj and not cpr:
            raise RuntimeError('class mask lost its projection')
    except Exception as exc:
        raise RuntimeError(f'class mask failed verification: {exc}')

    n = int((np.nan_to_num(arr) > 0.5).sum())
    msg = (f'  KGC class mask: {n:,} selected pixel(s) of '
           f'{rw * rh:,} -> {os.path.basename(out_bin)}')
    sys.stderr.write(msg + '\n')
    if log:
        try:
            log(msg)
        except Exception:
            pass
    return out_bin


def resolve_stack_for_source(fire, want_src: str, log=None) -> str:
    """AOI stack path for *want_src*, building it if needed.

    fire.crop_bin follows whichever source the fire was last switched
    to, and that can be flipped temporarily by the preview handler
    while it builds the other source's stash. Reading crop_bin here
    therefore raced with the UI and could feed the classifier the
    source the user was not looking at -- silently, since the run
    still succeeds.

    Asking ensure_aoi_stack for the source explicitly removes the race
    and makes the KGC input match the selector the user actually set.
    """
    cur = getattr(fire, 'post_source', 'l2') or 'l2'
    if not want_src or want_src == cur:
        if fire.crop_bin and os.path.isfile(fire.crop_bin):
            return fire.crop_bin
    from .aoi_stack import ensure_aoi_stack
    info = ensure_aoi_stack(fire.fire_numbe, fire.bbox_native,
                            post_source=want_src or cur)
    path = info['path'] if isinstance(info, dict) else info
    if log:
        log(f'  KGC source: {(want_src or cur).upper()} '
            f'-> {os.path.basename(path)}')
    return path


def compare_cpu_gpu(fire, params: dict, log=None, progress=None,
                    source: str = None) -> dict:
    """Run CPU then GPU on identical inputs and report the difference.

    Every memoised product is cleared between the two runs. Without
    that the second run reuses the first's dedup and neighbour table --
    they are keyed on the image, not the backend -- and the comparison
    would be of one computation against itself.

    CPU runs first and is treated as the reference, since it is the
    implementation that has been validated. The GPU result is what
    remains in place afterwards ONLY if the two agree; otherwise the
    CPU result is restored, because leaving a disagreeing GPU mask as
    the fire's classification would quietly promote the unverified one.
    """
    import numpy as np
    from osgeo import gdal

    def emit(msg):
        sys.stderr.write(msg + '\n')
        try:
            fire.console_log.append(msg)
        except Exception:
            pass
        if log:
            try:
                log(msg)
            except Exception:
                pass

    def snapshot(path):
        d = gdal.Open(path, gdal.GA_ReadOnly)
        if d is None:
            return None
        a = d.GetRasterBand(1).ReadAsArray()
        d = None
        return None if a is None else (np.nan_to_num(a) > 0)

    results = {}
    emit('  ' + '=' * 62)
    emit('  CPU vs GPU COMPARISON: two independent runs, caches cleared')
    emit('  ' + '=' * 62)

    for which in ('cpu', 'gpu'):
        p2 = dict(params or {})
        p2['kgc_cpu_only'] = (which == 'cpu')
        # --no-cache as well as the purge: belt and braces, since the
        # purge only covers the paths we can predict.
        p2['kgc_no_cache'] = True
        p2['kgc_compare'] = False        # never recurse
        emit('')
        emit(f'  --- {which.upper()} run ---')
        t0 = time.time()
        try:
            r = run_kgc(fire, p2, log=log, progress=progress,
                        source=source)
        except Exception as exc:
            emit(f'  {which.upper()} run FAILED: '
                 f'{type(exc).__name__}: {exc}')
            results[which] = {'ok': False, 'error': str(exc)}
            continue
        clf = r.get('classified')
        mask = snapshot(clf) if clf else None
        results[which] = {
            'ok': True,
            'seconds': time.time() - t0,
            'kgc_seconds': r.get('seconds'),
            'agreement_pct': r.get('agreement_pct'),
            'ml_area_ha': r.get('ml_area_ha'),
            'mask': mask,
            'classified': clf,
        }
        # Keep a copy of the CPU mask; the GPU run overwrites the file.
        if which == 'cpu' and clf and os.path.isfile(clf):
            keep = os.path.splitext(clf)[0] + '_cpuref.bin'
            try:
                shutil.copy2(clf, keep)
                h = os.path.splitext(clf)[0] + '.hdr'
                if os.path.isfile(h):
                    shutil.copy2(h, os.path.splitext(keep)[0] + '.hdr')
                results['cpu']['ref_path'] = keep
            except OSError as exc:
                emit(f'  (could not keep a CPU reference copy: {exc})')

    a = results.get('cpu', {})
    b = results.get('gpu', {})
    emit('')
    emit('  ' + '=' * 62)
    emit('  COMPARISON RESULT')
    emit('  ' + '=' * 62)

    if not (a.get('ok') and b.get('ok')):
        emit('  One of the runs failed; no comparison is possible.')
        return {'ok': False, 'results': results}

    ma, mb = a.get('mask'), b.get('mask')
    verdict = {}
    if ma is None or mb is None or ma.shape != mb.shape:
        emit('  Masks are missing or differently shaped; cannot compare.')
    else:
        inter = int(np.count_nonzero(ma & mb))
        union = int(np.count_nonzero(ma | mb))
        diff = int(np.count_nonzero(ma ^ mb))
        na, nb = int(np.count_nonzero(ma)), int(np.count_nonzero(mb))
        iou = (inter / union) if union else 1.0
        agree_px = int(ma.size - diff)
        verdict = {'identical': diff == 0, 'iou': iou,
                   'differing_px': diff, 'cpu_px': na, 'gpu_px': nb,
                   'pixel_agreement': agree_px / float(ma.size)}
        if diff == 0:
            emit(f'  IDENTICAL: both builds selected the same '
                 f'{na:,} pixel(s).')
        else:
            emit(f'  DIFFERENT: {diff:,} of {ma.size:,} pixel(s) '
                 f'disagree ({100.0 * diff / ma.size:.4f}%)')
            emit(f'    CPU selected {na:,} px, GPU selected {nb:,} px, '
                 f'IoU {iou:.6f}')
            emit(f'    pixel agreement {100.0 * agree_px / ma.size:.4f}%')
            emit('    A small difference is expected only from exact '
                 'ties; a large one means the ports diverge.')

    emit('')
    emit('  TIMING')
    emit(f'    {"stage":<28}{"CPU":>12}{"GPU":>12}{"speedup":>10}')
    ka, kb = a.get('kgc_seconds') or 0, b.get('kgc_seconds') or 0
    ta, tb = a.get('seconds') or 0, b.get('seconds') or 0
    def _row(label, x, y):
        sp = (f'{x / y:.2f}x' if (x and y) else '-')
        emit(f'    {label:<28}{x:>11.1f}s{y:>11.1f}s{sp:>10}')
    _row('kgc executable', ka, kb)
    _row('total incl. brush/score', ta, tb)
    emit('    (the K sweep runs on the CPU in BOTH builds; the GPU '
         'accelerates')
    emit('     the neighbour table only, so the speedup is bounded by '
         'that share)')
    emit('  ' + '=' * 62)

    # Restore the CPU result unless the two agree.
    if verdict and not verdict.get('identical'):
        ref = a.get('ref_path')
        clf = b.get('classified')
        if ref and clf and os.path.isfile(ref):
            try:
                shutil.copy2(ref, clf)
                h = os.path.splitext(ref)[0] + '.hdr'
                if os.path.isfile(h):
                    shutil.copy2(h, os.path.splitext(clf)[0] + '.hdr')
                emit('  The builds disagree, so the CPU result has been '
                     'restored as this fire\'s classification.')
                from .erase import refresh_after_edit
                refresh_after_edit(fire, clf, log=log)
            except Exception as exc:
                emit(f'  (could not restore the CPU result: {exc})')

    return {'ok': True, 'verdict': verdict,
            'cpu_seconds': ta, 'gpu_seconds': tb}


def run_kgc(fire, params: dict, log=None, progress=None,
            source: str = None) -> dict:
    """Run KGC for *fire* and leave the result where the UI expects it.

    Mirrors the CLI pipeline's outputs exactly:
      * ``<fire>_classified.bin`` in the fire cache (brushed),
      * ``previews/serial_1.png`` and ``previews/result.png``,
      * one entry in ``fire.serial_results`` so it appears in the
        gallery with an Accept button,
      * agreement and ML area computed by the same helpers.

    Returns a dict describing the run.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        try:
            fire.console_log.append(msg)
        except Exception:
            pass
        if log:
            try:
                log(msg)
            except Exception:
                pass

    def step(stage, detail='', frac=None):
        if progress:
            try:
                progress(stage, detail, frac)
            except Exception:
                pass

    from .mapping_cmd import reduced_stack
    from .mapping import (_compute_agreement, _compute_ml_area,
                          _overlay_mask_on_post)

    want_src = (source or getattr(fire, 'post_source', 'l2')
                or 'l2').strip().lower()
    if want_src not in ('l2', 'mrap'):
        want_src = getattr(fire, 'post_source', 'l2') or 'l2'
    if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
        raise RuntimeError('The AOI stack is missing; re-prepare the '
                           'fire first.')
    if not fire.hint_bin or not os.path.isfile(fire.hint_bin):
        raise RuntimeError('No hint mask is available. Choose a hint '
                           'mode that can be built for this AOI.')

    step('kgc_build', 'compiling / checking the KGC binary', 0.02)
    # Which build runs is decided AFTER the stack is known, since the
    # memory model needs the point count and band count. Resolve the
    # stack first, then choose.
    exe = None

    # Same band selection the ML pipeline receives, so the two methods
    # are comparable and the exclusion checkboxes mean one thing.
    step('kgc_stack', 'preparing the band stack', 0.06)
    base_stack = resolve_stack_for_source(fire, want_src, log=emit)
    image = reduced_stack(base_stack, fire, log=emit)
    emit(f'  KGC input stack: {os.path.basename(image)}')
    emit(f'  KGC hint: {os.path.basename(fire.hint_bin)} '
         f'({getattr(fire, "hint_mode", "?")})')

    # Work on the ramdisk, in a directory named for this fire so
    # concurrent runs cannot collide and leftovers are identifiable.
    # Same ramdisk the AOI stacks use, taken from aoi_stack rather than
    # hard-coded so the two cannot disagree.
    try:
        from .aoi_stack import RAM_DIR as ram
    except Exception:
        ram = '/ram'
    key = hashlib.sha1(
        f'{fire.fire_numbe}|{want_src}|{image}'.encode(
            'utf-8')).hexdigest()[:10]
    work = os.path.join(ram, f'kgc_{fire.fire_numbe}_{key}')
    os.makedirs(work, exist_ok=True)
    # Record which stack this directory serves. Deleting a fire matches
    # on this rather than on the fire's NAME, so a later fire reusing
    # the name cannot have its working directory removed.
    try:
        with open(os.path.join(work, '.stack'), 'w',
                  encoding='utf-8') as f:
            f.write(base_stack)
    except OSError:
        pass
    out_prefix = os.path.join(work, fire.fire_numbe)

    # KGC reads ENVI type 4 only. The stack is written float32, but a
    # hint mask often is not -- that is what "got type 1" was.
    image = ensure_float32(image, work, 'input stack', log=emit)
    hint = ensure_float32(fire.hint_bin, work, 'hint mask', log=emit)

    # Remove any product from a previous run BEFORE launching.
    #
    # The work directory is keyed on the fire, source and input path --
    # not on content -- so a re-run with different settings reuses it.
    # If a run then failed to write (or wrote under a different name),
    # the extractor would happily pick up the PREVIOUS run's
    # _selected.bin and present it as the new result. That is exactly
    # the "same wrong blob came back" symptom, and it is invisible
    # because every step reports success.
    for suffix in ('_selected.bin', '_selected.hdr', '_klevels.csv',
                   '_params.txt'):
        stale = out_prefix + suffix
        try:
            if os.path.isfile(stale):
                os.remove(stale)
                emit(f'  cleared stale {os.path.basename(stale)}')
        except OSError as exc:
            emit(f'  could not clear {os.path.basename(stale)}: {exc}')

    # Point count and dimensionality drive the memory model. Points are
    # bounded by the budget; dims come from the (already band-selected,
    # already scaled) stack.
    try:
        from osgeo import gdal as _g2
        _ds2 = _g2.Open(image, _g2.GA_ReadOnly)
        _dim = _ds2.RasterCount if _ds2 is not None else 3
        _npix = ((_ds2.RasterXSize * _ds2.RasterYSize)
                 if _ds2 is not None else 0)
        _ds2 = None
    except Exception:
        _dim, _npix = 3, 0
    _budget = int((params or {}).get('kgc_budget_points')
                  or KGC_DEFAULTS['kgc_budget_points'])
    _npts = min(_budget, _npix) if _npix else _budget
    exe, _which, _note = choose_kgc_build(params, _npts, _dim, log=emit)
    emit(f'  running the {_which.upper()} build: '
         f'{os.path.basename(exe)}')
    _est = KgcEstimator(params, _dim, _which)
    step('kgc_cluster',
         f'starting the {_which.upper()} build - first estimate '
         f'{_fmt_secs(_est.initial_estimate())} for '
         f'{_npts:,} points x {_dim} bands', 0.06)
    # The memoised dedup/neighbour tables are keyed on the image only,
    # so they must not survive a backend switch.
    enforce_backend_cache(image, _which, log=emit)

    cmd = build_kgc_cmd(exe, image, hint, out_prefix, params)
    emit('  ' + ' '.join(cmd))
    step('kgc_cluster', 'clustering (the long step)', 0.15)

    # Neighbour-table size, before anything is launched.
    #
    # It is n_points x k_max x 12 bytes, so the point budget and k_max
    # multiply: 20k x 10k is ~2.4 GB, 50k x 10k is ~6 GB. A run that
    # exceeds available memory does not fail cleanly -- it thrashes, or
    # the kernel kills it, and from the UI that is indistinguishable
    # from a run that is merely slow.
    try:
        _pts = int((params or {}).get('kgc_budget_points')
                   or KGC_DEFAULTS['kgc_budget_points'])
        _kmx = int((params or {}).get('kgc_kmax')
                   or KGC_DEFAULTS['kgc_kmax'])
        _gb = _pts * _kmx * 12 / 1e9
        emit(f'  neighbour table ~{_gb:.1f} GB '
             f'({_pts:,} points x k_max {_kmx:,})')
        if _gb > 4.0:
            emit(f'  WARNING: that is large. If the run stalls or dies, '
                 f'lower "Budget points" or "k_max" -- the table has to '
                 f'fit in RAM.')
        try:
            import shutil as _sh
            free = _sh.disk_usage(work).free / 1e9
            emit(f'  ramdisk free at {work}: {free:.1f} GB')
        except Exception:
            pass
    except Exception:
        pass

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
    try:
        fire.kgc_proc = proc
    except Exception:
        pass
    # Heartbeat on its OWN thread.
    #
    # KGC is silent for minutes while it builds the neighbour table, so
    # anything driven by its output cannot tick during exactly the
    # stretch where the user most needs to know the run is alive. A
    # separate thread updates the elapsed time regardless of whether
    # the subprocess has said anything.
    import threading as _th
    _beat_stop = _th.Event()

    def _beat():
        while not _beat_stop.wait(3.0):
            try:
                el = time.time() - t0
                # Names the phase and keeps the last measured
                # estimate visible, so a silent stretch still says
                # what is being waited on.
                ph = {'startup': 'starting up',
                      'neighbours': 'building the neighbour table',
                      'sweep': 'sweeping K'}.get(_est.phase, _est.phase)
                extra = f' - {_est.best}' if _est.best else ''
                step('kgc_cluster',
                     f'{ph} - {_fmt_secs(el)} elapsed{extra}', None)
            except Exception:
                return

    _beat_thread = _th.Thread(target=_beat, daemon=True)
    _beat_thread.start()

    lines = []
    for line in proc.stdout:
        # Cancel is cooperative: the flag is set by the endpoint, and
        # the subprocess is terminated here rather than from the
        # request thread, so cleanup happens in one place.
        if getattr(fire, 'kgc_cancel', False):
            emit('  Cancel requested -- terminating KGC')
            try:
                proc.terminate()
            except Exception:
                pass
            break
        line = line.rstrip('\n')
        if not line:
            continue
        lines.append(line)
        emit('  ' + line[:300])

        # Every line goes through the estimator, which knows the shape
        # of the work and can therefore say what is happening and how
        # long is left, rather than "elapsed --, estimating".
        upd = _est.feed(line)
        if upd:
            step(upd[0], upd[1][:200], upd[2])
            continue
    rc = proc.wait()
    _beat_stop.set()
    try:
        fire.kgc_proc = None
    except Exception:
        pass
    dt = time.time() - t0

    if getattr(fire, 'kgc_cancel', False):
        with state.lock:
            fire.kgc_cancel = False
            fire.status = FireStatus.READY
            fire.progress = {}
        emit(f'  KGC cancelled after {dt:.0f}s')
        return {'ok': False, 'cancelled': True}

    if rc != 0:
        tail = '\n'.join(lines[-15:])
        if rc < 0:
            # Negative return code means a signal. -9 is the OOM killer
            # in almost every case, and saying so saves an hour of
            # looking for a bug in the clustering.
            raise RuntimeError(
                f'KGC was killed by signal {-rc} after {dt:.0f}s '
                f'(signal 9 is almost always the out-of-memory killer). '
                f'Lower "Budget points" or "k_max" so the neighbour '
                f'table fits in RAM.\n{tail}')
        raise RuntimeError(
            f'KGC exited with code {rc} after {dt:.0f}s.\n{tail}')

    selected = out_prefix + '_selected.bin'
    if not os.path.isfile(selected):
        raise RuntimeError(
            f'KGC finished but wrote no {os.path.basename(selected)}. '
            f'Last output:\n' + '\n'.join(lines[-10:]))
    # Belt and braces after the pre-clean: a product older than this
    # run cannot be this run's.
    try:
        if os.path.getmtime(selected) < t0 - 1.0:
            raise RuntimeError(
                f'{os.path.basename(selected)} predates this run '
                f'(written {time.time() - os.path.getmtime(selected):.0f}s '
                f'ago); refusing to present a stale result')
    except OSError:
        pass
    emit(f'  KGC finished in {dt:.0f}s')

    # ---- from here on, identical to the CLI pipeline --------------
    # Hint and input sizes, so a result that looks wrong can be
    # attributed to its inputs rather than guessed at.
    try:
        from osgeo import gdal as _g
        import numpy as _np
        _hd = _g.Open(hint, _g.GA_ReadOnly)
        if _hd is not None:
            _ha = _hd.GetRasterBand(1).ReadAsArray()
            _hn = int(_np.count_nonzero(_np.nan_to_num(_ha) > 0))
            _gt = _hd.GetGeoTransform()
            _px = abs(_gt[1] * _gt[5]) / 10000.0
            emit(f'  hint: {_hn:,} px ({_hn * _px:.2f} ha) from '
                 f'{os.path.basename(hint)}')
            if _hn < 25:
                emit('  WARNING: the hint is nearly empty, so the '
                     'selected class will be tiny. Check the hint '
                     'mode and "Restrict hint to BCWS perimeter".')
            _hd = None
    except Exception:
        pass

    step('kgc_classify', 'extracting the class mask', 0.80)
    # Each run gets its OWN classified raster, numbered like the sweep's.
    #
    # Writing every run to <fire>_classified.bin meant a re-run
    # overwrote the file the previous entry pointed at: the old
    # thumbnail stayed in the gallery but its Accept button would have
    # promoted the NEW mask. Per-run files make the gallery a real
    # history that can be compared and accepted from.
    with state.lock:
        _prev = list(getattr(fire, 'serial_results', None) or [])
    _run_id = 1 + max([int(r.get('run_id') or 0) for r in _prev] or [0])
    clf = os.path.join(fire.cache_dir,
                       f'{fire.fire_numbe}_serial_{_run_id}_classified.bin')
    _extract_class_band(selected, base_stack, clf, log=emit)
    emit(f'  run #{_run_id} -> {os.path.basename(clf)}')

    # Keep the full KGC product with the fire: it carries the
    # diagnostics (log-likelihood ratio, neighbouring K levels) that
    # explain the result, and discarding them would make a run
    # unreviewable.
    for suffix in ('_selected.bin', '_selected.hdr', '_klevels.csv',
                   '_params.txt'):
        src_f = out_prefix + suffix
        if os.path.isfile(src_f):
            try:
                shutil.copy2(src_f,
                             os.path.join(fire.cache_dir,
                                          f'{fire.fire_numbe}_kgc'
                                          f'{suffix}'))
            except OSError as exc:
                emit(f'  (could not keep {suffix}: {exc})')

    # Render "before brushing" HERE, from the classifier's own output,
    # while it is still the file on disk.
    #
    # Deriving it later from the _raw.bin sibling made the layer depend
    # on the brush step having run, on the copy having succeeded, and
    # on a filename convention that differs between the KGC and CLI
    # paths -- three ways to end up with no layer, which is what kept
    # happening. Rendering it at the one moment the pre-brush mask is
    # unambiguously present removes all three.
    try:
        import numpy as _np
        from osgeo import gdal as _g
        _d = _g.Open(clf, _g.GA_ReadOnly)
        _n = (int(_np.count_nonzero(
            _np.nan_to_num(_d.GetRasterBand(1).ReadAsArray()) > 0))
            if _d is not None else -1)
        _d = None
        # Name the BASE too. The overlay is drawn onto previews/post.png,
        # so if that file is currently the other source's post image --
        # or a stale one -- the layer shows imagery that does not match
        # the pane, which is indistinguishable from "the layer is
        # broken". Recording both inputs makes the difference legible.
        _postp = os.path.join(fire.cache_dir, 'previews', 'post.png')
        try:
            _psz = os.path.getsize(_postp)
            _page = time.time() - os.path.getmtime(_postp)
        except OSError:
            _psz, _page = -1, -1
        emit(f'  before brushing: {_n:,} px; base=previews/post.png '
             f'({_psz:,} B, written {_page:.0f}s ago); '
             f'source={want_src.upper()}')
        _overlay_mask_on_post(fire, clf, 'result_prebrush',
                              (0.9, 0.1, 0.0))
        from .erase import _verify_overlay_differs
        _ok_pb = _verify_overlay_differs(fire, 'result_prebrush',
                                         log=emit)
        _pb = os.path.join(fire.cache_dir, 'previews',
                           'result_prebrush.png')
        if _ok_pb and os.path.isfile(_pb):
            with state.lock:
                if 'result_prebrush' not in fire.available_views:
                    fire.available_views.append('result_prebrush')
            emit('  pre-brush layer written')
        else:
            emit('  WARNING: the pre-brush overlay was not written '
                 '(empty mask?)')
    except Exception as exc:
        emit(f'  Pre-brush layer failed: {type(exc).__name__}: {exc}')

    step('kgc_brush', 'brushing the KGC result', 0.88)
    clf = _brush_classified(fire, clf, params, log=emit)

    # Clip AFTER brushing: brushing cleans up speckle, and clipping
    # then removes whatever survived outside the perimeter. Doing it
    # first would let the brush grow the mask back across the boundary.
    if bool(getattr(fire, 'clip_to_bcws', False)):
        step('kgc_brush', 'clipping to the BCWS perimeter', 0.92)
        try:
            from .prepare import clip_mask_to_bcws
            clip_mask_to_bcws(fire, clf, log=emit)
        except Exception as exc:
            emit(f'  Clip to BCWS failed: {exc}')

    # The brush rewrites the mask as raw floats plus a copied header,
    # so confirm the map info survived before anything is rendered or
    # scored against it.
    try:
        from .erase import ensure_geo
        ensure_geo(clf, fire.crop_bin, log=emit)
    except Exception as exc:
        emit(f'  Geo check failed: {exc}')

    step('kgc_figure', 'scoring and rendering', 0.94)
    agr = _compute_agreement(fire)
    ml_area = _compute_ml_area(fire, clf)

    _overlay_mask_on_post(fire, clf, f'serial_{_run_id}',
                          (0.9, 0.1, 0.0))
    prev_dir = os.path.join(fire.cache_dir, 'previews')
    s1 = os.path.join(prev_dir, f'serial_{_run_id}.png')
    res = os.path.join(prev_dir, 'result.png')
    if os.path.isfile(s1):
        shutil.copy2(s1, res)
        try:
            from .mapping import copy_preview_geo
            copy_preview_geo(fire.cache_dir, f'serial_{_run_id}',
                             'result')
        except Exception:
            pass
        if 'result' not in fire.available_views:
            fire.available_views.append('result')

    # One entry, shaped exactly like a serial run's, so the gallery and
    # its Accept button need no special case for KGC.
    # The canonical <fire>_classified.bin tracks the LATEST run, because
    # the eraser, rebrush and Download all read that path. Accepting an
    # older run copies its own file over this one.
    try:
        canon = os.path.join(fire.cache_dir,
                             f'{fire.fire_numbe}_classified.bin')
        shutil.copy2(clf, canon)
        _h = os.path.splitext(clf)[0] + '.hdr'
        if os.path.isfile(_h):
            shutil.copy2(_h, os.path.splitext(canon)[0] + '.hdr')
        _r = os.path.splitext(clf)[0] + '_raw.bin'
        if os.path.isfile(_r):
            shutil.copy2(_r, os.path.splitext(canon)[0] + '_raw.bin')
            _rh = os.path.splitext(_r)[0] + '.hdr'
            if os.path.isfile(_rh):
                shutil.copy2(_rh,
                             os.path.splitext(canon)[0] + '_raw.hdr')
    except OSError as exc:
        emit(f'  (could not update the canonical mask: {exc})')

    entry = {
        'run_id': _run_id,
        'setting_idx': 0,
        'run_idx': 0,
        'setting_label': f'KGC {_which.upper()} ({want_src.upper()})',
        'build': _which,
        'method': 'kgc',
        'params': dict(params or {}),
        'agreement_pct': agr,
        'ml_area_ha': ml_area,
        'comparison': None,
        'classified': clf,
    }
    with state.lock:
        # Keep the parameters that produced this result, so re-opening
        # the fire shows what was actually run rather than defaults.
        fire.kgc_params = dict(params or {})
        # APPEND: every run stays in the gallery so runs can be compared
        # and any of them accepted, including after one has been
        # accepted already.
        fire.serial_results = _prev + [entry]
        fire.agreement_pct = agr
        # FireInfo's field is ml_area_ha; writing ml_size_ha created a
        # stray attribute, left the header at '--', and made
        # hasMlResult() false -- which is why the pane was labelled
        # "Post-fire (no ML result)" while showing the red mask.
        fire.ml_area_ha = ml_area
        fire.status = FireStatus.MAPPED
    emit(f'  KGC result: agreement {agr}%, ML area {ml_area} ha')
    # One unambiguous line at the end. Which build ran is otherwise
    # inferred from tags that only appear on GPU-specific lines, and the
    # K sweep -- the most visible part of the log -- runs on the CPU in
    # both builds, so its output looks the same either way.
    emit('  ' + '=' * 62)
    emit(f'  KGC COMPLETE using the {_which.upper()} build '
         f'({os.path.basename(exe)})')
    if _which == 'gpu':
        emit('    neighbour table: GPU (CUDA)   |   K sweep: CPU '
             '(threads)')
    else:
        emit('    neighbour table: CPU (threads) |   K sweep: CPU '
             '(threads)')
    emit(f'    source={want_src.upper()}  bands={_dim}  '
         f'points<={_npts:,}  elapsed={dt:.0f}s'
         + (f'  [{_note}]' if _note else ''))
    emit('  ' + '=' * 62)
    step('kgc_figure', 'done', 1.0)
    return {'ok': True, 'agreement_pct': agr, 'ml_area_ha': ml_area,
            'classified': clf, 'seconds': dt}


def _brush_classified(fire, clf_path: str, params: dict, log=None):
    """Brush a classified mask in place, exactly as rebrush does.

    Uses the same helpers and the same convention as
    ``handlers/rebrush.py``: the pre-brush raster is preserved as
    ``_raw.bin`` and the brushed mask REPLACES the canonical
    ``_classified.bin``. Keeping the canonical path stable is what
    lets agreement, ML area, the overlay and the export all read the
    brushed result without knowing which method produced it.

    Order matters: the raw copy is taken BEFORE the canonical file is
    overwritten, or the "raw" backup would hold the brushed mask.

    Returns the canonical path (brushed if it worked), never None, so
    callers always have a mask to use.
    """
    try:
        from .brush import (_run_class_brush_only,
                            _write_envi_mask_like)
    except Exception as exc:
        if log:
            log(f'  Brush unavailable ({exc}); keeping the raw KGC mask')
        return clf_path

    try:
        size = int((params or {}).get('brush_size', 15) or 15)
        thresh = int((params or {}).get('point_threshold', 10) or 10)
        all_seg = bool((params or {}).get('brush_all_segments', False))
    except (TypeError, ValueError):
        size, thresh, all_seg = 15, 10, False

    if size <= 0:
        if log:
            log('  Brush skipped (brush_size <= 0)')
        return clf_path

    # Snapshot the classifier's own output BEFORE brushing is
    # attempted.
    #
    # Taking the copy only on success meant a failed brush (a stale
    # class_brush binary rejecting its arguments does exactly that)
    # left no pre-brush mask at all -- so the "before brushing" layer
    # had nothing to show. The snapshot is what that layer IS, and it
    # exists regardless of whether brushing then works.
    raw_backup = os.path.splitext(clf_path)[0] + '_raw.bin'
    if clf_path != raw_backup:
        try:
            shutil.copy2(clf_path, raw_backup)
            _h = os.path.splitext(clf_path)[0] + '.hdr'
            if not os.path.isfile(_h):
                _h = clf_path + '.hdr'
            if os.path.isfile(_h):
                shutil.copy2(_h, os.path.splitext(raw_backup)[0] + '.hdr')
            if log:
                log(f'  Kept the pre-brush classification as '
                    f'{os.path.basename(raw_backup)}')
        except OSError as exc:
            if log:
                log(f'  Could not keep a pre-brush copy: {exc}')

    try:
        if log:
            log(f'  Brushing: size={size} threshold={thresh} '
                f'all_segments={all_seg}')
        brushed, cancelled = _run_class_brush_only(
            clf_path, size, thresh, all_seg,
            fire_numbe=fire.fire_numbe)
        if cancelled:
            if log:
                log('  Brush cancelled; keeping the raw KGC mask')
            return clf_path
        if brushed is None:
            if log:
                log('  Brush produced nothing (is class_brush.exe '
                    'built?); keeping the raw KGC mask')
            return clf_path

        # The pre-brush copy was already taken above, before brushing
        # was attempted; copying again here would overwrite it with the
        # brushed result the moment this runs twice.
        _write_envi_mask_like(brushed, clf_path, clf_path)
        if log:
            import numpy as _np
            log(f'  Brushed: {int(_np.count_nonzero(brushed)):,} '
                f'pixel(s) retained')
        return clf_path
    except Exception as exc:
        if log:
            log(f'  Brush failed ({type(exc).__name__}: {exc}); '
                f'keeping the raw KGC mask')
        return clf_path
