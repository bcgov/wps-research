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
    exe = ensure_kgc_binary(log=emit)

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

    cmd = build_kgc_cmd(exe, image, hint, out_prefix, params)
    emit('  ' + ' '.join(cmd))
    step('kgc_cluster', 'clustering (the long step)', 0.15)

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
    try:
        fire.kgc_proc = proc
    except Exception:
        pass
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

        # Turn KGC's own output into progress with an ETA.
        #
        # The sweep prints "  N/M levels, best K=..." every 50 levels,
        # which is the only quantitative progress the tool emits and
        # covers the long stage. Everything else is a stage marker.
        # Without this the bar sat still for the entire run and there
        # was no way to tell work from a hang.
        m = re.match(r'\s*(\d+)/(\d+)\s+levels', line)
        if m:
            done_l, total_l = int(m.group(1)), int(m.group(2))
            if total_l > 0:
                f = max(0.0, min(1.0, done_l / float(total_l)))
                el = time.time() - t0
                eta = (el / f - el) if f > 0.02 and el > 2 else None
                # 0.15..0.78 of the whole run is the sweep; the stages
                # either side are quick by comparison.
                overall = 0.15 + 0.63 * f
                detail = (f'sweep {done_l}/{total_l} levels'
                          + (f' ({m.string.split("best K=")[1].strip()})'
                             if 'best K=' in line else ''))
                if eta is not None:
                    detail += f' - about {_fmt_secs(eta)} left'
                step('kgc_cluster', detail[:160], overall)
                continue
        low = line.lower()
        if low.startswith('[dedup]'):
            step('kgc_cluster', line.strip()[:160], 0.10)
        elif low.startswith('[knn]'):
            step('kgc_cluster', line.strip()[:160], 0.13)
        elif low.startswith('[hint]'):
            step('kgc_cluster', line.strip()[:160], 0.16)
        elif low.startswith('[out]'):
            step('kgc_classify', 'writing the KGC product', 0.79)
    rc = proc.wait()
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
    clf = os.path.join(fire.cache_dir, f'{fire.fire_numbe}_classified.bin')
    _extract_class_band(selected, base_stack, clf, log=emit)

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

    _overlay_mask_on_post(fire, clf, 'serial_1', (0.9, 0.1, 0.0))
    prev_dir = os.path.join(fire.cache_dir, 'previews')
    s1 = os.path.join(prev_dir, 'serial_1.png')
    res = os.path.join(prev_dir, 'result.png')
    if os.path.isfile(s1):
        shutil.copy2(s1, res)
        try:
            from .mapping import copy_preview_geo
            copy_preview_geo(fire.cache_dir, 'serial_1', 'result')
        except Exception:
            pass
        if 'result' not in fire.available_views:
            fire.available_views.append('result')

    # One entry, shaped exactly like a serial run's, so the gallery and
    # its Accept button need no special case for KGC.
    entry = {
        'run_id': 1,
        'setting_idx': 0,
        'run_idx': 0,
        'setting_label': f'KGC ({want_src.upper()})',
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
        fire.serial_results = [entry]
        fire.agreement_pct = agr
        # FireInfo's field is ml_area_ha; writing ml_size_ha created a
        # stray attribute, left the header at '--', and made
        # hasMlResult() false -- which is why the pane was labelled
        # "Post-fire (no ML result)" while showing the red mask.
        fire.ml_area_ha = ml_area
        fire.status = FireStatus.MAPPED
    emit(f'  KGC result: agreement {agr}%, ML area {ml_area} ha')
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

        raw_backup = os.path.splitext(clf_path)[0] + '_raw.bin'
        if clf_path != raw_backup:
            try:
                shutil.copy2(clf_path, raw_backup)
                hdr_src = os.path.splitext(clf_path)[0] + '.hdr'
                if not os.path.isfile(hdr_src):
                    hdr_src = clf_path + '.hdr'
                if os.path.isfile(hdr_src):
                    shutil.copy2(
                        hdr_src,
                        os.path.splitext(raw_backup)[0] + '.hdr')
            except OSError:
                pass
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
