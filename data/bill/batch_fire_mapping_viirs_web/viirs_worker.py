"""VIIRS prepare worker.

A user submits a fire-creation request → this module enqueues a
background thread that runs the per-fire VIIRS pipeline:

  1. accumulate per-granule .shp from the year-wide shared dir → cumulative .shp
  2. rasterize cumulative .shp onto the year's reference raster (full extent)
  3. derive a tight crop from the rasterized fire pixels (+ padding)
  4. crop the year's raster to that bbox; re-rasterize VIIRS onto crop
  5. generate previews; mark READY

Download + shapify are no longer per-fire stages — they happen once at
server boot in :mod:`year_viirs`. The worker only consumes the pre-built
shared shapefile dir, so per-fire prepare is seconds rather than minutes.

Cancellation is cooperative: the worker checks ``fire.cancel_event``
between stages. Each fire's cache_dir is wiped on cancel.

Module-level dispatch queue caps concurrency at
``state.viirs_concurrent_jobs`` (default 1)."""

import os
import re
import shutil
import signal
import sys
import threading
import time
from queue import Empty, Queue
from typing import Optional

import numpy as np
from osgeo import gdal

# Matches the "[k/N]" prefix accumulate.py emits per date-group iteration.
_ACC_PROGRESS_RE = re.compile(r'^\[(\d+)/(\d+)\]')

gdal.UseExceptions()

from .state import AppState, FireInfo, FireStatus

# Bound by ``init`` from app.init_app
state: AppState = None
_save_fire_state = None
_push_notification = None


def init(app_state, save_fire_state_cb, push_notification_cb):
    global state, _save_fire_state, _push_notification
    state = app_state
    _save_fire_state = save_fire_state_cb
    _push_notification = push_notification_cb
    _ensure_dispatcher()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class WorkerError(RuntimeError):
    """Raised by stage code with a user-facing error message."""


class WorkerCancelled(BaseException):
    """Internal control-flow signal for cooperative cancel."""


# ---------------------------------------------------------------------------
# Stage progress tracking
# ---------------------------------------------------------------------------

# Order matters — UI uses indices. We dropped the standalone
# 'rasterizing' stage when we removed the full-extent rasterize: bounds
# are now derived directly from the cumulative shapefile, so 'cropping'
# absorbs the only rasterize that's still needed (a fast one onto the
# small cropped frame).
STAGES = (
    'accumulating',
    'cropping',
)


def _set_progress(fire: FireInfo, stage: str, *, detail: str = '',
                  fraction: Optional[float] = None) -> None:
    """Write a progress snapshot under state.lock."""
    try:
        idx = STAGES.index(stage) + 1
    except ValueError:
        idx = 0
    snapshot = {
        'stage': stage,
        'stage_idx': idx,
        'total_stages': len(STAGES),
        'detail': detail,
        'updated_at': time.time(),
    }
    if fraction is not None:
        try:
            snapshot['fraction'] = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            pass
    with state.lock:
        fire.progress = snapshot


# ---------------------------------------------------------------------------
# Tight bounds from rasterized VIIRS bin
# ---------------------------------------------------------------------------

def _tight_bounds_from_viirs_bin(viirs_bin: str, padding: float):
    """Return (xmin, ymin, xmax, ymax) in the raster CRS, tightened to
    the bbox of nonzero pixels in viirs_bin and expanded by
    ``padding * max_dim_in_pixels``.

    Retained for backwards compatibility / tests. New code should use
    :func:`_tight_bounds_from_shapefile`, which avoids the full-extent
    rasterize that produces *viirs_bin* in the first place."""
    ds = gdal.Open(viirs_bin, gdal.GA_ReadOnly)
    if ds is None:
        raise WorkerError(f'Cannot open VIIRS bin: {viirs_bin}')
    try:
        arr = ds.GetRasterBand(1).ReadAsArray()
        gt = ds.GetGeoTransform()
        W, H = ds.RasterXSize, ds.RasterYSize
    finally:
        ds = None

    nz = np.nonzero(arr)
    if nz[0].size == 0:
        raise WorkerError('VIIRS hint has no fire pixels.')
    py_lo, py_hi = int(nz[0].min()), int(nz[0].max())
    px_lo, px_hi = int(nz[1].min()), int(nz[1].max())

    fire_max_dim = max(px_hi - px_lo, py_hi - py_lo)
    p = max(1, int(round(padding * fire_max_dim))) if padding > 0 else 0
    px_lo = max(0, px_lo - p)
    px_hi = min(W - 1, px_hi + p)
    py_lo = max(0, py_lo - p)
    py_hi = min(H - 1, py_hi + p)

    xmin = gt[0] + px_lo * gt[1]
    xmax = gt[0] + (px_hi + 1) * gt[1]
    ymax = gt[3] + py_lo * gt[5]   # gt[5] is negative
    ymin = gt[3] + (py_hi + 1) * gt[5]
    return xmin, ymin, xmax, ymax


# Default in viirs.utils.rasterize.rasterize_shapefile — the buffer that
# would otherwise have been baked into the full-extent rasterize, applied
# here in CRS units so we can skip the rasterize entirely.
_RASTERIZE_BUFFER_M = 375.0


def _tight_bounds_from_shapefile(acc_shp: str, ref_raster: str,
                                  padding: float,
                                  buffer_m: float = _RASTERIZE_BUFFER_M):
    """Derive the same tight crop bounds as ``_tight_bounds_from_viirs_bin``
    directly from the cumulative VIIRS shapefile, skipping the full-extent
    rasterize.

    Algorithm: read ``gdf.total_bounds`` (the geometry envelope in the
    shapefile's CRS — which is already the reference raster's CRS, set by
    ``accumulate``), expand by ``buffer_m`` to match the buffer that
    ``rasterize_shapefile`` would have applied, snap to the reference
    raster's pixel grid, then expand by ``padding * fire_max_dim`` pixels
    just like the rasterized version. Result is byte-equivalent modulo a
    sub-pixel rounding difference."""
    import geopandas as gpd

    gdf = gpd.read_file(acc_shp)
    if gdf.empty:
        raise WorkerError(
            'No VIIRS fire features in cumulative shapefile.')

    ds = gdal.Open(ref_raster, gdal.GA_ReadOnly)
    if ds is None:
        raise WorkerError(f'Cannot open reference raster: {ref_raster}')
    try:
        gt = ds.GetGeoTransform()
        W, H = ds.RasterXSize, ds.RasterYSize
    finally:
        ds = None

    sx_min, sy_min, sx_max, sy_max = (float(v) for v in gdf.total_bounds)
    sx_min -= buffer_m
    sy_min -= buffer_m
    sx_max += buffer_m
    sy_max += buffer_m

    # Convert CRS bounds → pixel indices using the inverse geotransform.
    # gt[5] is negative for north-up rasters, so sy_max maps to the
    # smaller pixel row (top) and sy_min to the larger (bottom).
    px_lo = int(np.floor((sx_min - gt[0]) / gt[1]))
    px_hi = int(np.ceil((sx_max - gt[0]) / gt[1])) - 1
    py_lo = int(np.floor((sy_max - gt[3]) / gt[5]))
    py_hi = int(np.ceil((sy_min - gt[3]) / gt[5])) - 1

    px_lo = max(0, px_lo)
    px_hi = min(W - 1, px_hi)
    py_lo = max(0, py_lo)
    py_hi = min(H - 1, py_hi)

    if px_hi < px_lo or py_hi < py_lo:
        raise WorkerError(
            'VIIRS shapefile lies outside the reference raster extent.')

    fire_max_dim = max(px_hi - px_lo, py_hi - py_lo)
    p = max(1, int(round(padding * fire_max_dim))) if padding > 0 else 0
    px_lo = max(0, px_lo - p)
    px_hi = min(W - 1, px_hi + p)
    py_lo = max(0, py_lo - p)
    py_hi = min(H - 1, py_hi + p)

    xmin = gt[0] + px_lo * gt[1]
    xmax = gt[0] + (px_hi + 1) * gt[1]
    ymax = gt[3] + py_lo * gt[5]
    ymin = gt[3] + (py_hi + 1) * gt[5]
    return xmin, ymin, xmax, ymax


def _verify_viirs_bin_nonzero(path: str) -> None:
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise WorkerError(f'Cannot open VIIRS bin: {path}')
    try:
        arr = ds.GetRasterBand(1).ReadAsArray()
    finally:
        ds = None
    if arr is None or np.nansum(arr) == 0:
        raise WorkerError(
            'No VIIRS fire pixels in bbox during the chosen date range.')


def _read_dims(path: str):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    try:
        return ds.RasterXSize, ds.RasterYSize
    finally:
        ds = None


def _compute_viirs_area_ha(viirs_bin: str) -> float:
    """Approximate fire-pixel area (hectares) from the rasterized hint."""
    ds = gdal.Open(viirs_bin, gdal.GA_ReadOnly)
    if ds is None:
        return 0.0
    try:
        arr = ds.GetRasterBand(1).ReadAsArray()
        gt = ds.GetGeoTransform()
    finally:
        ds = None
    if arr is None:
        return 0.0
    px_area_m2 = abs(gt[1] * gt[5])
    n_fire_px = int(np.count_nonzero(arr))
    return round(n_fire_px * px_area_m2 / 10000.0, 2)


# ---------------------------------------------------------------------------
# Year-wide VIIRS shp dir lookup
# ---------------------------------------------------------------------------

def _year_shp_dir_for(fire: FireInfo) -> str:
    """Return the per-year shared VIIRS shp dir, or raise WorkerError."""
    by_year = getattr(state, 'viirs_shp_dirs_by_year', None) or {}
    shp_dir = by_year.get(fire.fire_year)
    if not shp_dir:
        from . import year_viirs
        shp_dir = year_viirs.year_shp_dir(state, fire.fire_year)
    if not shp_dir or not os.path.isdir(shp_dir):
        raise WorkerError(
            f'Year-wide VIIRS data missing for {fire.fire_year} '
            f'(expected at {shp_dir!r}). Restart the server to run '
            f'the bootstrap download.')
    return shp_dir


def accumulate_for_fire(fire: FireInfo, cache_dir: str,
                        ref_raster: str) -> str:
    """Run ``viirs.utils.accumulate`` for *fire* against the year-wide shp
    dir, writing the cumulative shapefile into cache_dir. Returns the
    cumulative .shp path. Raises WorkerError on no-features.

    Idempotent: when a cumulative ``VIIRS_VNP14IMG_*.shp`` is already
    present at the top of *cache_dir* (e.g. seeded by the create handler
    after the user clicked Preview Hint), this returns it directly and
    skips the accumulate work entirely. The seed is trusted to match
    fire.bbox_native + fire.viirs_start_date/end — the create handler
    validates that before copying.

    Wires accumulate's progress_cb into ``_set_progress`` so the UI gets
    sub-stage updates ("date 5/12: ...") while the per-day shapefile
    loads run, and so cancel is observed inside the inner loop instead
    of waiting for accumulate to return."""
    import glob
    seeded = sorted(glob.glob(
        os.path.join(cache_dir, 'VIIRS_VNP14IMG_*.shp')))
    if seeded:
        return seeded[-1]

    from viirs.utils.accumulate import accumulate
    shp_dir = _year_shp_dir_for(fire)

    start_compact = fire.viirs_start_date.replace('-', '')
    end_compact = fire.viirs_end_date.replace('-', '')

    cancel_event = fire.cancel_event

    def _accumulate_cb(msg: str) -> None:
        # Cooperative cancel: accumulate emits one log line per date
        # group + setup lines. Raising here unwinds out of accumulate.
        if cancel_event is not None and cancel_event.is_set():
            raise WorkerCancelled()
        # Pluck "[k/N]" out of accumulate's log lines so the UI shows
        # progress within the accumulating stage. Setup lines (no
        # bracket prefix) become detail text but don't carry a fraction.
        m = _ACC_PROGRESS_RE.match(msg.strip())
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            frac = (k / n) if n > 0 else None
            _set_progress(fire, 'accumulating',
                          detail=f'date {k}/{n}', fraction=frac)
        else:
            # Setup / summary lines — pass the trimmed message as detail
            # without a fraction so the bar holds its position.
            stripped = msg.strip()
            if stripped.startswith('[INFO]'):
                stripped = stripped[len('[INFO]'):].strip()
            if stripped:
                _set_progress(fire, 'accumulating', detail=stripped[:100])

    try:
        acc_paths = accumulate(
            shp_dir=shp_dir,
            start_str=start_compact,
            end_str=end_compact,
            reference_raster=ref_raster,
            output_dir=cache_dir,
            final_only=True,
            bbox=fire.bbox_native,
            progress_cb=_accumulate_cb,
        )
    except WorkerCancelled:
        raise
    except Exception as exc:
        raise WorkerError(f'accumulate failed: {exc}')
    if not acc_paths:
        raise WorkerError(
            'No VIIRS fire pixels in bbox / date range. '
            'Try a larger bbox or a wider date window.')
    return acc_paths[-1]


# ---------------------------------------------------------------------------
# Worker thread body
# ---------------------------------------------------------------------------

def _viirs_worker(fire: FireInfo) -> None:
    """Run the full VIIRS prepare pipeline for *fire*. Sets fire.status
    on completion / failure / cancel."""
    cache_dir = os.path.join(state.output_root, '.web_cache',
                             fire.fire_numbe)
    os.makedirs(cache_dir, exist_ok=True)
    fire.cache_dir = cache_dir
    fire.cancel_event = fire.cancel_event or threading.Event()

    ref_raster = state.rasters_by_year.get(fire.fire_year) \
        or state.raster_path

    try:
        # ---- Stage 1: accumulate from year-wide shp dir ----
        _set_progress(fire, 'accumulating',
                      detail='aggregating shapefiles')
        acc_shp = accumulate_for_fire(fire, cache_dir, ref_raster)
        if fire.cancel_event.is_set():
            raise WorkerCancelled()

        # ---- Stage 2: tight crop derived from shapefile, then crop +
        # rasterize on the small frame + previews. The full-extent
        # rasterize that used to live here is gone — its only purpose was
        # to derive tight bounds, which we now read straight from
        # gdf.total_bounds (orders of magnitude faster on year-wide tiles).
        _set_progress(fire, 'cropping', detail='deriving tight bounds',
                      fraction=0.0)
        xmin, ymin, xmax, ymax = _tight_bounds_from_shapefile(
            acc_shp, ref_raster, padding=state.padding)
        if fire.cancel_event.is_set():
            raise WorkerCancelled()

        _set_progress(fire, 'cropping', detail='cropping reference raster',
                      fraction=0.25)
        from batch_fire_mapping.run_fire_mapping import crop_raster
        crop_bin = os.path.join(cache_dir, f'{fire.fire_numbe}_crop.bin')
        if not crop_raster(ref_raster, crop_bin, xmin, ymin, xmax, ymax):
            raise WorkerError('GDAL crop failed.')
        if fire.cancel_event.is_set():
            raise WorkerCancelled()

        # Rasterize VIIRS onto the cropped extent so the hint aligns
        # with crop_bin pixel-for-pixel. Cheap because the crop frame is
        # small.
        _set_progress(fire, 'cropping',
                      detail='rasterizing onto crop', fraction=0.5)
        from viirs.utils.rasterize import rasterize_shapefile
        crop_rast_dir = os.path.join(cache_dir, '_viirs_crop')
        os.makedirs(crop_rast_dir, exist_ok=True)
        viirs_cropped = rasterize_shapefile(
            shp_path=acc_shp,
            ref_image=crop_bin,
            output_dir=crop_rast_dir,
            buffer_m=_RASTERIZE_BUFFER_M,
        )
        if not viirs_cropped or not os.path.isfile(viirs_cropped):
            raise WorkerError(
                'Rasterize onto crop produced no output.')
        _verify_viirs_bin_nonzero(viirs_cropped)
        if fire.cancel_event.is_set():
            raise WorkerCancelled()

        # Generate previews from the crop.
        _set_progress(fire, 'cropping', detail='generating previews',
                      fraction=0.75)
        from .preview import generate_all_previews
        views = generate_all_previews(crop_bin, cache_dir, fire.fire_numbe)

        crop_w, crop_h = _read_dims(crop_bin)
        sample_size = max(state.min_samples, min(
            state.max_samples,
            int(round(crop_w * crop_h * state.sample_rate))))

        # Set fire.cache_dir so _overlay_mask_on_post can resolve it.
        fire.cache_dir = cache_dir

        # Generate hint overlay so the fire-mapping page already has a
        # green mask over the post preview the moment status flips to
        # READY (no classification needed).
        try:
            from .mapping import _overlay_mask_on_post
            _overlay_mask_on_post(
                fire, viirs_cropped, 'hint', (0.0, 0.8, 0.2))
        except Exception as exc:
            sys.stderr.write(
                f'[viirs_worker] hint overlay generation failed: '
                f'{exc}\n')

        with state.lock:
            fire.crop_bin = crop_bin
            fire.viirs_bin = viirs_cropped
            fire.hint_bin = viirs_cropped
            fire.crop_w = crop_w
            fire.crop_h = crop_h
            fire.padding_used = state.padding
            fire.sample_size = sample_size
            fire.acc_start = fire.viirs_start_date
            fire.acc_end = fire.viirs_end_date
            fire.perimeter_type = 'viirs'
            fire.available_views = views
            if 'hint' not in fire.available_views \
                    and os.path.isfile(os.path.join(
                        cache_dir, 'previews', 'hint.png')):
                fire.available_views.append('hint')
            fire.fire_size_ha = _compute_viirs_area_ha(viirs_cropped)
            fire.status = FireStatus.READY
            fire.is_new = True
            fire.progress = {}
            fire.error_msg = ''

        if _save_fire_state is not None:
            _save_fire_state()
        if _push_notification is not None:
            try:
                _push_notification(
                    None, 'success',
                    'Fire prepared',
                    f'{fire.fire_numbe} is ready to map.',
                    fire=fire.fire_numbe)
            except Exception:
                pass

    except WorkerCancelled:
        with state.lock:
            fire.status = FireStatus.PENDING
            fire.progress = {}
        shutil.rmtree(cache_dir, ignore_errors=True)
        # Cancel handler removes the FireInfo from state.fires.

    except WorkerError as exc:
        with state.lock:
            fire.status = FireStatus.ERROR
            fire.error_msg = str(exc)
            fire.progress = {}
        if _save_fire_state is not None:
            _save_fire_state()
        if _push_notification is not None:
            try:
                _push_notification(
                    None, 'error', 'Prepare failed',
                    f'{fire.fire_numbe}: {exc}',
                    fire=fire.fire_numbe)
            except Exception:
                pass

    except Exception as exc:
        sys.stderr.write(
            f'[viirs_worker] unexpected error in {fire.fire_numbe}: '
            f'{exc!r}\n')
        with state.lock:
            fire.status = FireStatus.ERROR
            fire.error_msg = f'unexpected error: {exc}'
            fire.progress = {}
        if _save_fire_state is not None:
            _save_fire_state()

    finally:
        with state.lock:
            state.viirs_jobs.pop(fire.fire_numbe, None)


# ---------------------------------------------------------------------------
# Module-level FIFO dispatcher
# ---------------------------------------------------------------------------

_dispatch_queue: 'Queue[FireInfo]' = Queue()
_dispatcher_started = False
_dispatcher_lock = threading.Lock()


def _ensure_dispatcher():
    """Lazy-start the dispatch thread on first init()."""
    global _dispatcher_started
    with _dispatcher_lock:
        if _dispatcher_started:
            return
        n = max(1, int(getattr(state, 'viirs_concurrent_jobs', 1) or 1))
        for i in range(n):
            t = threading.Thread(
                target=_dispatch_loop, daemon=True,
                name=f'viirs-dispatch-{i}')
            t.start()
        _dispatcher_started = True


def _dispatch_loop():
    while True:
        try:
            fire = _dispatch_queue.get(timeout=60)
        except Empty:
            continue
        if fire is None:
            continue
        try:
            _viirs_worker(fire)
        except Exception as exc:
            sys.stderr.write(
                f'[viirs_worker] dispatch loop error: {exc!r}\n')
        finally:
            _dispatch_queue.task_done()


def submit_fire(fire: FireInfo) -> None:
    """Enqueue *fire* for VIIRS prepare. Should be called *after* the
    fire has been added to state.fires under state.lock with status
    PREPARING. The dispatcher will pick it up FIFO."""
    if state is None:
        raise RuntimeError('viirs_worker not initialised')
    _ensure_dispatcher()
    with state.lock:
        state.viirs_jobs[fire.fire_numbe] = None
    _dispatch_queue.put(fire)


def cancel_fire(fire: FireInfo) -> bool:
    """Signal cancel to a running prepare. Idempotent."""
    if state is None:
        return False
    if fire.cancel_event is None:
        fire.cancel_event = threading.Event()
    fire.cancel_event.set()
    with state.lock:
        proc = state.viirs_subprocs.get(fire.fire_numbe)
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return True
