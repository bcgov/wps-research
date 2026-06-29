"""Synchronous prepare + accept flow.

Both functions run from the request thread (or from the serial worker's
re-prepare path) and own the per-fire cache_dir → canonical-output-dir
hand-off. Holds no GPU lock; the caller arranges that.
"""

import datetime
import glob
import os
import shutil
import sys
import threading

import numpy as np
from osgeo import gdal

from .state import AppState, FireInfo, FireStatus
from .io_utils import _atomic_yaml_dump
from .preview import generate_all_previews, detect_band_groups, parse_envi_band_names
from .mapping import (
    _compute_ml_area, _overlay_mask_on_post, _generate_result_preview,
)
from .brush import _read_envi_mask, _render_brush_comparison_png
from .kml import _export_kml
from .persistence import _save_fire_state

gdal.UseExceptions()


# -----------------------------------------------------------------------
# "Red wins" dominant-band hint generation
# -----------------------------------------------------------------------

def generate_redwins_hint(crop_bin: str, band_indices: list[int],
                          output_path: str) -> bool:
    """Generate a binary hint mask using the "red wins" rule.

    For each pixel, the first of the three bands in *band_indices* is
    compared against the other two.  Where it exceeds both, the pixel
    is marked 1 (burned); elsewhere 0.  NaN in any input band produces
    NaN in the output.  The result is written as a single-band ENVI
    float32 raster whose geotransform and projection match *crop_bin*,
    so it plugs straight into the same hint-overlay / mapping-CLI path
    that the VIIRS rasterised mask already uses.

    *band_indices* are 1-based GDAL band numbers — typically the three
    bands of the ``post`` or ``diff1`` group from
    :func:`preview.detect_band_groups`.

    Returns True on success.
    """
    ds = gdal.Open(crop_bin, gdal.GA_ReadOnly)
    if ds is None:
        return False
    try:
        w, h = ds.RasterXSize, ds.RasterYSize
        n_bands = ds.RasterCount
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()

        channels = []
        for b_idx in band_indices[:3]:
            if b_idx < 1 or b_idx > n_bands:
                channels.append(np.full((h, w), np.nan, dtype=np.float32))
                continue
            arr = ds.GetRasterBand(b_idx).ReadAsArray().astype(np.float32)
            channels.append(arr)
    finally:
        ds = None

    if len(channels) < 3:
        return False

    red, green, blue = channels[0], channels[1], channels[2]

    # "Red wins" = the first band strictly exceeds the other two at
    # this pixel.  This is the core logic from dominant_band.py.
    mask = (red > green) & (red > blue)

    # Propagate NaN from any input band.
    any_nan = np.isnan(red) | np.isnan(green) | np.isnan(blue)

    result = np.where(any_nan, np.nan, mask.astype(np.float32))

    # Write as ENVI flat binary (BSQ, float32) with matching header.
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(output_path, w, h, 1, gdal.GDT_Float32)
    if out_ds is None:
        return False
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(result)
    out_ds.FlushCache()
    out_ds = None
    return True


def switch_hint_mode(fire: FireInfo, mode: str) -> dict:
    """Switch a fire's hint mask between viirs / redwins_post / redwins_diff.

    Regenerates ``fire.hint_bin``, the hint overlay preview PNG, and
    updates ``fire.perimeter_type`` / ``fire.hint_mode``.

    Returns a dict with 'ok' (bool) and 'error' (str, if not ok).
    """
    if mode not in ('viirs', 'redwins_post', 'redwins_diff'):
        return {'ok': False, 'error': f'Unknown hint mode: {mode}'}

    if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
        return {'ok': False, 'error': 'Fire has no crop raster.'}

    if mode == 'viirs':
        # Restore the original VIIRS hint.
        if not fire.viirs_bin or not os.path.isfile(fire.viirs_bin):
            return {'ok': False,
                    'error': 'No VIIRS hint available for this fire.'}
        fire.hint_bin = fire.viirs_bin
        fire.perimeter_type = 'viirs'
        fire.hint_mode = 'viirs'

    else:
        # Red-wins mode: pick band group.
        band_names = parse_envi_band_names(fire.crop_bin)
        if not band_names:
            ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
            if ds:
                try:
                    n = ds.RasterCount
                finally:
                    ds = None
                band_names = [f'band {i + 1}' for i in range(n)]
        groups = detect_band_groups(band_names)

        if mode == 'redwins_post':
            indices = groups.get('post', [])
            label = 'redwins_post'
        else:
            indices = groups.get('diff1', [])
            label = 'redwins_diff'

        if len(indices) < 3:
            return {'ok': False,
                    'error': f'Not enough bands for {label} '
                             f'(need 3, found {len(indices)}).'}

        out_dir = os.path.join(fire.cache_dir, '_redwins')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{label}_hint.bin')

        if not generate_redwins_hint(fire.crop_bin, indices, out_path):
            return {'ok': False,
                    'error': f'Failed to generate {label} hint mask.'}

        fire.hint_bin = out_path
        fire.perimeter_type = label
        fire.hint_mode = mode

    # Regenerate the hint overlay preview PNG.
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))
        if 'hint' not in fire.available_views:
            fire.available_views.append('hint')

    return {'ok': True}

# Bound by ``init`` from app.init_app — these live in ``app.py`` because
# they coordinate with locks/registries shared across the worker, the
# accept handler, and the cache sweeper.
state: AppState = None
_set_fire_status = None
_accept_in_progress = None
_accept_in_progress_lock = None
_accept_file_lock = None
_CSV_FIELDNAMES = None


def init(app_state, set_fire_status, accept_in_progress,
         accept_in_progress_lock, accept_file_lock, csv_fieldnames):
    global state, _set_fire_status, _accept_in_progress
    global _accept_in_progress_lock, _accept_file_lock, _CSV_FIELDNAMES
    state = app_state
    _set_fire_status = set_fire_status
    _accept_in_progress = accept_in_progress
    _accept_in_progress_lock = accept_in_progress_lock
    _accept_file_lock = accept_file_lock
    _CSV_FIELDNAMES = csv_fieldnames


def _prepare_fire_sync(fire_numbe: str, padding: float | None = None):
    """Re-prepare a fire after padding change or cache eviction.

    Initial prepare lives in ``viirs_worker._viirs_worker``. This function
    handles the re-prepare path: re-accumulate from the **year-wide**
    shared shapefile dir, re-rasterize on the year reference, derive
    tight bounds with the new padding, re-crop the raster, and re-rasterize
    onto the cropped frame.

    Re-accumulating from the shared dir (instead of a per-fire copy in
    cache_dir) is what makes the padding-change path on serial mapping
    survive — the cache_dir wipe below removes the per-fire cumulative
    shapefile, and recovering it from the shared dir is fast.
    """
    from batch_fire_mapping.run_fire_mapping import crop_raster
    from .viirs_worker import (
        _tight_bounds_from_shapefile, _verify_viirs_bin_nonzero,
        _read_dims, _compute_viirs_area_ha, accumulate_for_fire,
        _RASTERIZE_BUFFER_M, WorkerError,
        _invalidate_stale_rasterize,
    )
    from viirs.utils.rasterize import rasterize_shapefile

    fire = state.fires[fire_numbe]

    with state.lock:
        if fire.status == FireStatus.PREPARING:
            fire.error_msg = 'Cannot prepare: fire is currently preparing'
            return
        fire.status = FireStatus.PREPARING
        fire.error_msg = ""

    pad = padding if padding is not None else state.padding
    cache_dir = fire.cache_dir or os.path.join(
        state.output_root, '.web_cache', fire_numbe)
    os.makedirs(cache_dir, exist_ok=True)

    if not getattr(fire, 'bbox_native', None) \
            or not fire.viirs_start_date \
            or not fire.viirs_end_date:
        _set_fire_status(
            fire, FireStatus.ERROR,
            'Cannot re-prepare: fire has no bbox or date range on '
            'record. Re-create the fire from /new_fire.')
        return

    ref_raster = state.rasters_by_year.get(fire.fire_year) \
        or state.raster_path

    # ---- Re-accumulate from year-wide shared dir into cache_dir ----
    try:
        acc_shp = accumulate_for_fire(fire, cache_dir, ref_raster)
    except WorkerError as exc:
        _set_fire_status(fire, FireStatus.ERROR, str(exc))
        return
    except Exception as exc:
        _set_fire_status(fire, FireStatus.ERROR,
                         f'accumulate failed: {exc}')
        return

    # ---- Tight crop bounds straight from the cumulative shapefile.
    # Skips the full-extent rasterize that this used to do — for a
    # year-wide reference raster that rasterize was the dominant cost
    # of every padding change between settings, even though its output
    # was thrown away after deriving these four bounds.
    try:
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = \
            _tight_bounds_from_shapefile(acc_shp, ref_raster, padding=pad)
    except WorkerError as exc:
        _set_fire_status(fire, FireStatus.ERROR, str(exc))
        return
    except Exception as exc:
        _set_fire_status(fire, FireStatus.ERROR,
                         f'tight bounds failed: {exc}')
        return

    crop_w = max(1, int(round(
        (crop_xmax - crop_xmin) / abs(state.raster_gt[1] or 1))))
    crop_h = max(1, int(round(
        (crop_ymax - crop_ymin) / abs(state.raster_gt[5] or 1))))
    old_pad = fire.padding_used
    fire.crop_w = crop_w
    fire.crop_h = crop_h
    fire.padding_used = pad

    sample_size = int(round(crop_w * crop_h * state.sample_rate))
    sample_size = max(state.min_samples, min(state.max_samples, sample_size))
    fire.sample_size = sample_size

    # -- Crop raster --
    # We deliberately do NOT wipe cache_dir here. The previous behaviour
    # was to delete every top-level file when padding changed (to avoid
    # stale results from a different crop extent), but it also took out
    # the cumulative shapefile and full-extent VIIRS bin we just re-built
    # above, plus the {fire}_crop.{bin,hdr} we are about to overwrite.
    # We only need to drop preview PNGs (tied to the old post extent) and
    # any serial overlay PNGs that referenced the old crop frame.
    fire.cache_dir = cache_dir
    previews_dir = os.path.join(cache_dir, 'previews')
    if old_pad != 0 and old_pad != pad and os.path.isdir(previews_dir):
        shutil.rmtree(previews_dir, ignore_errors=True)

    crop_bin = os.path.join(cache_dir, f'{fire_numbe}_crop.bin')
    # crop_raster (gdal.Translate to ENVI) overwrites if the file exists,
    # but the .hdr / .aux.xml siblings can stick around. Remove them so
    # the new geotransform isn't overlaid on stale header lines.
    for ext in ('.bin', '.hdr', '.bin.aux.xml'):
        try:
            os.remove(os.path.join(cache_dir, f'{fire_numbe}_crop{ext}'))
        except FileNotFoundError:
            pass
        except OSError:
            pass
    if not crop_raster(ref_raster, crop_bin,
                       crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        _set_fire_status(fire, FireStatus.ERROR, "GDAL crop failed")
        return
    fire.crop_bin = crop_bin
    fire.perim_bin = ''

    # -- Re-rasterize the cumulative VIIRS shapefile onto the crop frame --
    # rasterize_shapefile already skips when the output exists, but its
    # output reflects whatever crop frame produced it. Invalidate only
    # when the crop bounds actually changed; otherwise let the built-in
    # skip return immediately. This is what makes a serial mapping sweep
    # over many paddings cheap on the steady state — the dominant cost
    # of the re-prepare path used to be rasterize on every padding bump.
    crop_rast_dir = os.path.join(cache_dir, '_viirs_crop')
    bounds_file = os.path.join(crop_rast_dir, '.crop_bounds')
    bounds_key = (f'{crop_xmin:.3f},{crop_ymin:.3f},'
                  f'{crop_xmax:.3f},{crop_ymax:.3f}')
    cached_bounds = None
    if os.path.isfile(bounds_file):
        try:
            with open(bounds_file, 'r') as f:
                cached_bounds = f.read().strip()
        except OSError:
            cached_bounds = None
    if cached_bounds != bounds_key and os.path.isdir(crop_rast_dir):
        shutil.rmtree(crop_rast_dir, ignore_errors=True)
    os.makedirs(crop_rast_dir, exist_ok=True)
    # The bounds_key cache only invalidates on crop-extent change. If
    # the cumulative shapefile itself was rebuilt with new days but the
    # crop bounds happen to match, rasterize_shapefile would short-
    # circuit on the existing .bin and silently return stale data.
    # Force re-rasterize whenever the .shp is newer than the .bin.
    _invalidate_stale_rasterize(acc_shp, crop_rast_dir)
    viirs_bin = None
    try:
        viirs_bin = rasterize_shapefile(
            shp_path=acc_shp, ref_image=crop_bin,
            output_dir=crop_rast_dir, buffer_m=375.0,
        )
        if viirs_bin and cached_bounds != bounds_key:
            try:
                with open(bounds_file, 'w') as f:
                    f.write(bounds_key)
            except OSError:
                pass
    except Exception as exc:
        sys.stderr.write(
            f'[prepare] [{fire_numbe}] re-rasterize failed: {exc}\n')
        sys.stderr.flush()
        viirs_bin = None

    if not viirs_bin or not os.path.isfile(viirs_bin):
        _set_fire_status(
            fire, FireStatus.ERROR,
            'Cannot re-prepare hint: VIIRS shapefile missing or empty.')
        return

    fire.hint_bin = viirs_bin
    fire.viirs_bin = viirs_bin
    fire.perimeter_type = 'viirs'

    if fire.viirs_start_date:
        fire.acc_start = fire.viirs_start_date
    if fire.viirs_end_date:
        fire.acc_end = fire.viirs_end_date

    # -- Generate preview images --
    views = generate_all_previews(crop_bin, cache_dir, fire_numbe)
    fire.available_views = views

    # -- Copy results from canonical dir for previously accepted fires --
    canon_dir = os.path.join(state.output_root, fire_numbe)
    if os.path.isdir(canon_dir):
        copied = []
        for fname in os.listdir(canon_dir):
            src = os.path.join(canon_dir, fname)
            dst = os.path.join(cache_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(fname)
        if copied:
            sys.stderr.write(
                f'[prepare] [{fire_numbe}] Restored {len(copied)} '
                f'file(s) from accepted dir\n')
            sys.stderr.flush()

    # -- Find classified raster (try multiple naming patterns) --
    clf_path = None
    for pattern in (f'{fire_numbe}_crop.bin_classified.bin',
                    f'{fire_numbe}_crop_classified.bin',
                    f'{fire_numbe}_classified.bin'):
        candidate = os.path.join(cache_dir, pattern)
        if os.path.isfile(candidate):
            clf_path = candidate
            break
    if clf_path is None:
        # Last resort: any *classified*.bin
        for candidate in glob.glob(
                os.path.join(cache_dir, '*classified*.bin')):
            clf_path = candidate
            break

    # -- Generate overlay previews (always try both) --
    if clf_path and os.path.isfile(clf_path):
        # Point fire at the classified raster for overlay generation
        _overlay_mask_on_post(fire, clf_path, 'result', (0.9, 0.1, 0.0))
        if 'result' not in fire.available_views:
            fire.available_views.append('result')
        sys.stderr.write(
            f'[prepare] [{fire_numbe}] Generated ML classification '
            f'overlay from {os.path.basename(clf_path)}\n')
        sys.stderr.flush()
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))

    fire.status = FireStatus.READY
    _save_fire_state()


def _ensure_brush_comparison_in_cache(fire: 'FireInfo', cache_dir: str) -> None:
    """If the cache is missing a brush comparison PNG, try to render one
    from the pre- and post-brush masks available on disk.

    Inputs resolved in cache_dir:
      - brushed mask = ``{fire}_crop.bin_classified.bin`` (canonical;
        contains the brushed mask when brush succeeded, else the raw
        classification — the same data either way).
      - raw mask    = ``{fire}_crop.bin_classified_raw.bin`` (pre-brush
        backup; only exists when brush succeeded at least once).

    When both exist, renders a full before/after figure. When only the
    canonical mask exists, renders a figure where "After" falls back to
    the raw view and the title reflects the missing brush output. When
    neither exists, silently no-ops — the canonical dir just won't have
    a brush PNG, same as before.

    Best-effort: any rendering error is logged and swallowed so accept
    never fails because of a cosmetic figure.
    """
    fire_numbe = fire.fire_numbe
    out_path = os.path.join(cache_dir, f'{fire_numbe}_brush_comparison.png')
    if os.path.isfile(out_path):
        return

    brushed_path = os.path.join(
        cache_dir, f'{fire_numbe}_crop.bin_classified.bin')
    if not os.path.isfile(brushed_path):
        return

    raw_path = os.path.join(
        cache_dir, f'{fire_numbe}_crop.bin_classified_raw.bin')
    post_png = os.path.join(cache_dir, 'previews', 'post.png')
    if not os.path.isfile(post_png):
        return

    try:
        brushed = _read_envi_mask(brushed_path)
        if os.path.isfile(raw_path):
            raw = _read_envi_mask(raw_path)
            brushed_for_fig = brushed
        else:
            # No pre-brush backup on disk — we only have one mask. Show
            # it as "Before" and flag "After" as unavailable so the
            # figure is informative rather than misleadingly claiming
            # brushing happened.
            raw = brushed
            brushed_for_fig = None

        start = getattr(fire, 'acc_start', '') or ''
        end = getattr(fire, 'acc_end', '') or ''
        title = f'Fire: {fire_numbe}  —  class_brush comparison'
        if start or end:
            title += f'\nStart: {start}   |   End: {end}'
        _render_brush_comparison_png(
            raw, brushed_for_fig, post_png, out_path, title)
    except Exception as exc:
        sys.stderr.write(
            f'[accept] WARNING: brush comparison regen for '
            f'{fire_numbe}: {exc}\n')
        sys.stderr.flush()


def _accept_fire_sync(fire_numbe: str) -> str:
    """Copy results from cache to canonical dir, write params. Returns path."""
    fire = state.fires[fire_numbe]
    cache_dir = fire.cache_dir
    # Refuse to run with no cache_dir — glob.glob(os.path.join('',
    # '*.bin')) would silently fall through to the process CWD and
    # copy unrelated files into the canonical output dir.
    if not cache_dir or not os.path.isdir(cache_dir):
        raise RuntimeError(
            f'Cannot accept {fire_numbe}: cache_dir missing or invalid '
            f'({cache_dir!r}). Re-prepare the fire and try again.')
    if not state.output_root:
        raise RuntimeError(
            f'Cannot accept {fire_numbe}: output_root not configured.')
    fire_dir = os.path.join(state.output_root, fire_numbe)

    # Register this accept as in-progress so the background cache
    # sweeper treats cache_dir as hard-pinned for the duration.
    # Without this, _cache_sweep (which uses its own lock, not
    # _gpu_lock) could rmtree cache_dir mid-copy.
    # AUDIT-C3: refuse re-entry for the same fire. The set is intended
    # for cache-sweeper coordination, not mutual exclusion — but two
    # concurrent accepts on the same fire would race fire_dir rmtree
    # vs makedirs. Caller-side _gpu_lock currently serialises the only
    # call sites, but make this contract explicit so a future caller
    # that forgets the lock fails fast instead of corrupting fire_dir.
    with _accept_in_progress_lock:
        if fire_numbe in _accept_in_progress:
            raise RuntimeError(
                f'Accept already in progress for {fire_numbe}')
        _accept_in_progress.add(fire_numbe)
    try:
        if os.path.isdir(fire_dir):
            shutil.rmtree(fire_dir)
        os.makedirs(fire_dir)

        # Safety net: ensure {fire}_brush_comparison.png exists in cache
        # before the copy, regenerating from the pre/post-brush masks on
        # disk if it's missing. Guarantees the canonical dir always has a
        # brush comparison figure, even for fires mapped before
        # class_brush.exe was available (where the CLI produced a
        # "FAILED" figure that may have been cleaned up) or where the
        # serial accept path didn't supply one.
        _ensure_brush_comparison_in_cache(fire, cache_dir)

        # Only canonical/final artifacts belong in the output dir. Per-run
        # serial artifacts ({fire}_serial_{rid}*) live in .web_cache and
        # must not leak into the final result. Same for rebrush backups
        # (*_raw.bin / *_raw.hdr) which are cache-only pre-brush snapshots.
        for pattern in ('*.bin', '*.hdr', '*.png', '*.shp', '*.dbf',
                         '*.shx', '*.prj', '*.cpg'):
            for f in glob.glob(os.path.join(cache_dir, pattern)):
                basename = os.path.basename(f)
                if '_serial_' in basename:
                    continue
                if basename.endswith('_raw.bin') or basename.endswith('_raw.hdr'):
                    continue
                shutil.copy2(f, fire_dir)

        # Per-view preview PNGs (pre, post, hint, diff1..diffN, result)
        # live under cache_dir/previews/ — a subdirectory the top-level
        # glob above never traverses. Without this copy the canonical
        # accept dir loses every diff/anomaly group view as soon as the
        # cache sweeper reaps .web_cache. Mirror the previews/ tree
        # into the fire_dir, skipping per-run serial overlays which
        # are gallery-only.
        src_previews = os.path.join(cache_dir, 'previews')
        if os.path.isdir(src_previews):
            dst_previews = os.path.join(fire_dir, 'previews')
            os.makedirs(dst_previews, exist_ok=True)
            for fname in os.listdir(src_previews):
                if fname.startswith('serial_'):
                    continue
                src = os.path.join(src_previews, fname)
                if not os.path.isfile(src):
                    continue
                try:
                    shutil.copy2(src, os.path.join(dst_previews, fname))
                except OSError as exc:
                    sys.stderr.write(
                        f'[accept] [{fire_numbe}] previews copy '
                        f'{fname}: {exc}\n')
                    sys.stderr.flush()

        # Compute ML area from the accepted dir
        clf_bin = os.path.join(
            fire_dir, f'{fire_numbe}_crop.bin_classified.bin')
        ml_area_val = _compute_ml_area(fire, clf_bin)
        ml_area_ha = ml_area_val if ml_area_val >= 0 else None
        ml_area_m2 = (ml_area_ha * 10000.0) if ml_area_ha is not None else None
        fire.ml_area_ha = ml_area_val

        # AUDIT-M4: yaml is a hard dependency; the prior `except ImportError`
        # was unreachable. Run the dict construction inline and narrow the
        # except to OSError around the actual disk write.
        # Write params YAML
        params_dict = {
            'fire': {
                'fire_numbe': fire_numbe,
                'fire_size_ha': fire.fire_size_ha,
                'ml_area_ha': ml_area_ha,
                'ml_area_m2': ml_area_m2,
                'agreement_pct': fire.agreement_pct,
                'notes': fire.notes or '',
            },
            'run': {
                'timestamp': datetime.datetime.now().isoformat(
                    timespec='seconds'),
                'source': 'web',
            },
            'inputs': {
                'raster': state.raster_path,
                'perimeter_type': fire.perimeter_type,
            },
            'crop': {
                'padding': fire.padding_used,
                'width_px': fire.crop_w,
                'height_px': fire.crop_h,
                'total_px': fire.crop_w * fire.crop_h,
            },
            'sampling': {
                'sample_rate': state.sample_rate,
                'actual_sample_size': fire.sample_size,
            },
            'accumulation': {
                'start_date': fire.acc_start,
                'end_date': fire.acc_end,
            },
        }
        if fire.last_params:
            # fire.last_params is a FLAT CLI-style dict (e.g.
            # 'hdbscan_min_samples', 'tsne_perplexity', 'embed_bands',
            # 'rf_n_estimators', 'brush_size'). The previous version
            # expected nested sub-dicts under 'tsne'/'hdbscan'/
            # 'random_forest' keys and silently wrote nothing, so
            # every accepted YAML (and the PDF built from it) lost
            # bands, t-SNE, RF, HDBSCAN, and brush settings. Group by
            # prefix so readers can pull a whole stage without string
            # parsing; unknown keys fall into 'misc'.
            _prefix_to_section = (
                ('tsne_',    'tsne'),
                ('hdbscan_', 'hdbscan'),
                ('rf_',      'random_forest'),
                ('brush_',   'brush'),
            )
            _explicit = {
                'embed_bands':       'bands',
                'point_threshold':   'brush',
                'controlled_ratio':  'random_forest',
                'contour_width':     'output',
                # New A* / B* tuning — group under semantically clean
                # section names instead of falling into 'misc'.
                'hint_aware_brush':       'brush',
                'stratify':               'sampling',
                'stratify_inside_ratio':  'sampling',
                'scale_features':         'preprocessing',
                'spatial_weight':         'embedding',
                'cluster_score_threshold': 'vote',
            }
            # These are already represented in higher-level sections
            # (crop/sampling). Skip to avoid duplication/conflicting
            # values if the per-run override differs from the global.
            _skip = {'padding', 'sample_rate', 'min_samples', 'max_samples'}
            for k, v in fire.last_params.items():
                if v is None or v == '':
                    continue
                if k in _skip:
                    continue
                section = None
                for prefix, sec in _prefix_to_section:
                    if k.startswith(prefix):
                        section = sec
                        break
                if section is None:
                    section = _explicit.get(k, 'misc')
                params_dict.setdefault(section, {})[k] = v

        path = os.path.join(fire_dir, f'{fire_numbe}_params.yaml')
        try:
            _atomic_yaml_dump(path, params_dict, mode=0o644)
        except OSError as exc:
            sys.stderr.write(
                f'[save] WARNING: {fire_numbe}_params.yaml: {exc}\n')
            sys.stderr.flush()

        # Update fire_status.yaml (atomic write). Hold the file lock across
        # the read-modify-write so concurrent accepts of different fires
        # don't lose each other's entries.
        try:
            import yaml
            status_path = os.path.join(state.output_root, 'fire_status.yaml')
            with _accept_file_lock:
                idx = {}
                if os.path.exists(status_path):
                    with open(status_path) as f:
                        idx = yaml.safe_load(f) or {}
                idx[fire_numbe] = {
                    'status': 'accepted',
                    'timestamp': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                    'fire_dir': fire_dir,
                    'source': 'web',
                }
                _atomic_yaml_dump(status_path, idx)
        except Exception as exc:
            # AUDIT-C2: don't swallow fire_status.yaml write failures
            # silently — surface to stderr like other persistence helpers.
            sys.stderr.write(
                f'[save] WARNING: fire_status.yaml update failed for '
                f'{fire_numbe}: {exc}\n')
            sys.stderr.flush()

        # Clean up XML artefacts
        for xml in glob.glob(os.path.join(fire_dir, '*.xml')):
            try:
                os.remove(xml)
            except Exception:
                pass

        # Append to accepted_params.csv for parameter learning (deduplicate).
        # The full read-dedupe-rewrite-append sequence runs under the file
        # lock so concurrent accepts cannot interleave and corrupt the file.
        try:
            import csv
            csv_path = os.path.join(state.output_root, 'accepted_params.csv')
            with _accept_file_lock:
                # Read existing rows (if any), drop the row for this fire
                # (dedupe on re-accept), then write everything + the new row
                # in a single tmp-file + rename so a crash or disk-full
                # cannot truncate the CSV mid-write.
                existing = []
                if os.path.isfile(csv_path):
                    with open(csv_path, newline='') as cf:
                        reader = csv.DictReader(cf)
                        existing = [r for r in reader
                                    if r.get('fire_numbe') != fire_numbe]

                row_data = {
                    'fire_numbe': fire_numbe,
                    'fire_size_ha': fire.fire_size_ha,
                    'agreement_pct': fire.agreement_pct,
                    'padding': fire.padding_used,
                    'timestamp': datetime.datetime.now().isoformat(
                        timespec='seconds'),
                }
                if fire.last_params:
                    for k, v in fire.last_params.items():
                        row_data[k] = v

                tmp_path = (
                    f'{csv_path}.{os.getpid()}.{threading.get_ident()}.tmp')
                try:
                    with open(tmp_path, 'w', newline='') as cf:
                        writer = csv.DictWriter(
                            cf, fieldnames=_CSV_FIELDNAMES,
                            extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(existing)
                        writer.writerow(row_data)
                        cf.flush()
                        os.fsync(cf.fileno())
                    os.replace(tmp_path, csv_path)
                    # AUDIT-C1: parent dir fsync — see AUDIT_REPORT.md.
                    dir_fd = os.open(
                        os.path.dirname(csv_path) or '.', os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
        except Exception as exc:
            sys.stderr.write(
                f'[save] WARNING: Failed to update accepted_params.csv: '
                f'{exc}\n')

        # Generate KML deliverable in EPSG:4326. Warn-and-continue on
        # failure — KML is for Google Earth viewing, not analysis.
        _export_kml(fire_numbe, fire_dir)

        # Re-point last_comparison at the canonical copy. Until now
        # it points into cache_dir, which _cache_sweep is free to
        # reap once status flips to ACCEPTED — that would leave the
        # UI / PDF builder pointing at a deleted file.
        canon_comp = os.path.join(
            fire_dir, f'{fire_numbe}_comparison.png')
        if os.path.isfile(canon_comp):
            fire.last_comparison = canon_comp

        # Flip status + clear ephemeral tracking state under state.lock
        # so readers never observe a fire that is ACCEPTED but still
        # has a live progress snapshot. Per-run serial gallery cleanup
        # (fire.serial_results + on-disk serial_* files) is the
        # caller's responsibility — the mapping worker has the full
        # list and deletes the files in its cancel path; clearing the
        # list here would strand those files.
        with state.lock:
            fire.status = FireStatus.ACCEPTED
            fire.previously_accepted = False
            fire.previously_accepted_agreement_pct = -1.0
            fire.progress = {}
            if state.current_job:
                cur = state.current_job.get('fire_numbe', '')
                if cur.split(' (run')[0].strip() == fire_numbe:
                    state.current_job = None
        _save_fire_state()
        return fire_dir
    finally:
        with _accept_in_progress_lock:
            _accept_in_progress.discard(fire_numbe)
