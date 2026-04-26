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
from .preview import generate_all_previews
from .mapping import (
    _compute_ml_area, _overlay_mask_on_post, _generate_result_preview,
)
from .brush import _read_envi_mask, _render_brush_comparison_png
from .persistence import _save_fire_state

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
    """Prepare a fire for mapping: crop, VIIRS accumulate, hint, previews."""
    from batch_fire_mapping.run_fire_mapping import (
        raster_native_extent, crop_raster, rasterize_polygon,
    )
    from shapely.geometry import box as shapely_box

    fire = state.fires[fire_numbe]

    # Refuse only if another prepare is already running on this fire.
    # MAPPING is *not* a refusal condition: _serial_map_worker sets
    # status=MAPPING before calling this (see app.py:1090, 1384) and
    # then asks us to re-prep when padding changes or the crop/hint
    # files are missing. Treating MAPPING as "busy" here was the root
    # cause of batch-mapping failures on never-opened fires — the
    # guard no-op'd, crop_bin/hint_bin stayed empty, and the CLI
    # subprocess crashed on empty positional args.
    if fire.status == FireStatus.PREPARING:
        fire.error_msg = 'Cannot prepare: fire is currently preparing'
        return

    fire.status = FireStatus.PREPARING
    fire.error_msg = ""

    pad = padding if padding is not None else state.padding

    try:
        row = state.gdf[
            state.gdf['FIRE_NUMBE'].astype(str) == fire_numbe
        ].iloc[0]
    except (IndexError, KeyError):
        _set_fire_status(fire, FireStatus.ERROR,
                         f"Fire {fire_numbe} not found in shapefile")
        return

    # -- Parse FIRE_DATE --
    raw = row.get('FIRE_DATE', '')
    try:
        if hasattr(raw, 'date'):
            fire_date = datetime.datetime(raw.year, raw.month, raw.day)
        else:
            fire_date = datetime.datetime.strptime(
                str(raw).split()[0], '%Y-%m-%d')
    except (ValueError, AttributeError):
        _set_fire_status(fire, FireStatus.ERROR,
                         f"Cannot parse FIRE_DATE: {raw!r}")
        return

    acc_start = fire_date - datetime.timedelta(days=5)

    # -- Clip polygon to raster, compute crop bounds --
    gt = state.raster_gt
    W, H = state.raster_W, state.raster_H
    rx1, ry1, rx2, ry2 = raster_native_extent(gt, W, H)
    raster_box = shapely_box(rx1, ry1, rx2, ry2)

    clipped = row.geometry.intersection(raster_box)
    if clipped.is_empty:
        _set_fire_status(fire, FireStatus.ERROR,
                         "Fire polygon does not overlap the raster")
        return

    bounds = clipped.bounds  # (minx, miny, maxx, maxy)
    px_lo = int((bounds[0] - gt[0]) / gt[1])
    px_hi = int((bounds[2] - gt[0]) / gt[1])
    py_lo = int((bounds[3] - gt[3]) / gt[5])  # maxy -> top row
    py_hi = int((bounds[1] - gt[3]) / gt[5])  # miny -> bottom row

    fire_max_dim = max(px_hi - px_lo, py_hi - py_lo)
    p = max(1, int(round(pad * fire_max_dim)))
    px_lo = max(0, px_lo - p)
    px_hi = min(W - 1, px_hi + p)
    py_lo = max(0, py_lo - p)
    py_hi = min(H - 1, py_hi + p)

    if px_lo >= px_hi or py_lo >= py_hi:
        _set_fire_status(fire, FireStatus.ERROR,
                         "Crop box has zero area after clipping")
        return

    crop_xmin = gt[0] + px_lo * gt[1]
    crop_xmax = gt[0] + px_hi * gt[1]
    crop_ymax = gt[3] + py_lo * gt[5]
    crop_ymin = gt[3] + py_hi * gt[5]

    crop_w = px_hi - px_lo
    crop_h = py_hi - py_lo
    # Capture the old padding BEFORE mutating fire.padding_used so the
    # cache-wipe comparison below can detect a real change. Previously the
    # assignment happened first and the check was always False, so stale
    # crops survived a padding change and were reused with the new label.
    old_pad = fire.padding_used
    fire.crop_w = crop_w
    fire.crop_h = crop_h
    fire.padding_used = pad

    sample_size = int(round(crop_w * crop_h * state.sample_rate))
    sample_size = max(state.min_samples, min(state.max_samples, sample_size))
    fire.sample_size = sample_size

    # -- Create / clear cache directory --
    # Only wipe when padding actually changed; preserve existing results
    cache_dir = os.path.join(state.output_root, '.web_cache', fire_numbe)
    padding_changed = (old_pad != 0
                       and old_pad != pad
                       and os.path.isdir(cache_dir))
    if padding_changed:
        # Selective wipe: keep {fire}_serial_* (the gallery's
        # classified.bin + standalone comparison PNGs) and drop
        # everything else. A recommended-settings sweep whose
        # settings span multiple padding values re-enters this
        # branch between settings; the old shutil.rmtree wiped
        # every prior setting's gallery files off disk, leaving
        # in-memory fire.serial_results entries pointing at
        # non-existent files (ghost cards with no thumbnails).
        # Serial overlay PNGs under previews/ are tied to the
        # old post.png extent, so drop those — handle_api_serial_image
        # regenerates them on demand from the surviving
        # classified.bin via _overlay_mask_on_post, which aligns
        # across crop extents using GeoTransform.
        serial_prefix = f'{fire_numbe}_serial_'
        for fname in os.listdir(cache_dir):
            full_path = os.path.join(cache_dir, fname)
            if (os.path.isfile(full_path)
                    and not fname.startswith(serial_prefix)):
                try:
                    os.remove(full_path)
                except OSError:
                    pass
        previews_dir = os.path.join(cache_dir, 'previews')
        if os.path.isdir(previews_dir):
            shutil.rmtree(previews_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    fire.cache_dir = cache_dir

    # -- Crop raster --
    crop_bin = os.path.join(cache_dir, f'{fire_numbe}_crop.bin')
    if not crop_raster(state.raster_path, crop_bin,
                       crop_xmin, crop_ymin, crop_xmax, crop_ymax):
        _set_fire_status(fire, FireStatus.ERROR, "GDAL crop failed")
        return
    fire.crop_bin = crop_bin

    crop_gt = (crop_xmin, gt[1], gt[2], crop_ymax, gt[4], gt[5])

    # -- Rasterize traditional perimeter --
    perim_bin = os.path.join(cache_dir, f'{fire_numbe}_perimeter.bin')
    try:
        rasterize_polygon(
            state.polygon_file, fire_numbe, state.raster_crs,
            crop_bin, perim_bin, geometry=clipped,
            crop_gt=crop_gt, crop_w=crop_w, crop_h=crop_h)
    except Exception:
        perim_bin = None
    fire.perim_bin = perim_bin or ''

    # -- VIIRS accumulation --
    viirs_bin = None
    acc_end = fire_date
    plot_start = acc_start.date()
    plot_end = fire_date.date()

    if (state.perimeter_mode == 'viirs'
            and state.viirs_gdf is not None
            and not state.viirs_gdf.empty):
        from viirs.utils.accumulate import accumulate
        from viirs.utils.rasterize import rasterize_shapefile

        inside = state.viirs_gdf[
            state.viirs_gdf.geometry.within(row.geometry)]

        if not inside.empty:
            acc_end = datetime.datetime.combine(
                inside['detection_date'].max(), datetime.time.min)
            inside_window = inside[
                inside['detection_datetime'] >= acc_start]
            if not inside_window.empty:
                plot_start = inside_window['detection_date'].min()
            plot_end = acc_end.date()

        try:
            acc_paths = accumulate(
                shp_dir=state.viirs_shp_dir,
                start_str=acc_start.strftime('%Y%m%d'),
                end_str=acc_end.strftime('%Y%m%d'),
                reference_raster=crop_bin,
                output_dir=cache_dir,
                final_only=True,
                bbox=(crop_xmin, crop_ymin, crop_xmax, crop_ymax),
            )
        except Exception:
            acc_paths = []

        if acc_paths:
            try:
                viirs_bin = rasterize_shapefile(
                    shp_path=acc_paths[-1],
                    ref_image=crop_bin,
                    output_dir=cache_dir,
                    buffer_m=375.0,
                )
                if viirs_bin:
                    ds = gdal.Open(viirs_bin, gdal.GA_ReadOnly)
                    arr = ds.GetRasterBand(1).ReadAsArray()
                    ds = None
                    if np.nansum(arr) == 0:
                        viirs_bin = None
            except Exception:
                viirs_bin = None

    # -- Select hint raster --
    if state.perimeter_mode == 'traditional':
        fire.hint_bin = perim_bin or ''
        fire.perimeter_type = 'traditional'
    elif viirs_bin:
        fire.hint_bin = viirs_bin
        fire.viirs_bin = viirs_bin
        fire.perimeter_type = 'viirs'
    elif perim_bin and os.path.exists(perim_bin):
        fire.hint_bin = perim_bin
        fire.perimeter_type = 'traditional'
    else:
        _set_fire_status(fire, FireStatus.ERROR,
                         "No classification hint available")
        return

    fire.acc_start = str(plot_start)
    fire.acc_end = str(plot_end)

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
    with _accept_in_progress_lock:
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

        # Compute ML area from the accepted dir
        clf_bin = os.path.join(
            fire_dir, f'{fire_numbe}_crop.bin_classified.bin')
        ml_area_val = _compute_ml_area(fire, clf_bin)
        ml_area_ha = ml_area_val if ml_area_val >= 0 else None
        ml_area_m2 = (ml_area_ha * 10000.0) if ml_area_ha is not None else None
        fire.ml_area_ha = ml_area_val

        # Write params YAML
        try:
            import yaml
            params_dict = {
                'fire': {
                    'fire_numbe': fire_numbe,
                    'fire_date': fire.fire_date,
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
            _atomic_yaml_dump(path, params_dict, mode=0o644)
        except ImportError:
            pass

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
        except Exception:
            pass

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
                    os.replace(tmp_path, csv_path)
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
