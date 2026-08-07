"""Mapping result helpers — overlay PNG generation, area + agreement metrics.

These are pure-ish functions that read from raster files and write
preview PNGs into ``fire.cache_dir/previews/``. Stateful access to the
shared ``state`` (only ``state.raster_gt`` for pixel area) is wired by
:func:`init` at server boot.
"""

import json
import os
import re
import sys

import numpy as np
from osgeo import gdal

from .state import classified_names, find_classified, AppState, FireInfo

state: AppState = None


def init(app_state: AppState):
    global state
    state = app_state


def _compute_ml_area(fire: 'FireInfo',
                     clf_path: str = None) -> float:
    """Compute ML burned area in hectares from a classified raster.

    Returns area in ha or -1 if computation fails.
    """
    if clf_path is None:
        # Name follows the stack the CLI was given, not the old
        # <fire>_crop.bin convention.
        clf_path = find_classified(
            fire, [fire.cache_dir, os.path.dirname(fire.crop_bin or '')]) \
            or os.path.join(fire.cache_dir, classified_names(fire)[0])
    if not os.path.isfile(clf_path):
        return -1.0
    try:
        gt = state.raster_gt
        pixel_area_m2 = abs(gt[1] * gt[5])
        ds = gdal.Open(clf_path, gdal.GA_ReadOnly)
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        burned_px = int(np.nansum(arr > 0))
        ml_area_ha = burned_px * pixel_area_m2 / 10000.0
        return round(ml_area_ha, 2)
    except Exception as exc:
        sys.stderr.write(
            f'[ml_area] WARNING: Failed to compute ML area: {exc}\n')
        return -1.0


def diagnose_run(fire, label: str, clf_path: str,
                 hint_path: str = None, log=None) -> dict:
    """Report everything about a run's mask, hint and their overlap.

    Always runs, for every result -- not just failures. A 0% agreement
    or 0 ha area has many possible causes (empty classification, empty
    hint, misaligned geotransforms, mismatched shapes, an all-nodata
    band) and they are indistinguishable from the summary numbers
    alone. Emitting the raw counts every time means the next oddity is
    diagnosable from the log that was already captured, without having
    to reproduce it.
    """
    def emit(msg):
        if log:
            try:
                log(msg)
            except Exception:
                pass
        sys.stderr.write(msg + '\n')

    out = {'label': label}
    emit(f'  [diag] ===== {label} =====')

    def _describe(tag, path):
        info = {'path': path, 'exists': False}
        if not path:
            emit(f'  [diag] {tag}: <no path>')
            return info, None
        info['exists'] = os.path.isfile(path)
        if not info['exists']:
            emit(f'  [diag] {tag}: MISSING at {path}')
            return info, None
        try:
            info['bytes'] = os.path.getsize(path)
        except OSError:
            info['bytes'] = -1
        try:
            from osgeo import gdal
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                emit(f'  [diag] {tag}: GDAL could not open {path}')
                return info, None
            arr = ds.GetRasterBand(1).ReadAsArray()
            gt = ds.GetGeoTransform()
            info.update({'w': ds.RasterXSize, 'h': ds.RasterYSize,
                         'bands': ds.RasterCount, 'gt': list(gt),
                         'dtype': str(arr.dtype)})
            finite = np.isfinite(arr)
            nz = int(np.count_nonzero(np.nan_to_num(arr) > 0))
            info.update({
                'total_px': int(arr.size),
                'nonzero_px': nz,
                'nan_px': int((~finite).sum()),
                'min': float(np.nanmin(arr)) if finite.any() else None,
                'max': float(np.nanmax(arr)) if finite.any() else None,
            })
            vals, cnts = np.unique(
                np.nan_to_num(arr).astype('float32'), return_counts=True)
            info['uniques'] = [
                [float(v), int(c)] for v, c in
                zip(vals[:8], cnts[:8])]
            emit(f'  [diag] {tag}: {info["w"]}x{info["h"]} '
                 f'{info["dtype"]}  nonzero={nz:,}/{arr.size:,} '
                 f'({100.0 * nz / max(1, arr.size):.2f}%)  '
                 f'nan={info["nan_px"]:,}  range=[{info["min"]}, '
                 f'{info["max"]}]')
            emit(f'  [diag] {tag}: gt origin=({gt[0]:.1f}, {gt[3]:.1f}) '
                 f'px=({gt[1]:.4f}, {gt[5]:.4f})')
            emit(f'  [diag] {tag}: first uniques={info["uniques"]}')
            ds = None
            return info, np.nan_to_num(arr) > 0
        except Exception as exc:
            emit(f'  [diag] {tag}: read failed: {exc}')
            return info, None

    out['clf'], m_clf = _describe('classification', clf_path)
    hp = hint_path if hint_path is not None else getattr(
        fire, 'hint_bin', '')
    out['hint'], m_hint = _describe('hint', hp)

    if m_clf is None or m_hint is None:
        emit('  [diag] overlap: not computable (one raster unreadable)')
        emit(f'  [diag] ===== end {label} =====')
        return out

    if m_clf.shape != m_hint.shape:
        # Not fatal -- _compute_agreement aligns via geotransform --
        # but it is the usual reason a cross-padding run scores 0.
        emit(f'  [diag] overlap: SHAPE MISMATCH clf={m_clf.shape} '
             f'hint={m_hint.shape}; agreement uses geotransform '
             f'alignment over the common rectangle')
        emit(f'  [diag] ===== end {label} =====')
        return out

    inter = int(np.count_nonzero(m_clf & m_hint))
    union = int(np.count_nonzero(m_clf | m_hint))
    out['intersection'] = inter
    out['union'] = union
    out['iou'] = (100.0 * inter / union) if union else -1.0
    emit(f'  [diag] overlap: clf={int(m_clf.sum()):,}  '
         f'hint={int(m_hint.sum()):,}  intersection={inter:,}  '
         f'union={union:,}  IoU={out["iou"]:.2f}%')
    if inter == 0:
        emit('  [diag] overlap: ZERO intersection -- classification and '
             'hint do not share a single pixel. Check the hint mode '
             'and that both were built from the SAME post source.')
    emit(f'  [diag] ===== end {label} =====')
    return out


def ensure_overlay_current(fire, out_name: str, clf_path: str,
                           colour=(0.9, 0.1, 0.0)) -> bool:
    """Guarantee an overlay PNG is in the CURRENT crop's grid.

    This removes the misalignment at its source instead of describing
    it. Every previous fix tried to TELL the client what extent a
    result PNG had -- via names, then a sidecar, then HTTP headers --
    and each one broke somewhere new, because 'result.png' is a copy
    and its provenance kept getting lost.

    The overlay renderer already resamples a mask from any extent into
    the current crop's grid (same pixel size, integer offset -- always
    true here, everything is 20 m). So if the overlay is simply
    re-rendered whenever the crop has changed underneath it, the
    result is ALWAYS pixel-identical in extent and size to post.png.
    Split-view alignment then needs no georeferencing at all for the
    result: it is aligned by construction, and cannot drift again.

    Returns True if a re-render happened.
    """
    try:
        if not clf_path or not os.path.isfile(clf_path):
            return False
        if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
            return False

        from osgeo import gdal
        ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
        if ds is None:
            return False
        cur = {'gt': [float(v) for v in ds.GetGeoTransform()],
               'rw': ds.RasterXSize, 'rh': ds.RasterYSize}
        ds = None

        png = os.path.join(fire.cache_dir, 'previews', f'{out_name}.png')
        entry = None
        gj = os.path.join(fire.cache_dir, 'previews', 'geo.json')
        if os.path.isfile(gj):
            try:
                with open(gj, encoding='utf-8') as f:
                    entry = (json.load(f) or {}).get(out_name)
            except (OSError, ValueError):
                entry = None

        same = (
            entry is not None
            and os.path.isfile(png)
            and int(entry.get('rw', -1)) == cur['rw']
            and int(entry.get('rh', -1)) == cur['rh']
            and all(abs(a - b) < 1e-6 for a, b in
                    zip(entry.get('gt', []), cur['gt']))
        )
        if same:
            return False

        why = ('no overlay on disk' if not os.path.isfile(png)
               else 'no recorded geo' if entry is None
               else f"grid changed "
                    f"({entry.get('rw')}x{entry.get('rh')} -> "
                    f"{cur['rw']}x{cur['rh']})")
        sys.stderr.write(
            f'[geo] {out_name}: re-rendering into the current AOI grid '
            f'({why}) so it matches the post-fire preview exactly\n')
        _overlay_mask_on_post(fire, clf_path, out_name, colour)
        return True
    except Exception as exc:
        sys.stderr.write(
            f'[geo] {out_name}: re-render check failed: {exc}\n')
        return False


def rerender_run_overlays(fire, log=None) -> int:
    """Re-render every run overlay onto the CURRENT crop grid.

    This removes the bug class rather than patching it again.

    The recurring failure was that previews are rendered into whatever
    crop existed AT RENDER TIME, and a settings sweep re-preps at
    several paddings. So serial_1.png, result.png and the live
    post.png could each sit on a DIFFERENT grid, and the split view
    had to reconcile them from recorded metadata. Every mechanism for
    carrying that metadata (view names, sidecars, file copies, HTTP
    headers) is one more thing that can desync -- and each one did.

    _overlay_mask_on_post already resamples a mask whose geotransform
    differs onto the current crop, so re-running it after any crop
    change makes ALL previews share ONE grid. Alignment then needs no
    metadata at all: identical extents are identical by construction,
    and the geo plumbing becomes belt-and-braces rather than the thing
    correctness depends on.

    Returns the number of overlays re-rendered.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    n = 0
    try:
        if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
            return 0
        cache = fire.cache_dir
        # Every run's classified raster, plus whichever one 'result'
        # currently represents.
        pat = re.compile(
            rf'^{re.escape(fire.fire_numbe)}_serial_(\d+)'
            rf'_classified\.bin$')
        runs = []
        for f in sorted(os.listdir(cache)):
            m = pat.match(f)
            if m:
                runs.append((int(m.group(1)), os.path.join(cache, f)))
        runs.sort()

        for rid, clf in runs:
            try:
                _overlay_mask_on_post(fire, clf, f'serial_{rid}',
                                      (0.9, 0.1, 0.0))
                n += 1
            except Exception as exc:
                emit(f'  [geo] re-render serial_{rid} failed: {exc}')

        if runs:
            # 'result' mirrors the newest run.
            newest_id, newest_clf = runs[-1]
            try:
                _overlay_mask_on_post(fire, newest_clf, 'result',
                                      (0.9, 0.1, 0.0))
                copy_preview_geo(cache, f'serial_{newest_id}', 'result')
                n += 1
            except Exception as exc:
                emit(f'  [geo] re-render result failed: {exc}')

        if n:
            emit(f'  [geo] re-rendered {n} run overlay(s) onto the '
                 f'current AOI grid -- all views now share one '
                 f'geotransform, so the split view aligns exactly.')
    except Exception as exc:
        emit(f'  [geo] run overlay re-render failed: {exc}')
    return n


def copy_preview_geo(cache_dir: str, src_name: str,
                     dst_name: str) -> bool:
    """Make a preview's georeferencing follow a file copy.

    previews/result.png is a byte copy of previews/serial_<N>.png, but
    copying the pixels left geo.json['result'] pointing at whatever was
    recorded earlier -- typically the crop at a DIFFERENT padding. The
    result view then advertised the wrong extent, which is the ML
    misalignment and the phantom zoom on flicker.

    Returns True when an entry was carried across.
    """
    try:
        gj = os.path.join(cache_dir, 'previews', 'geo.json')
        if not os.path.isfile(gj):
            return False
        with open(gj, encoding='utf-8') as f:
            data = json.load(f) or {}
        if src_name not in data:
            sys.stderr.write(
                f'[geo] copy {src_name} -> {dst_name}: no source entry; '
                f'{dst_name} would report a stale extent\n')
            return False
        data[dst_name] = dict(data[src_name])
        tmp = gj + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, gj)
        sys.stderr.write(
            f'[geo] copied georeferencing {src_name} -> {dst_name}\n')
        return True
    except Exception as exc:
        sys.stderr.write(f'[geo] copy failed: {exc}\n')
        return False


def record_preview_geo(cache_dir: str, raster_path: str,
                       out_name: str, png_path: str):
    """Record the georeferencing of a rendered preview PNG.

    Preview PNGs are rendered in the CROP's coordinate space at the
    moment of rendering, then downsampled. A settings sweep re-preps
    the fire at different paddings, so serial_1.png, serial_4.png and
    the live post.png can each cover a DIFFERENT ground extent.

    Nothing about the finished PNG records which extent it came from,
    so split-view sync had no way to line two of them up -- matching by
    pixel or by fraction is wrong whenever the extents differ, which is
    precisely the ML-result case.

    Writing a sidecar at render time makes each PNG self-describing:
    <cache>/previews/geo.json maps view name -> {gt, rw, rh, w, h},
    where gt/rw/rh are the source raster's geotransform and size and
    w/h are the PNG's.
    """
    try:
        from osgeo import gdal
        src = raster_path
        if not src or not os.path.isfile(src):
            return
        ds = gdal.Open(src, gdal.GA_ReadOnly)
        if ds is None:
            return
        gt = [float(v) for v in ds.GetGeoTransform()]
        rw, rh = ds.RasterXSize, ds.RasterYSize
        ds = None

        pw = ph = 0
        try:
            from matplotlib.image import imread
            a = imread(png_path)
            ph, pw = a.shape[0], a.shape[1]
        except Exception:
            pass

        gj = os.path.join(cache_dir, 'previews', 'geo.json')
        data = {}
        if os.path.isfile(gj):
            try:
                with open(gj, encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, ValueError):
                data = {}
        data[out_name] = {'gt': gt, 'rw': rw, 'rh': rh,
                          'w': pw or rw, 'h': ph or rh}
        tmp = gj + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, gj)
    except Exception as exc:
        sys.stderr.write(f'[overlay] geo record failed for '
                         f'{out_name}: {exc}\n')


def diagnose_run(fire, clf_path: str = None, hint_path: str = None,
                 label: str = 'run') -> list:
    """Full accounting of one mapping run's inputs and outputs.

    Emitted for EVERY run, not just suspicious ones: when a number
    later looks wrong there is no way to go back and collect the state
    that produced it, and re-running does not reproduce a transient.

    Reports each raster's existence, dtype, dimensions, geotransform,
    and value histogram, then the overlap arithmetic that agreement is
    computed from. That is enough to distinguish the failure modes that
    all present identically as "0% agreement / 0 ha":

      * classification empty      -> clf positives = 0
      * hint empty                -> hint positives = 0
      * grids disagree            -> dims or geotransform differ
      * no spatial overlap        -> intersection area = 0
      * mask written but not read -> file missing at the expected path
    """
    out = []

    def _say(m):
        out.append(m)

    try:
        from osgeo import gdal
        import numpy as _np

        clf_path = clf_path or find_classified(
            fire, [fire.cache_dir,
                   os.path.dirname(fire.crop_bin or '')])
        hint_path = hint_path or fire.hint_bin

        _say(f'  [diag:{label}] ---- mapping diagnostics ----')
        _say(f'  [diag:{label}] post_source={getattr(fire, "post_source", "?")} '
             f'hint_mode={getattr(fire, "hint_mode", "?")} '
             f'padding={getattr(fire, "padding_used", "?")}')
        _say(f'  [diag:{label}] stack={os.path.basename(fire.crop_bin or "-")}')

        def _describe(tag, path):
            if not path:
                _say(f'  [diag:{label}] {tag}: <no path recorded>')
                return None
            if not os.path.isfile(path):
                _say(f'  [diag:{label}] {tag}: MISSING at {path}')
                return None
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            if ds is None:
                _say(f'  [diag:{label}] {tag}: unreadable {path}')
                return None
            gt = ds.GetGeoTransform()
            a = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            n = a.size
            nz = int(_np.count_nonzero(_np.nan_to_num(a)))
            nan = int(_np.isnan(a).sum()) if a.dtype.kind == 'f' else 0
            vals, cnts = _np.unique(
                _np.nan_to_num(a).astype('float64'), return_counts=True)
            top = ', '.join(
                f'{v:g}x{c}' for v, c in
                sorted(zip(vals, cnts), key=lambda t: -t[1])[:5])
            _say(f'  [diag:{label}] {tag}: {os.path.basename(path)} '
                 f'{a.shape[1]}x{a.shape[0]} dtype={a.dtype} '
                 f'nonzero={nz:,}/{n:,} ({100.0 * nz / max(1, n):.2f}%) '
                 f'nan={nan:,}')
            _say(f'  [diag:{label}] {tag}: origin=({gt[0]:.1f}, {gt[3]:.1f}) '
                 f'px={gt[1]:.4f}x{gt[5]:.4f}  values[{top}]')
            return {'a': a, 'gt': gt}

        clf = _describe('clf ', clf_path)
        hnt = _describe('hint', hint_path)

        if clf is None or hnt is None:
            _say(f'  [diag:{label}] cannot compare -- a raster is missing.')
            return out

        ca, ha = clf['a'], hnt['a']
        cpos = _np.nan_to_num(ca) > 0
        hpos = _np.nan_to_num(ha) > 0
        _say(f'  [diag:{label}] clf positives={int(cpos.sum()):,}  '
             f'hint positives={int(hpos.sum()):,}')

        if ca.shape != ha.shape:
            _say(f'  [diag:{label}] SHAPE MISMATCH clf{ca.shape} vs '
                 f'hint{ha.shape} -- agreement needs geotransform '
                 f'alignment; a mismatch here with differing origins '
                 f'means the two came from different paddings.')
            dox = (clf['gt'][0] - hnt['gt'][0]) / (clf['gt'][1] or 1)
            doy = (clf['gt'][3] - hnt['gt'][3]) / (clf['gt'][5] or 1)
            _say(f'  [diag:{label}] origin offset = ({dox:.1f}, {doy:.1f}) px')
        else:
            inter = int((cpos & hpos).sum())
            union = int((cpos | hpos).sum())
            _say(f'  [diag:{label}] intersection={inter:,}  union={union:,}  '
                 f'IoU={100.0 * inter / max(1, union):.2f}%')
            if int(cpos.sum()) == 0:
                _say(f'  [diag:{label}] CAUSE: classification is empty -- '
                     f'no cluster passed the score threshold, or the '
                     f'burned cluster was written as zeros.')
            elif int(hpos.sum()) == 0:
                _say(f'  [diag:{label}] CAUSE: hint mask is empty -- '
                     f'red-wins found no fire pixels, so there is '
                     f'nothing to agree with.')
            elif inter == 0:
                _say(f'  [diag:{label}] CAUSE: masks do not overlap at all '
                     f'despite matching grids -- likely classifying the '
                     f'inverse cluster.')
        _say(f'  [diag:{label}] ---- end diagnostics ----')
    except Exception as exc:
        _say(f'  [diag:{label}] diagnostics failed: {exc}')
    return out


def record_base_preview_geo(cache_dir: str, crop_bin: str) -> None:
    """Record geo for the previews generate_all_previews() writes."""
    for _v in ('pre', 'post', 'diff1', 'diff2', 'diff3'):
        _p = os.path.join(cache_dir, 'previews', f'{_v}.png')
        if os.path.isfile(_p):
            record_preview_geo(cache_dir, crop_bin, _v, _p)


def _overlay_mask_on_post(fire: 'FireInfo', raster_path: str,
                          out_name: str, color: tuple):
    """Overlay a binary raster on the post-fire preview.

    *color* is (r, g, b) floats 0-1 for the tint.
    Produces a pixel-aligned PNG at the same dimensions as post.png.

    When the overlay raster has different dimensions from the current
    crop (e.g. a previously accepted classification after re-cropping
    with different padding), uses GDAL geotransforms to place it at
    the correct geographic position rather than naively stretching.
    """
    try:
        post_path = os.path.join(fire.cache_dir, 'previews', 'post.png')
        if not os.path.isfile(post_path):
            sys.stderr.write(
                f'[overlay] WARNING: cannot build {out_name} overlay — '
                f'post preview missing at {post_path}\n')
            return
        if not os.path.isfile(raster_path):
            sys.stderr.write(
                f'[overlay] WARNING: cannot build {out_name} overlay — '
                f'mask raster missing at {raster_path}\n')
            return

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.image import imread, imsave
        from scipy.ndimage import zoom as scipy_zoom

        post = imread(post_path)
        if post.ndim == 2:
            post = np.stack([post] * 3, axis=2)

        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        arr = ds.GetRasterBand(1).ReadAsArray()
        old_gt = ds.GetGeoTransform()
        ds = None

        ph, pw = post.shape[:2]
        ah, aw = arr.shape

        if ah != ph or aw != pw:
            aligned = False
            # Try geospatial alignment using crop geotransform.
            # Both rasters are crops of the same source, so pixel
            # sizes match — we just need to compute the offset.
            if fire.crop_bin and os.path.isfile(fire.crop_bin):
                try:
                    ds_crop = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
                    new_gt = ds_crop.GetGeoTransform()
                    new_w = ds_crop.RasterXSize
                    new_h = ds_crop.RasterYSize
                    ds_crop = None

                    if (old_gt and new_gt
                            and abs(old_gt[1] - new_gt[1]) < 1e-6
                            and abs(old_gt[5] - new_gt[5]) < 1e-6):
                        # Pixel sizes match — compute offset
                        off_x = round(
                            (old_gt[0] - new_gt[0]) / new_gt[1])
                        off_y = round(
                            (old_gt[3] - new_gt[3]) / new_gt[5])

                        # Place old raster in crop-sized array
                        arr_aligned = np.zeros(
                            (new_h, new_w), dtype=arr.dtype)
                        src_y0 = max(0, -off_y)
                        src_x0 = max(0, -off_x)
                        dst_y0 = max(0, off_y)
                        dst_x0 = max(0, off_x)
                        copy_h = min(ah - src_y0, new_h - dst_y0)
                        copy_w = min(aw - src_x0, new_w - dst_x0)
                        if copy_h > 0 and copy_w > 0:
                            arr_aligned[
                                dst_y0:dst_y0 + copy_h,
                                dst_x0:dst_x0 + copy_w,
                            ] = arr[
                                src_y0:src_y0 + copy_h,
                                src_x0:src_x0 + copy_w,
                            ]

                        # Scale to match preview PNG dimensions
                        # (preview may be downsampled from crop)
                        if new_h != ph or new_w != pw:
                            arr_aligned = scipy_zoom(
                                arr_aligned.astype(np.float32),
                                (ph / new_h, pw / new_w), order=0)

                        arr = arr_aligned
                        aligned = True
                except Exception:
                    pass

            if not aligned:
                # Fallback: naive resize (same-extent rasters)
                arr = scipy_zoom(
                    arr.astype(np.float32),
                    (ph / ah, pw / aw), order=0)

        mask = arr > 0
        result = post[:, :, :3].copy()
        r, g, b = color
        result[mask, 0] = np.clip(result[mask, 0] * 0.3 + r * 0.7, 0, 1)
        result[mask, 1] = np.clip(result[mask, 1] * 0.3 + g * 0.7, 0, 1)
        result[mask, 2] = np.clip(result[mask, 2] * 0.3 + b * 0.7, 0, 1)

        # An empty mask produces a PNG identical to the post-fire
        # preview. Writing it anyway means the pane is labelled
        # "ML classification" while showing plain post imagery, and --
        # worse -- split sync then uses the result's padded extent for
        # a picture that is really the crop, which is the phantom zoom
        # on flicker. Refuse to create the view instead.
        n_mask = int(np.count_nonzero(mask))
        if n_mask == 0 and out_name != 'hint':
            sys.stderr.write(
                f'[overlay] {out_name}: mask is empty (0 pixels) -- '
                f'not writing a preview, so the view is not offered.\n')
            for _stale in (os.path.join(fire.cache_dir, 'previews',
                                        f'{out_name}.png'),):
                try:
                    os.remove(_stale)
                except OSError:
                    pass
            if out_name in fire.available_views:
                fire.available_views.remove(out_name)
            return

        out_path = os.path.join(fire.cache_dir, 'previews', f'{out_name}.png')
        imsave(out_path, np.clip(result, 0, 1))
        # The PNG is in the CURRENT crop's space; record that so split
        # sync can align it against previews from other paddings.
        record_preview_geo(fire.cache_dir, fire.crop_bin,
                           out_name, out_path)

        # Register as a selectable view only if it IS one. Per-mode
        # hint renders (hint_redwins_post, hint_redwins_diff, ...) are
        # written through this same helper but are chosen with the Hint
        # buttons and served via ?hint=; letting them auto-register put
        # them in the view dropdown and, since the client validates
        # against that list, made real views report as unavailable.
        if (out_name not in fire.available_views
                and not out_name.startswith('serial_')
                and not out_name.startswith('hint_')):
            fire.available_views.append(out_name)
    except Exception as exc:
        import traceback
        sys.stderr.write(
            f'[overlay] WARNING: Failed to generate {out_name} '
            f'overlay: {exc}\n{traceback.format_exc()}')


def _generate_result_preview(fire: 'FireInfo'):
    """Generate pixel-aligned overlay previews after mapping."""
    clf_path = find_classified(
        fire, [fire.cache_dir, os.path.dirname(fire.crop_bin or '')])
    if not clf_path:
        clf_path = os.path.join(fire.cache_dir, classified_names(fire)[0])
    _overlay_mask_on_post(fire, clf_path, 'result', (0.9, 0.1, 0.0))

    # Also generate hint overlay if hint raster exists
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))


def _compute_agreement(fire: 'FireInfo',
                       clf_path: str | None = None) -> float:
    """Compute overlap % between ML classification and hint perimeter.

    When *clf_path* is None, reads the main crop's classified.bin;
    callers can pass a per-run classified.bin for serial-run agreement.
    Returns percentage (0-100) or -1 if computation fails.

    When clf and hint have different shapes (e.g. rebrush on a serial
    run whose padding differs from the current fire.hint_bin extent —
    recommended_settings sweeps can span multiple paddings), aligns
    them via GeoTransform and computes IoU over the common overlap
    rectangle. Without this, every rebrush of a cross-padding run
    collapses to agreement=-1 → Accept button disappears.
    """
    try:
        if clf_path is None:
            clf_path = find_classified(
                fire, [fire.cache_dir,
                       os.path.dirname(fire.crop_bin or '')])
        hint_path = fire.hint_bin
        if not clf_path or not hint_path:
            sys.stderr.write("[agreement] not computable (bail 1)\n")
            return -1.0
        if not os.path.isfile(clf_path) or not os.path.isfile(hint_path):
            sys.stderr.write("[agreement] not computable (bail 2)\n")
            return -1.0

        ds_clf = ds_hint = None
        try:
            ds_clf = gdal.Open(clf_path, gdal.GA_ReadOnly)
            ds_hint = gdal.Open(hint_path, gdal.GA_ReadOnly)
            if ds_clf is None or ds_hint is None:
                sys.stderr.write("[agreement] not computable (bail 3)\n")
                return -1.0

            clf = ds_clf.GetRasterBand(1).ReadAsArray()
            hint = ds_hint.GetRasterBand(1).ReadAsArray()
            clf_gt = ds_clf.GetGeoTransform()
            hint_gt = ds_hint.GetGeoTransform()
        finally:
            ds_clf = ds_hint = None

        if clf.shape != hint.shape:
            # Cross-extent: crops are from the same source raster so
            # pixel sizes must match — only the origin / extent differ.
            # Refuse if pixel sizes don't line up (can't align sensibly).
            if not (clf_gt and hint_gt
                    and abs(clf_gt[1] - hint_gt[1]) < 1e-6
                    and abs(clf_gt[5] - hint_gt[5]) < 1e-6):
                sys.stderr.write("[agreement] not computable (bail 4)\n")
                return -1.0
            # Offset of clf's origin expressed in hint's pixel frame.
            off_x = round((clf_gt[0] - hint_gt[0]) / hint_gt[1])
            off_y = round((clf_gt[3] - hint_gt[3]) / hint_gt[5])
            ch, cw = clf.shape
            hh, hw = hint.shape
            # Intersection rectangle in hint's pixel coordinates.
            com_x0 = max(0, off_x)
            com_y0 = max(0, off_y)
            com_x1 = min(hw, off_x + cw)
            com_y1 = min(hh, off_y + ch)
            if com_x1 <= com_x0 or com_y1 <= com_y0:
                sys.stderr.write("[agreement] not computable (bail 5)\n")
                return -1.0
            hint = hint[com_y0:com_y1, com_x0:com_x1]
            clf = clf[
                com_y0 - off_y:com_y1 - off_y,
                com_x0 - off_x:com_x1 - off_x,
            ]
            if clf.shape != hint.shape:
                sys.stderr.write("[agreement] not computable (bail 6)\n")
                return -1.0

        clf_mask = clf > 0
        hint_mask = hint > 0
        union = np.sum(clf_mask | hint_mask)
        if union == 0:
            return 0.0
        intersection = np.sum(clf_mask & hint_mask)
        return round(float(intersection / union) * 100, 1)
    except Exception as exc:
        sys.stderr.write(
            f'[agreement] WARNING: Failed to compute agreement: {exc}\n')
        sys.stderr.write("[agreement] not computable (bail 7)\n")
        return -1.0
