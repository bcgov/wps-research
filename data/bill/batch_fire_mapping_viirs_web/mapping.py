"""Mapping result helpers — overlay PNG generation, area + agreement metrics.

These are pure-ish functions that read from raster files and write
preview PNGs into ``fire.cache_dir/previews/``. Stateful access to the
shared ``state`` (only ``state.raster_gt`` for pixel area) is wired by
:func:`init` at server boot.
"""

import os
import sys

import numpy as np
from osgeo import gdal

from .state import AppState, FireInfo

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
        clf_path = os.path.join(
            fire.cache_dir,
            f'{fire.fire_numbe}_crop.bin_classified.bin')
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

        out_path = os.path.join(fire.cache_dir, 'previews', f'{out_name}.png')
        imsave(out_path, np.clip(result, 0, 1))

        if (out_name not in fire.available_views
                and not out_name.startswith('serial_')):
            fire.available_views.append(out_name)
    except Exception as exc:
        import traceback
        sys.stderr.write(
            f'[overlay] WARNING: Failed to generate {out_name} '
            f'overlay: {exc}\n{traceback.format_exc()}')


def _generate_result_preview(fire: 'FireInfo'):
    """Generate pixel-aligned overlay previews after mapping."""
    clf_path = os.path.join(
        fire.cache_dir,
        f'{fire.fire_numbe}_crop.bin_classified.bin')
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
            clf_path = os.path.join(
                fire.cache_dir,
                f'{fire.fire_numbe}_crop.bin_classified.bin')
        hint_path = fire.hint_bin
        if not clf_path or not hint_path:
            return -1.0
        if not os.path.isfile(clf_path) or not os.path.isfile(hint_path):
            return -1.0

        ds_clf = ds_hint = None
        try:
            ds_clf = gdal.Open(clf_path, gdal.GA_ReadOnly)
            ds_hint = gdal.Open(hint_path, gdal.GA_ReadOnly)
            if ds_clf is None or ds_hint is None:
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
                return -1.0
            hint = hint[com_y0:com_y1, com_x0:com_x1]
            clf = clf[
                com_y0 - off_y:com_y1 - off_y,
                com_x0 - off_x:com_x1 - off_x,
            ]
            if clf.shape != hint.shape:
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
        return -1.0
