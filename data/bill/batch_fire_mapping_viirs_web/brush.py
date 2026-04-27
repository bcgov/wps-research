"""class_brush rebrush helpers + figure rendering.

Wraps the C++ ``class_brush.exe`` binary with cleanup of intermediate
files, plus matplotlib renderers for the comparison and brush
comparison PNGs that the rebrush flow writes into the cache.

Stateful access is limited: only ``state.project_root`` is read (to
locate ``class_brush.exe``) — wired by :func:`init` at server boot.

Subprocess registry (``_rebrush_procs`` / ``_rebrush_procs_lock``) is
owned by ``app.py`` and passed in here, so the same dict can be shared
with the cancel handler and the cache-retention pin logic without an
import cycle.
"""

import glob
import os
import shutil
import subprocess
import sys
import threading

import numpy as np
from osgeo import gdal

from .state import AppState, FireInfo

state: AppState = None
_rebrush_procs: dict = None
_rebrush_procs_lock: threading.Lock = None


def init(app_state: AppState,
         rebrush_procs: dict, rebrush_procs_lock: threading.Lock):
    global state, _rebrush_procs, _rebrush_procs_lock
    state = app_state
    _rebrush_procs = rebrush_procs
    _rebrush_procs_lock = rebrush_procs_lock


def _class_brush_exe() -> str:
    """Locate the compiled class_brush.exe relative to project_root."""
    # state.project_root is wps-research/data/bill; class_brush.exe lives at
    # wps-research/cpp/class_brush.exe
    root = state.project_root
    repo_root = os.path.dirname(os.path.dirname(root))
    return os.path.join(repo_root, 'cpp', 'class_brush.exe')


def _read_envi_mask(path: str) -> np.ndarray:
    """Read an ENVI .bin classification as a boolean mask (first band)."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'Cannot open {path}')
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return (arr > 0) & np.isfinite(arr)


def _write_envi_mask_like(mask: np.ndarray, out_path: str,
                           ref_path: str) -> None:
    """Write a boolean mask as float32 ENVI .bin, copying the reference
    file's .hdr geometry so downstream GDAL readers see identical
    dimensions/projection."""
    mask.astype(np.float32).tofile(out_path)
    # Copy the sibling .hdr from ref_path (handles both foo.hdr and
    # foo.bin.hdr naming conventions).
    ref_hdr = os.path.splitext(ref_path)[0] + '.hdr'
    if not os.path.isfile(ref_hdr):
        ref_hdr = ref_path + '.hdr'
    if os.path.isfile(ref_hdr):
        out_hdr = os.path.splitext(out_path)[0] + '.hdr'
        shutil.copy2(ref_hdr, out_hdr)



def _run_class_brush_only(clf_path: str, brush_size: int,
                          point_threshold: int,
                          all_segments: bool,
                          fire_numbe: str | None = None
                          ) -> tuple[np.ndarray | None, bool]:
    """Shell to class_brush.exe on an existing classification.

    Returns ``(brushed_mask, cancelled)``:
      - brushed_mask: boolean mask, or None if exe unavailable / produced
        nothing / was cancelled.
      - cancelled:    True iff a cancel signal terminated the subprocess.

    When ``fire_numbe`` is provided, the subprocess handle is registered
    in ``_rebrush_procs[fire_numbe]`` so ``handle_api_rebrush_cancel``
    can terminate it. Intermediate files are always cleaned up.
    """
    brush_exe = _class_brush_exe()
    if not os.path.isfile(brush_exe):
        return None, False

    cmd = [brush_exe]
    if all_segments:
        cmd.append('--all_segments')
    cmd += [clf_path, str(int(brush_size)), str(int(point_threshold))]

    proc = subprocess.Popen(
        cmd, cwd=os.path.dirname(clf_path),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if fire_numbe is not None:
        with _rebrush_procs_lock:
            _rebrush_procs[fire_numbe] = proc
    try:
        # communicate() drains stdout+stderr concurrently on internal
        # threads. Using proc.wait() here would deadlock once the OS
        # pipe buffer fills up (class_brush.exe emits one line per
        # component — easily >64KB for complex fires).
        proc.communicate()
        rc = proc.returncode
    finally:
        if fire_numbe is not None:
            with _rebrush_procs_lock:
                _rebrush_procs.pop(fire_numbe, None)

    # rc < 0 on POSIX when terminated by signal (e.g. SIGTERM = -15).
    cancelled = rc is not None and rc < 0

    comp_files = sorted(glob.glob(clf_path + '_comp_*.bin'))
    brushed = None
    if comp_files:
        if all_segments:
            for cf in comp_files:
                try:
                    m = _read_envi_mask(cf)
                    brushed = m if brushed is None else (brushed | m)
                except Exception:
                    continue
        else:
            largest_count = -1
            for cf in comp_files:
                try:
                    m = _read_envi_mask(cf)
                    c = int(m.sum())
                    if c > largest_count:
                        largest_count = c
                        brushed = m
                except Exception:
                    continue

    # Clean up component files + their headers
    for cf in comp_files:
        for p in (cf, os.path.splitext(cf)[0] + '.hdr'):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

    # Clean up C++ stage intermediaries
    for suffix in ('_flood4.bin', '_flood4.hdr',
                   '_flood4.bin_link.bin', '_flood4.bin_link.hdr',
                   '_flood4.bin_link.bin_recode.bin',
                   '_flood4.bin_link.bin_recode.hdr',
                   '_flood4.bin_link.bin_recode.bin_wheel.bin',
                   '_flood4.bin_link.bin_recode.bin_wheel.hdr'):
        p = clf_path + suffix
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    # Discard any partial output if the subprocess was cancelled.
    if cancelled:
        return None, True
    return brushed, False


def _align_mask_to_crop_frame(mask_path: str, crop_bin_path: str,
                              target_h: int, target_w: int):
    """Align a raster mask to the current crop's geographic frame and
    resize to ``(target_h, target_w)``.

    When the mask and the crop share pixel size but differ in extent
    (e.g. the classification was produced with a different padding
    epoch than the current crop), this places the mask in the crop's
    coordinate frame at the correct pixel offset computed from the
    geotransforms — the same technique ``_overlay_mask_on_post`` uses
    for the interactive web UI. That path is why the main-page overlay
    looks right; the PDF report used a naive stretch which visibly
    mis-scaled the overlay. Falls back to a naive zoom when
    geotransforms aren't available or pixel sizes don't match.

    Returns a boolean ``ndarray`` of shape ``(target_h, target_w)``,
    or ``None`` if the mask can't be loaded.
    """
    if not mask_path or not os.path.isfile(mask_path):
        return None
    try:
        from scipy.ndimage import zoom as scipy_zoom
        ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
        if ds is None:
            return None
        arr = ds.GetRasterBand(1).ReadAsArray()
        old_gt = ds.GetGeoTransform()
        ds = None
        if arr is None:
            return None

        ah, aw = arr.shape
        aligned = False

        if crop_bin_path and os.path.isfile(crop_bin_path):
            try:
                ds_crop = gdal.Open(crop_bin_path, gdal.GA_ReadOnly)
                if ds_crop is not None:
                    new_gt = ds_crop.GetGeoTransform()
                    new_w = ds_crop.RasterXSize
                    new_h = ds_crop.RasterYSize
                    ds_crop = None

                    if (old_gt and new_gt
                            and abs(old_gt[1] - new_gt[1]) < 1e-6
                            and abs(old_gt[5] - new_gt[5]) < 1e-6):
                        off_x = round(
                            (old_gt[0] - new_gt[0]) / new_gt[1])
                        off_y = round(
                            (old_gt[3] - new_gt[3]) / new_gt[5])
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
                        arr = arr_aligned
                        aligned = True
            except Exception:
                pass

        ah2, aw2 = arr.shape
        if (ah2, aw2) != (target_h, target_w):
            arr = scipy_zoom(
                arr.astype(np.uint8),
                (target_h / ah2, target_w / aw2), order=0)

        return arr.astype(bool)
    except Exception as exc:
        sys.stderr.write(
            f'[align_mask] WARNING: {mask_path}: {exc}\n')
        return None


def _render_comparison_png(fire: 'FireInfo', classified_path: str,
                           out_path: str) -> bool:
    """Regenerate the perimeter-style comparison PNG.

    Mirrors fire_mapping_cli.make_comparison_figure so callers (rebrush)
    can keep ``<fire>_comparison.png`` in sync with the current
    ``classified.bin`` without re-running the full mapping pipeline.
    Background is taken from ``cache_dir/previews/post.png``.

    Returns True on success, False if inputs are missing or render fails.
    """
    try:
        post_path = os.path.join(fire.cache_dir, 'previews', 'post.png')
        if not (os.path.isfile(post_path)
                and os.path.isfile(classified_path)):
            return False

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
        from matplotlib.lines import Line2D
        from scipy.ndimage import zoom as scipy_zoom, binary_dilation

        bg = imread(post_path)
        if bg.ndim == 2:
            bg = np.stack([bg] * 3, axis=2)
        bh, bw = bg.shape[:2]

        def _contour_rgba(mask, rgb):
            dil = binary_dilation(mask)
            boundary = dil & (~mask)
            rgba = np.zeros((bh, bw, 4), dtype=np.float32)
            rgba[..., 0] = rgb[0]
            rgba[..., 1] = rgb[1]
            rgba[..., 2] = rgb[2]
            rgba[..., 3] = boundary.astype(np.float32)
            return rgba

        # Use geotransform-aware placement so masks produced under a
        # different crop/padding epoch still land at the right position
        # relative to the current post-fire background.
        crop_bin = fire.crop_bin or ''
        clf_mask = _align_mask_to_crop_frame(
            classified_path, crop_bin, bh, bw)
        hint_path = fire.hint_bin or fire.viirs_bin or ''
        hint_mask = _align_mask_to_crop_frame(
            hint_path, crop_bin, bh, bw)
        perim_mask = _align_mask_to_crop_frame(
            fire.perim_bin, crop_bin, bh, bw)

        # If perimeter == hint file, drop the separate cyan outline.
        if (fire.perim_bin and hint_path
                and os.path.abspath(fire.perim_bin)
                == os.path.abspath(hint_path)):
            perim_mask = None

        pt = (fire.perimeter_type or '').lower()
        if pt == 'viirs':
            hint_label = 'VIIRS hint'
        elif pt in ('polygon_perimeter', 'polygon', 'traditional'):
            hint_label = 'Traditional perimeter (hint)'
        else:
            hint_label = 'Hint'

        def _iou(a, b):
            if a is None or b is None:
                return None
            inter = int(np.sum(a & b))
            union = int(np.sum(a | b))
            return inter / union if union > 0 else 0.0

        def _acc(a, b):
            if a is None or b is None:
                return None
            return float(np.sum(a == b)) / a.size if a.size else 0.0

        iou_cv = _iou(clf_mask, hint_mask)
        acc_cv = _acc(clf_mask, hint_mask)
        parts = []
        if iou_cv is not None:
            parts.append(f'IoU(ours/{hint_label})={iou_cv:.3f}')
        if acc_cv is not None:
            parts.append(f'Acc(ours/{hint_label})={acc_cv:.3f}')
        metrics = '  '.join(parts)
        if perim_mask is not None and clf_mask is not None:
            iou_cp = _iou(clf_mask, perim_mask)
            iou_vp = _iou(hint_mask, perim_mask) if hint_mask is not None else None
            extras = []
            if iou_cp is not None:
                extras.append(f'IoU(ours/perim)={iou_cp:.3f}')
            if iou_vp is not None:
                extras.append(f'IoU({hint_label}/perim)={iou_vp:.3f}')
            if extras:
                metrics += '\n' + '  '.join(extras)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(bg, interpolation='nearest', origin='upper')
        if clf_mask is not None:
            ax.imshow(_contour_rgba(clf_mask, (1.0, 0.0, 0.0)),
                      interpolation='nearest', origin='upper')
        if hint_mask is not None:
            ax.imshow(_contour_rgba(hint_mask, (0.0, 1.0, 0.0)),
                      interpolation='nearest', origin='upper')
        if perim_mask is not None:
            ax.imshow(_contour_rgba(perim_mask, (0.0, 1.0, 1.0)),
                      interpolation='nearest', origin='upper')

        ax.set_xlim(0, bw)
        ax.set_ylim(bh, 0)
        ax.set_xlabel('Column (px)', fontsize=8)
        ax.set_ylabel('Row (px)', fontsize=8)
        ax.tick_params(labelsize=7)

        handles = []
        if clf_mask is not None:
            handles.append(Line2D([0], [0], color='red', linewidth=2,
                                  label='Our mapping'))
        if hint_mask is not None:
            handles.append(Line2D([0], [0], color='lime', linewidth=2,
                                  label=hint_label))
        if perim_mask is not None:
            handles.append(Line2D([0], [0], color='cyan', linewidth=2,
                                  label='Traditional perimeter'))
        if handles:
            ax.legend(handles=handles, loc='lower right', fontsize=9,
                      framealpha=0.7, edgecolor='white')

        title = f'Fire: {fire.fire_numbe}'
        if fire.acc_start or fire.acc_end:
            title += (f'   |   Start: {fire.acc_start}'
                      f'   |   End: {fire.acc_end}')
        if metrics:
            title += f'\n{metrics}'
        ax.set_title(title, fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as exc:
        sys.stderr.write(
            f'[comparison] WARNING: regen failed: {exc}\n')
        sys.stderr.flush()
        return False


def _render_ml_classification_png(fire_numbe: str, post_path: str,
                                  classified_path: str, out_path: str,
                                  subtitle: str = '',
                                  hint_path: str = '',
                                  hint_label: str = 'Hint',
                                  crop_bin: str = '') -> bool:
    """Render an ML-classification view: post-fire background with the
    burned mask tinted red (filled). Optionally overlays a lime contour
    for the hint perimeter (VIIRS or traditional) so the reader can see
    the ground truth alongside the ML output.

    Ephemeral — used only for the PDF report's Pass-1 hero pages, never
    saved to a fire's permanent directory. ``subtitle`` is appended to
    the figure's title.

    Returns True on success.
    """
    try:
        if not (os.path.isfile(post_path)
                and os.path.isfile(classified_path)):
            return False

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
        from matplotlib.lines import Line2D
        from scipy.ndimage import zoom as scipy_zoom, binary_dilation

        bg = imread(post_path)
        if bg.ndim == 2:
            bg = np.stack([bg] * 3, axis=2)
        bh, bw = bg.shape[:2]

        # Geotransform-aware placement when crop_bin is known; falls
        # back to naive resize otherwise (preserves prior behavior for
        # any external callers that don't pass crop_bin).
        if crop_bin:
            aligned = _align_mask_to_crop_frame(
                classified_path, crop_bin, bh, bw)
            if aligned is None:
                return False
            mask = aligned
        else:
            ds = gdal.Open(classified_path, gdal.GA_ReadOnly)
            if ds is None:
                return False
            arr = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            ah, aw = arr.shape
            if (ah, aw) != (bh, bw):
                arr = scipy_zoom(arr.astype(np.uint8),
                                 (bh / ah, bw / aw), order=0)
            mask = arr > 0

        rgb = bg[:, :, :3].astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        r, g, b = 0.9, 0.1, 0.0
        result = rgb.copy()
        result[mask, 0] = np.clip(rgb[mask, 0] * 0.3 + r * 0.7, 0, 1)
        result[mask, 1] = np.clip(rgb[mask, 1] * 0.3 + g * 0.7, 0, 1)
        result[mask, 2] = np.clip(rgb[mask, 2] * 0.3 + b * 0.7, 0, 1)

        # Optional hint contour (lime). Same contour technique as
        # _render_comparison_png so outline style matches the detail
        # pages. Silently skipped if the hint raster is missing.
        hint_rgba = None
        if hint_path and os.path.isfile(hint_path):
            try:
                if crop_bin:
                    hmask = _align_mask_to_crop_frame(
                        hint_path, crop_bin, bh, bw)
                else:
                    ds_h = gdal.Open(hint_path, gdal.GA_ReadOnly)
                    hmask = None
                    if ds_h is not None:
                        harr = ds_h.GetRasterBand(1).ReadAsArray()
                        ds_h = None
                        hh, hw = harr.shape
                        if (hh, hw) != (bh, bw):
                            harr = scipy_zoom(
                                harr.astype(np.uint8),
                                (bh / hh, bw / hw), order=0)
                        hmask = harr > 0
                if hmask is not None and hmask.any():
                    dil = binary_dilation(hmask)
                    boundary = dil & (~hmask)
                    hint_rgba = np.zeros((bh, bw, 4), dtype=np.float32)
                    hint_rgba[..., 1] = 1.0  # lime
                    hint_rgba[..., 3] = boundary.astype(np.float32)
            except Exception:
                hint_rgba = None

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(result, interpolation='nearest', origin='upper')
        if hint_rgba is not None:
            ax.imshow(hint_rgba, interpolation='nearest', origin='upper')
        ax.set_xlim(0, bw)
        ax.set_ylim(bh, 0)
        ax.set_xlabel('Column (px)', fontsize=8)
        ax.set_ylabel('Row (px)', fontsize=8)
        ax.tick_params(labelsize=7)

        handles = [Line2D([0], [0], color='red', linewidth=6,
                          alpha=0.7, label='ML classification')]
        if hint_rgba is not None:
            handles.append(Line2D([0], [0], color='lime', linewidth=2,
                                  label=hint_label))
        ax.legend(handles=handles, loc='lower right', fontsize=9,
                  framealpha=0.7, edgecolor='white')

        title = f'Fire: {fire_numbe}  —  ML classification'
        if subtitle:
            title += f'\n{subtitle}'
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as exc:
        sys.stderr.write(
            f'[ml_overlay] WARNING: render failed: {exc}\n')
        sys.stderr.flush()
        return False


def _render_brush_comparison_png(raw: np.ndarray, brushed: np.ndarray | None,
                                 bg_path: str, out_path: str,
                                 title: str) -> None:
    """Draw a two-panel (raw vs brushed) contour figure on bg_path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from scipy.ndimage import zoom as scipy_zoom

    bg = imread(bg_path)
    bh, bw = bg.shape[:2]

    def _resize(mask):
        mh, mw = mask.shape
        if (mh, mw) == (bh, bw):
            return mask.astype(bool)
        zy = bh / mh
        zx = bw / mw
        return scipy_zoom(mask.astype(np.uint8),
                          (zy, zx), order=0).astype(bool)

    def _contour_rgba(mask_bg):
        from scipy.ndimage import binary_dilation
        mask_bg = mask_bg.astype(bool)
        dil = binary_dilation(mask_bg)
        boundary = dil & (~mask_bg)
        rgba = np.zeros((bh, bw, 4), dtype=np.float32)
        rgba[..., 0] = 1.0  # red
        rgba[..., 3] = boundary.astype(np.float32)
        return rgba

    raw_bg = _resize(raw)
    after_mask = _resize(brushed) if brushed is not None else raw_bg
    after_title = ('After class_brush\n(brushed)'
                   if brushed is not None
                   else 'After class_brush\n(no output)')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title, fontsize=10, fontweight='bold')
    for ax, m, t in [
        (axes[0], raw_bg,     'Before class_brush\n(raw classification)'),
        (axes[1], after_mask, after_title),
    ]:
        ax.imshow(bg, interpolation='nearest', origin='upper')
        ax.imshow(_contour_rgba(m), interpolation='nearest', origin='upper')
        ax.set_title(t, fontsize=9)
        ax.set_xlim(0, bw)
        ax.set_ylim(bh, 0)
        ax.set_xlabel('Column (px)', fontsize=8)
        ax.set_ylabel('Row (px)', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

