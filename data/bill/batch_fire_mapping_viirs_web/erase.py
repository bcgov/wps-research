"""Manual eraser: knock pixels out of a classification mask by hand.

The classifier gets the shape roughly right and then an analyst spends
their time removing things it should not have caught -- a neighbouring
burn, a shadow, a bright rock. Brushing cleans up speckle but cannot
know which blob is wrong; only a person can.

The client paints locally for immediate feedback and posts the boxes
here, where they are applied to the canonical mask so that agreement,
area, the overlay, polygonisation and export all reflect the edit. The
pre-edit mask is kept beside it so the whole session can be undone.
"""

import os
import shutil
import sys

from .state import AppState

state: AppState = None


def init(app_state: AppState):
    global state
    state = app_state


def ensure_geo(path: str, ref_path: str, log=None) -> bool:
    """Make sure *path* carries the same map info as *ref_path*.

    ENVI keeps georeferencing in the sidecar .hdr, and several places
    write a mask as raw floats plus a copied header. If that copy is
    missed -- or the header came from a raster on a different grid --
    the mask still opens and still has the right dimensions, so nothing
    complains until it is drawn against the imagery after a restart and
    lands in the wrong place.

    Cheap to check on every write, and it repairs rather than reports.
    """
    try:
        from osgeo import gdal
        if not path or not os.path.isfile(path):
            return False
        ds = gdal.Open(path, gdal.GA_Update)
        if ds is None:
            return False
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        need = (not proj) or gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if not need:
            ds = None
            return False
        ref = gdal.Open(ref_path, gdal.GA_ReadOnly) if ref_path else None
        if ref is None:
            ds = None
            return False
        if (ref.RasterXSize, ref.RasterYSize) != (ds.RasterXSize,
                                                  ds.RasterYSize):
            ds = None
            ref = None
            return False
        ds.SetGeoTransform(ref.GetGeoTransform())
        rp = ref.GetProjection()
        if rp:
            ds.SetProjection(rp)
        ds.FlushCache()
        ds = None
        ref = None
        msg = (f'  Restored map info on {os.path.basename(path)} from '
               f'{os.path.basename(ref_path)}')
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
        return True
    except Exception as exc:
        sys.stderr.write(f'[geo] check failed for {path}: {exc}\n')
        return False


def _backup_path(clf_path: str) -> str:
    """Where the pre-erase mask is kept.

    Distinct from ``_raw.bin``, which is the pre-BRUSH mask: reverting
    an erase session must restore the brushed result, not an unbrushed
    one, so the two backups cannot share a name.
    """
    return os.path.splitext(clf_path)[0] + '_preerase.bin'


def _hdr_for(path: str) -> str:
    h = os.path.splitext(path)[0] + '.hdr'
    return h if os.path.isfile(h) else path + '.hdr'


def ensure_backup(clf_path: str, log=None) -> str:
    """Snapshot the mask before the first edit of a session.

    Taken once: a later call must not overwrite it, or Revert would
    only undo the most recent stroke rather than the whole session.
    """
    bak = _backup_path(clf_path)
    if os.path.isfile(bak):
        return bak
    try:
        shutil.copy2(clf_path, bak)
        src_hdr = _hdr_for(clf_path)
        if os.path.isfile(src_hdr):
            shutil.copy2(src_hdr, os.path.splitext(bak)[0] + '.hdr')
        msg = f'  Eraser: kept a pre-edit copy as {os.path.basename(bak)}'
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
    except OSError as exc:
        sys.stderr.write(f'[erase] could not back up {clf_path}: {exc}\n')
    return bak


def perimeter_mask_png(fire, out_path: str, width=None,
                       height=None) -> str:
    """Render the BCWS perimeter as a plain 8-bit PNG (255 = inside).

    The client needs to know which pixels are protected so its live
    preview matches what the server will do. It cannot read ENVI, and
    the hint PREVIEW is a green overlay on post-fire imagery -- fine to
    look at, useless to threshold. A clean binary PNG is unambiguous
    and tiny.
    """
    from osgeo import gdal
    import numpy as np
    from matplotlib.image import imsave

    from .prepare import build_bcws_hint_for_fire
    mask_path, err = build_bcws_hint_for_fire(fire)
    if not mask_path or not os.path.isfile(mask_path):
        raise RuntimeError(err or 'no BCWS perimeter for this AOI')

    ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError('cannot open the perimeter mask')
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    m = (np.nan_to_num(arr) > 0)

    # Downsample by decimation to the preview size when asked. Nearest
    # is right here: a perimeter is a hard boundary, and interpolating
    # would invent half-protected pixels.
    if width and height and (m.shape[1] != width or m.shape[0] != height):
        ys = (np.linspace(0, m.shape[0] - 1, int(height))
              ).astype('int32')
        xs = (np.linspace(0, m.shape[1] - 1, int(width))
              ).astype('int32')
        m = m[ys][:, xs]

    rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype='uint8')
    rgb[m] = 255
    tmp = out_path + '.tmp.png'
    imsave(tmp, rgb)
    os.replace(tmp, out_path)
    return out_path


def apply_erase(fire, clf_path: str, boxes, size: int, log=None,
                outside_bcws_only: bool = False) -> dict:
    """Zero an N x N box in the mask around each (x, y) in *boxes*.

    Coordinates are IMAGE pixels in the mask's own grid, so the result
    is independent of zoom, pan and preview downsampling -- the client
    converts before sending. *size* is likewise in image pixels, so a
    given eraser setting removes the same ground area at any zoom.

    Returns counts so the caller can report what changed.
    """
    import numpy as np
    from osgeo import gdal

    if not clf_path or not os.path.isfile(clf_path):
        return {'ok': False, 'error': 'no classification to edit'}
    if not boxes:
        return {'ok': True, 'erased': 0, 'remaining': -1}

    try:
        size = max(1, int(size))
    except (TypeError, ValueError):
        size = 11

    ensure_backup(clf_path, log=log)

    ds = gdal.Open(clf_path, gdal.GA_Update)
    if ds is None:
        ds = gdal.Open(clf_path, gdal.GA_ReadOnly)
        if ds is None:
            return {'ok': False,
                    'error': f'cannot open {os.path.basename(clf_path)}'}
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    if arr is None:
        ds = None
        return {'ok': False, 'error': 'mask has no data'}

    h, w = arr.shape[0], arr.shape[1]
    before = int(np.count_nonzero(np.nan_to_num(arr) > 0))

    # Pixels the perimeter protects, when the caller asked for it.
    # Loaded once per request rather than per stroke point.
    protect = None
    if outside_bcws_only:
        try:
            from .prepare import build_bcws_hint_for_fire
            mp, merr = build_bcws_hint_for_fire(fire)
            if mp and os.path.isfile(mp):
                pds = gdal.Open(mp, gdal.GA_ReadOnly)
                parr = pds.GetRasterBand(1).ReadAsArray() if pds else None
                pds = None
                if parr is not None and parr.shape == arr.shape:
                    protect = np.nan_to_num(parr) > 0
                elif parr is not None:
                    if log:
                        log(f'  Eraser: perimeter is {parr.shape} but '
                            f'the mask is {arr.shape}; erasing '
                            f'everywhere instead')
            elif log:
                log(f'  Eraser: no BCWS perimeter ({merr or "none"}); '
                    f'erasing everywhere')
        except Exception as exc:
            if log:
                log(f'  Eraser: could not load the perimeter ({exc}); '
                    f'erasing everywhere')

    half = size // 2
    for pt in boxes:
        try:
            cx, cy = int(round(float(pt[0]))), int(round(float(pt[1])))
        except (TypeError, ValueError, IndexError):
            continue
        # Clamp rather than skip: a stroke that runs off the edge
        # should still erase the part that is on the image.
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half + 1)
        y1 = min(h, cy + half + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        if protect is None:
            arr[y0:y1, x0:x1] = 0.0
        else:
            # Clear only the part of the box outside the perimeter, so
            # a stroke that overlaps the official boundary trims the
            # outside without eating into what BCWS reported.
            sub = arr[y0:y1, x0:x1]
            keep = protect[y0:y1, x0:x1]
            arr[y0:y1, x0:x1] = np.where(keep, sub, 0.0)

    after = int(np.count_nonzero(np.nan_to_num(arr) > 0))
    band.WriteArray(arr.astype('float32'))
    band.FlushCache()
    band = None
    ds = None

    ensure_geo(clf_path, getattr(fire, 'crop_bin', ''), log=log)
    msg = (f'  Eraser: {len(boxes)} stroke point(s) at {size}x{size} px '
           f'-> {before - after:,} pixel(s) removed, {after:,} remain')
    sys.stderr.write(msg + '\n')
    if log:
        log(msg)
    return {'ok': True, 'erased': before - after, 'remaining': after,
            'before': before}


def revert(fire, clf_path: str, log=None) -> dict:
    """Restore the mask saved before the first erase of the session."""
    bak = _backup_path(clf_path)
    if not os.path.isfile(bak):
        return {'ok': False, 'error': 'nothing to revert to'}
    try:
        shutil.copy2(bak, clf_path)
        bak_hdr = _hdr_for(bak)
        if os.path.isfile(bak_hdr):
            shutil.copy2(bak_hdr, os.path.splitext(clf_path)[0] + '.hdr')
        # Remove the backup so the NEXT session snapshots afresh.
        # Keeping it would make a later Revert jump back past edits the
        # user had already accepted by starting a new session.
        for p in (bak, os.path.splitext(bak)[0] + '.hdr'):
            try:
                os.remove(p)
            except OSError:
                pass
        ensure_geo(clf_path, getattr(fire, 'crop_bin', ''), log=log)
        msg = '  Eraser: reverted to the pre-edit classification'
        sys.stderr.write(msg + '\n')
        if log:
            log(msg)
        return {'ok': True}
    except OSError as exc:
        return {'ok': False, 'error': str(exc)}


def refresh_after_edit(fire, clf_path: str, log=None) -> dict:
    """Re-render and re-score after the mask changed.

    The same steps a finished run performs, so an edited mask is
    indistinguishable downstream from one the classifier produced:
    the overlay the pane shows, the agreement and area in the header,
    and the entry the results gallery reads.
    """
    from .mapping import (_compute_agreement, _compute_ml_area,
                          _overlay_mask_on_post, copy_preview_geo)

    out = {}
    try:
        # Re-render the overlay for the run this mask BELONGS TO, not
        # serial_1.
        #
        # Runs are numbered now, so a fire can be showing serial_3 while
        # this wrote serial_1. The edit reached the raster, the gallery
        # and the pane kept fetching the old serial_3 image, and the
        # change appeared to revert the moment the pane reloaded.
        run_name = 'serial_1'
        for r in (getattr(fire, 'serial_results', None) or []):
            if r.get('classified') == clf_path and r.get('run_id'):
                run_name = f"serial_{int(r['run_id'])}"
                break
        _overlay_mask_on_post(fire, clf_path, run_name, (0.9, 0.1, 0.0))
        prev = os.path.join(fire.cache_dir, 'previews')
        s1 = os.path.join(prev, f'{run_name}.png')
        res = os.path.join(prev, 'result.png')
        if os.path.isfile(s1):
            shutil.copy2(s1, res)
            try:
                copy_preview_geo(fire.cache_dir, run_name, 'result')
            except Exception:
                pass
            if 'result' not in fire.available_views:
                fire.available_views.append('result')
            sys.stderr.write(
                f'[erase] refreshed {run_name}.png and result.png\n')
    except Exception as exc:
        sys.stderr.write(f'[erase] overlay refresh failed: {exc}\n')

    try:
        agr = _compute_agreement(fire)
        area = _compute_ml_area(fire, clf_path)
        with state.lock:
            fire.agreement_pct = agr
            fire.ml_area_ha = area
            for r in (fire.serial_results or []):
                # Keep the gallery entry in step; it is what Accept
                # reads, so a stale area there would be promoted.
                r['agreement_pct'] = agr
                r['ml_area_ha'] = area
        out['agreement_pct'] = agr
        out['ml_area_ha'] = area
        if log:
            log(f'  Eraser: agreement {agr}%, ML area {area} ha')
    except Exception as exc:
        sys.stderr.write(f'[erase] rescore failed: {exc}\n')

    try:
        from .persistence import _save_fire_state
        _save_fire_state()
    except Exception:
        pass
    return out


def prebrush_path(clf_path: str) -> str:
    """Where the pre-BRUSH mask lives for a given classification.

    Written by every brushing path as ``<name>_raw.bin`` before the
    brushed result overwrites the canonical file.
    """
    return os.path.splitext(clf_path)[0] + '_raw.bin'


def _verify_overlay_differs(fire, out_name: str, log=None) -> bool:
    """True when previews/<out_name>.png actually differs from the base.

    _overlay_mask_on_post() draws the mask onto post.png. If the mask it
    was given was empty -- or if it silently fell back -- the output is
    a copy of the base imagery, which renders as "the layer shows the
    wrong picture" rather than as an error. Comparing the bytes is
    cheap and catches exactly that.
    """
    try:
        prev = os.path.join(fire.cache_dir, 'previews')
        out = os.path.join(prev, f'{out_name}.png')
        if not os.path.isfile(out):
            return False
        import hashlib

        def _sig(p_):
            try:
                with open(p_, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
            except OSError:
                return None
        sig = _sig(out)
        # Only the BASE imagery counts as "drew nothing". Being
        # identical to result.png is legitimate -- it means brushing
        # changed nothing, which is a fact worth showing, not a fault.
        for base in ('post.png', 'pre.png'):
            if sig and sig == _sig(os.path.join(prev, base)):
                msg = (f'  WARNING: {out_name}.png is byte-identical to '
                       f'{base} -- the overlay drew no mask, so the '
                       f'layer would show plain imagery. Removing it.')
                sys.stderr.write(msg + '\n')
                if log:
                    log(msg)
                try:
                    os.remove(out)
                except OSError:
                    pass
                return False
        return True
    except Exception:
        return True


def render_prebrush_overlay(fire, clf_path: str = None,
                            log=None) -> bool:
    """Render 'ML classification - before brushing' as a map layer.

    The old brush comparison was a two-panel FIGURE: not on the AOI
    grid, so it could not be flickered against the result or read
    against the imagery. Rendering the pre-brush mask as an ordinary
    overlay on the same grid makes it a peer of every other layer --
    it can sit in either split pane, flicker against the brushed
    result, and line up with the base imagery.

    Returns True when the layer exists afterwards.
    """
    from .mapping import _overlay_mask_on_post

    try:
        clf = clf_path or active_classified(fire)
        if not clf:
            return False
        raw = prebrush_path(clf)
        if not os.path.isfile(raw):
            # The serial/CLI path names its pre-brush sibling after the
            # image it processed rather than after the classification,
            # so fall back to any *_raw.bin on the same grid before
            # concluding there is nothing to show.
            import glob as _glob
            cands = sorted(
                _glob.glob(os.path.join(fire.cache_dir, '*_raw.bin')),
                key=lambda p_: os.path.getmtime(p_), reverse=True)
            raw = cands[0] if cands else ''
        if not raw or not os.path.isfile(raw):
            # No pre-brush mask on disk means brushing never modified
            # this result -- either it was skipped, or class_brush
            # failed (a stale binary rejecting its arguments does
            # exactly this). "Before brushing" is then simply the
            # classification itself.
            #
            # Render it anyway, from the canonical mask. Leaving the
            # layer absent is what produced the real complaint: the
            # request 404s, the pane keeps whatever it was showing, and
            # the label changes -- so the pane claims to show a product
            # it is not showing. An honest identical layer is far
            # better than a mislabelled stale one.
            raw = clf
            if log:
                log('  No pre-brush mask on disk: brushing has not '
                    'modified this result, so "before brushing" is '
                    'identical to the classification.')

        # Report both pixel counts. If they are equal the layer is
        # real but shows nothing new (brushing changed nothing), and
        # that is worth saying rather than leaving the operator to
        # wonder whether the feature works.
        try:
            import numpy as _np
            from osgeo import gdal as _g
            def _count(path_):
                d = _g.Open(path_, _g.GA_ReadOnly)
                if d is None:
                    return -1
                a = d.GetRasterBand(1).ReadAsArray()
                d = None
                return int(_np.count_nonzero(_np.nan_to_num(a) > 0))
            n_raw, n_brushed = _count(raw), _count(clf)
            if log:
                log(f'  before brushing: {n_raw:,} px '
                    f'({os.path.basename(raw)}); after: {n_brushed:,} px '
                    f'({os.path.basename(clf)})')
                if n_raw == n_brushed:
                    log('  (identical -- brushing removed nothing, so '
                        'the two layers will look the same)')
        except Exception:
            pass

        # SAME red as 'ML classification': the two layers are the same
        # product before and after brushing, and a colour difference
        # would read as a difference in kind rather than in processing
        # -- which defeats the point of flickering between them.
        _overlay_mask_on_post(fire, raw, 'result_prebrush',
                              (0.9, 0.1, 0.0))
        png = os.path.join(fire.cache_dir, 'previews',
                           'result_prebrush.png')
        if not _verify_overlay_differs(fire, 'result_prebrush', log=log):
            if 'result_prebrush' in (fire.available_views or []):
                fire.available_views.remove('result_prebrush')
            return False
        if os.path.isfile(png):
            if 'result_prebrush' not in fire.available_views:
                fire.available_views.append('result_prebrush')
            if log:
                log('  Rendered "ML classification - before brushing"')
            return True
        return False
    except Exception as exc:
        sys.stderr.write(
            f'[prebrush] overlay failed: {type(exc).__name__}: {exc}\n')
        return False


def active_classified(fire, run_id=None) -> str:
    """The mask the eraser should edit.

    The canonical classified raster: it is what the overlay, agreement,
    area, Accept and export all read, so editing anything else would
    show a change that nothing downstream honoured.
    """
    from .state import find_classified

    # Prefer the run the UI is SHOWING, then the most recent.
    #
    # Taking the first entry meant that with several runs in the
    # gallery an edit landed on run 1's raster while the pane displayed
    # run 3: the change appeared while painting locally, then vanished
    # the moment the pane refetched run 3. Silent, and it corrupted an
    # older result at the same time.
    runs = [r for r in (getattr(fire, 'serial_results', None) or [])
            if r.get('classified')]
    if run_id is not None:
        for r in runs:
            if int(r.get('run_id') or 0) == int(run_id):
                c = r.get('classified')
                if c and os.path.isfile(c):
                    return c
    for r in sorted(runs, key=lambda r: int(r.get('run_id') or 0),
                    reverse=True):
        c = r.get('classified')
        if c and os.path.isfile(c):
            return c
    return find_classified(fire, [fire.cache_dir]) or ''
