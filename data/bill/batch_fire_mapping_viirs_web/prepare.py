"""Synchronous prepare + accept flow.

Both functions run from the request thread (or from the serial worker's
re-prepare path) and own the per-fire cache_dir → canonical-output-dir
hand-off. Holds no GPU lock; the caller arranges that.
"""

import datetime
import glob
import json
import os
import shutil
import queue
import re
import sys
import threading
import time

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
                          output_path: str) -> int:
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

    Returns the number of fire (1) pixels written, or -1 on failure.
    A return of 0 means the rule matched nothing anywhere in the crop:
    the file is still valid, but it is useless as a hint and callers
    should treat it as an error rather than hand it to the mapping CLI.
    """
    ds = gdal.Open(crop_bin, gdal.GA_ReadOnly)
    if ds is None:
        return -1
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
        return -1

    red, green, blue = channels[0], channels[1], channels[2]

    # "Red wins" = the first band strictly exceeds the other two at
    # this pixel.  This is the core logic from dominant_band.py.
    mask = (red > green) & (red > blue)

    # Pixels where any input band is NaN (nodata, usually the crop
    # margins) count as "no evidence of burn" -> 0.
    #
    # This must NOT write NaN. The mapping CLI validates the hint with
    # a strict "single-band 0/1 raster" check, and a NaN is neither 0
    # nor 1, so a mask carrying nodata is rejected outright with
    # "VIIRS hint is not a valid single-band 0/1 raster". Writing the
    # band as Byte makes that invariant structural rather than a
    # convention -- a NaN simply cannot be represented, so this class
    # of failure cannot recur.
    any_nan = np.isnan(red) | np.isnan(green) | np.isnan(blue)
    result = np.where(any_nan, 0, mask).astype(np.uint8)
    n_fire = int(result.sum())

    # Single-band ENVI, 0/1 values but written as data type 4
    # (float32).
    #
    # Byte was the natural choice for a 1/0 mask and it is what this
    # wrote for a long time, but every consumer here expects type 4:
    # the KGC binary rejects anything else outright ("only ENVI data
    # type 4 is supported; got type 1"), and the stack and classified
    # rasters are all float32. Uniform typing costs 3 bytes a pixel on
    # a mask and removes a whole class of downstream failure.
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(output_path, w, h, 1, gdal.GDT_Float32)
    if out_ds is None:
        return -1
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    # Cast explicitly: GDAL would coerce, but writing a bool/uint8
    # array into a float32 band without saying so is the kind of
    # implicit conversion that quietly changes if the array's dtype
    # changes upstream.
    out_ds.GetRasterBand(1).WriteArray(result.astype('float32'))
    out_ds.FlushCache()
    out_ds = None
    return n_fire


# Hint modes that are DERIVED FROM THE AOI STACK OR VECTOR DATA, i.e.
# rebuilt whenever the crop changes. VIIRS is not here: its mask comes
# from downloaded granules rather than from the stack.
DERIVED_HINT_MODES = ('redwins_post', 'redwins_diff', 'bcws_perimeter')
ALL_HINT_MODES = ('viirs',) + DERIVED_HINT_MODES


def rename_fire(old_name: str, new_name: str) -> dict:
    """Rename a fire, moving everything the old name keyed.

    The name is not just a label -- it identifies the fire in
    ``state.fires``, names the working cache directory, names the
    accepted-result directory, and is embedded in per-fire filenames.
    Renaming only the label would leave the fire pointing at
    directories under its old name: it would keep working until
    something rebuilt a path from the new name, then fail confusingly.

    Directories are moved first, because that is the step that can fail
    (permissions, a file held open); in-memory state is only updated
    once the filesystem is consistent, so a failure leaves the fire
    exactly as it was rather than half-renamed.

    Per-fire FILENAMES inside the cache keep the old name. They are
    referenced through absolute paths held on the FireInfo, so they
    stay valid, and renaming them would mean rewriting several
    sidecars for a cosmetic gain. The accepted-result directory is
    what carries the name into exports, and it is moved.

    Returns ``{'ok': True, 'name': new}`` or
    ``{'ok': False, 'error': ...}``.
    """
    from .validation import _validate_fire_name

    old_name = (old_name or '').strip()
    new_name = (new_name or '').strip()
    if old_name not in state.fires:
        return {'ok': False, 'error': 'Fire not found'}
    if not new_name:
        return {'ok': False, 'error': 'New name is required'}
    if new_name == old_name:
        return {'ok': True, 'name': old_name, 'unchanged': True}

    fire = state.fires[old_name]

    # Refuse while work is in flight: a worker thread holds this
    # FireInfo and writes into the old directories, so moving them
    # underneath it would corrupt the run.
    busy = {FireStatus.PENDING, FireStatus.PREPARING, FireStatus.MAPPING}
    if fire.status in busy:
        return {'ok': False,
                'error': f'Cannot rename while the fire is '
                         f'{fire.status.value}. Wait for it to finish '
                         f'or cancel it first.'}

    others = [n for n in state.fires if n != old_name]
    try:
        new_name = _validate_fire_name(new_name, existing_names=others)
    except ValueError as exc:
        return {'ok': False, 'error': str(exc)}

    moves = []
    try:
        # 1. Working cache directory.
        old_cache = getattr(fire, 'cache_dir', '') or ''
        if old_cache and os.path.isdir(old_cache):
            parent = os.path.dirname(old_cache)
            if os.path.basename(old_cache) == old_name:
                new_cache = os.path.join(parent, new_name)
                if os.path.exists(new_cache):
                    return {'ok': False,
                            'error': f'A directory already exists at '
                                     f'{new_cache}'}
                os.rename(old_cache, new_cache)
                moves.append((new_cache, old_cache))
                fire.cache_dir = new_cache

        # 2. Accepted-result directory, which is what exports carry.
        if state.output_root:
            old_out = os.path.join(state.output_root, old_name)
            new_out = os.path.join(state.output_root, new_name)
            if os.path.isdir(old_out):
                if os.path.exists(new_out):
                    raise OSError(
                        f'A result directory already exists at {new_out}')
                os.rename(old_out, new_out)
                moves.append((new_out, old_out))
    except Exception as exc:
        # Undo any move already made, so a partial rename cannot
        # survive the failure.
        for src, dst in reversed(moves):
            try:
                os.rename(src, dst)
            except OSError:
                pass
        if moves:
            fire.cache_dir = moves[0][1] if moves else fire.cache_dir
        return {'ok': False,
                'error': f'Could not move files: '
                         f'{type(exc).__name__}: {exc}'}

    # Filesystem is consistent; now update state under the lock.
    with state.lock:
        fire.fire_numbe = new_name
        state.fires[new_name] = state.fires.pop(old_name)
        # Every OTHER registry keyed by the fire name has to follow,
        # or a later lookup under the new name misses and the fire
        # looks idle when it is not (or a lock stops protecting it).
        for attr in ('viirs_jobs', 'viirs_subprocs'):
            d = getattr(state, attr, None)
            if isinstance(d, dict) and old_name in d:
                d[new_name] = d.pop(old_name)

    # Module-level registries live outside AppState. Renaming is
    # refused while the fire is busy, so these should be empty for it,
    # but a stale entry would otherwise be orphaned under the old name.
    for mod_name, reg_name in (('.app', '_serial_procs'),
                               ('.brush', '_rebrush_procs'),
                               ('.prepare', '_SOURCE_SWITCH_LOCKS')):
        try:
            if mod_name == '.prepare':
                reg = globals().get(reg_name)
            else:
                import importlib
                mod = importlib.import_module(mod_name, __package__)
                reg = getattr(mod, reg_name, None)
            if isinstance(reg, dict) and old_name in reg:
                reg[new_name] = reg.pop(old_name)
                sys.stderr.write(
                    f'[rename] moved {reg_name} entry\n')
        except Exception as exc:
            sys.stderr.write(
                f'[rename] could not move {reg_name}: {exc}\n')

    with state.lock:

        # Repoint absolute paths whose directory moved.
        for attr in ('crop_bin', 'hint_bin', 'viirs_bin', 'perim_bin'):
            val = getattr(fire, attr, '') or ''
            for new_dir, old_dir in moves:
                if val.startswith(old_dir + os.sep):
                    setattr(fire, attr,
                            new_dir + val[len(old_dir):])
                    break

    try:
        _save_fire_state()
    except Exception as exc:
        sys.stderr.write(f'[rename] state save failed: {exc}\n')

    sys.stderr.write(
        f'[rename] "{old_name}" -> "{new_name}" '
        f'({len(moves)} director(y/ies) moved)\n')
    return {'ok': True, 'name': new_name, 'moved': len(moves)}


def clip_mask_to_bcws(fire: FireInfo, clf_path: str,
                      log=None) -> bool:
    """Remove classified pixels outside the BCWS perimeter polygons.

    An official perimeter bounds where the fire is; anything the
    classifier finds beyond it is either a different fire or a false
    positive, and for a product that will be compared against BCWS
    records the outside pixels are noise.

    Reuses the BCWS hint raster, so the clip and the "BCWS perimeter"
    hint are the same geometry by construction -- clipping against a
    separately-rasterised copy could disagree with what the operator
    sees on screen.

    Modifies *clf_path* in place and returns True when anything was
    removed. A missing perimeter is reported and left alone rather
    than clearing the mask: an empty result is worse than an unclipped
    one.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    try:
        import numpy as np
        from osgeo import gdal

        if not clf_path or not os.path.isfile(clf_path):
            return False

        mask_path, err = build_bcws_hint_for_fire(fire)
        if not mask_path or not os.path.isfile(mask_path):
            emit(f'  Clip to BCWS skipped: {err or "no perimeter mask"}')
            return False

        ds_c = gdal.Open(clf_path, gdal.GA_Update)
        if ds_c is None:
            ds_c = gdal.Open(clf_path, gdal.GA_ReadOnly)
            if ds_c is None:
                emit('  Clip to BCWS skipped: cannot open the mask')
                return False
        arr = ds_c.GetRasterBand(1).ReadAsArray()
        ds_p = gdal.Open(mask_path, gdal.GA_ReadOnly)
        per = ds_p.GetRasterBand(1).ReadAsArray() if ds_p else None
        ds_p = None
        if arr is None or per is None:
            ds_c = None
            emit('  Clip to BCWS skipped: could not read a raster')
            return False
        if arr.shape != per.shape:
            ds_c = None
            emit(f'  Clip to BCWS skipped: shapes differ '
                 f'{arr.shape} vs {per.shape}')
            return False

        keep = np.nan_to_num(per) > 0
        before = int(np.count_nonzero(np.nan_to_num(arr) > 0))
        out = np.where(keep, np.nan_to_num(arr), 0.0).astype('float32')
        after = int(np.count_nonzero(out > 0))
        if after == 0 and before > 0:
            ds_c = None
            emit(f'  Clip to BCWS skipped: it would have removed all '
                 f'{before:,} classified pixel(s) -- the perimeter and '
                 f'the result do not overlap')
            return False

        band = ds_c.GetRasterBand(1)
        band.WriteArray(out)
        band.FlushCache()
        band = None
        ds_c = None
        emit(f'  Clipped to BCWS perimeter: {before:,} -> {after:,} '
             f'pixel(s) ({before - after:,} removed)')
        return before != after
    except Exception as exc:
        emit(f'  Clip to BCWS failed ({type(exc).__name__}: {exc}); '
             f'the mask was left unclipped')
        return False


def vectorize_classified(fire: FireInfo, clf_path: str = None,
                         log=None) -> dict:
    """Polygonize the accepted classification to Shapefile and KML.

    The raster mask is the model's output, but a fire perimeter is a
    VECTOR product: it goes into GIS, gets shared, gets edited. The
    export lost these at some point -- the accept step still copies
    *.shp/*.dbf/*.shx/*.prj, so nothing was ever producing them.

    Both formats, deliberately:
      * Shapefile in the raster's own CRS, for GIS work at full
        precision.
      * KML in EPSG:4326, because KML is defined in WGS84 and writing
        anything else produces a file that silently lands in the wrong
        place.

    Only class-1 (burned) pixels become polygons; the zero background
    is dropped. Returns a dict of what was written.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    out = {'shp': None, 'kml': None, 'polygons': 0, 'error': None}
    try:
        from osgeo import gdal, ogr, osr

        if clf_path is None:
            # find_classified() takes the FIRE (classified_names()
            # reads fire.crop_bin to derive the stack-based name), not
            # the fire number. Passing the string made every candidate
            # name wrong, so the lookup always missed and the download
            # reported "no classified raster found" even with a result
            # plainly on screen.
            #
            # Also search the ACCEPTED directory: on a download the
            # canonical copy lives there, and the cache may have been
            # swept.
            from .state import find_classified
            dirs = [fire.cache_dir]
            try:
                if state.output_root:
                    dirs.append(os.path.join(state.output_root,
                                             fire.fire_numbe))
            except Exception:
                pass
            clf_path = find_classified(fire, dirs)
            if not clf_path:
                # Last resort: whatever the active run recorded.
                try:
                    from .erase import active_classified
                    clf_path = active_classified(fire)
                except Exception:
                    clf_path = ''
        if not clf_path or not os.path.isfile(clf_path):
            out['error'] = 'no classified raster found'
            emit(f'[vector] {fire.fire_numbe}: {out["error"]}')
            return out

        ds = gdal.Open(clf_path, gdal.GA_ReadOnly)
        if ds is None:
            out['error'] = f'cannot open {clf_path}'
            emit(f'[vector] {fire.fire_numbe}: {out["error"]}')
            return out
        band = ds.GetRasterBand(1)
        proj = ds.GetProjection()
        srs = osr.SpatialReference()
        if proj:
            srs.ImportFromWkt(proj)

        # Mask so only burned pixels are polygonized -- without it
        # GDALPolygonize emits one huge polygon for the background too.
        mask_band = band.GetMaskBand()
        try:
            import numpy as np
            arr = band.ReadAsArray()
            mem_drv = gdal.GetDriverByName('MEM')
            mds = mem_drv.Create('', ds.RasterXSize, ds.RasterYSize, 1,
                                 gdal.GDT_Byte)
            mds.SetGeoTransform(ds.GetGeoTransform())
            if proj:
                mds.SetProjection(proj)
            mds.GetRasterBand(1).WriteArray(
                (np.nan_to_num(arr) > 0).astype('uint8') * 255)
            mask_band = mds.GetRasterBand(1)
        except Exception:
            mds = None

        base = os.path.join(fire.cache_dir, f'{fire.fire_numbe}_perimeter')
        shp_path = base + '.shp'
        for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
            try:
                os.remove(base + ext)
            except OSError:
                pass

        shp_drv = ogr.GetDriverByName('ESRI Shapefile')
        shp_ds = shp_drv.CreateDataSource(shp_path)
        layer = shp_ds.CreateLayer(
            f'{fire.fire_numbe}_perimeter', srs, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn('DN', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('FIRE_NUM', ogr.OFTString))
        layer.CreateField(ogr.FieldDefn('AREA_HA', ogr.OFTReal))

        gdal.Polygonize(band, mask_band, layer, 0, [], callback=None)

        # Drop background polygons and stamp attributes.
        n_poly = 0
        total_ha = 0.0
        layer.ResetReading()
        doomed = []
        for feat in layer:
            dn = feat.GetField('DN')
            geom = feat.GetGeometryRef()
            if dn is None or dn <= 0 or geom is None:
                doomed.append(feat.GetFID())
                continue
            ha = (geom.GetArea() or 0.0) / 10000.0
            feat.SetField('FIRE_NUM', str(fire.fire_numbe))
            feat.SetField('AREA_HA', round(ha, 4))
            layer.SetFeature(feat)
            total_ha += ha
            n_poly += 1
        for fid in doomed:
            layer.DeleteFeature(fid)
        shp_ds.ExecuteSQL(f'REPACK {layer.GetName()}')
        shp_ds = None
        mds = None
        ds = None

        out['shp'] = shp_path
        out['polygons'] = n_poly

        # KML must be WGS84.
        kml_path = base + '.kml'
        try:
            os.remove(kml_path)
        except OSError:
            pass
        try:
            src_ds = ogr.Open(shp_path)
            wgs = osr.SpatialReference()
            wgs.ImportFromEPSG(4326)
            try:
                wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            except AttributeError:
                pass
            gdal.VectorTranslate(
                kml_path, src_ds, format='KML',
                dstSRS='EPSG:4326', reproject=True)
            src_ds = None
            if os.path.isfile(kml_path):
                out['kml'] = kml_path
        except Exception as exc:
            emit(f'[vector] {fire.fire_numbe}: KML export failed: '
                 f'{exc} (shapefile was still written)')

        emit(f'[vector] {fire.fire_numbe}: {n_poly} polygon(s), '
             f'{total_ha:.2f} ha -> '
             f'{os.path.basename(shp_path)}'
             + (f' + {os.path.basename(kml_path)}'
                if out['kml'] else ''))
        return out
    except Exception as exc:
        out['error'] = f'{type(exc).__name__}: {exc}'
        emit(f'[vector] {fire.fire_numbe}: vectorization failed: '
             f'{out["error"]}')
        return out


def verify_and_repair_fire(fire: FireInfo, log=None) -> dict:
    """Check a fire's on-disk artifacts and rebuild what is missing.

    A fire's STATUS and its FILES can disagree. Preparation is several
    steps across a worker thread, a ramdisk and a cache directory, and
    a crash, a cleared /ram, a cache sweep or an interrupted run can
    leave the state saying READY while the previews are gone. The
    symptom is unhelpful -- 'View "Post-fire" not available' -- and
    there was no way to recover short of deleting and redrawing the
    AOI.

    Repairs in increasing order of cost, doing only what is needed:

      1. available_views empty but previews present -> re-derive the
         list from the directory (no raster work).
      2. previews missing but the stack is present -> regenerate the
         previews from the stack.
      3. stack missing -> needs a full re-prepare; reported, not
         attempted here, because it belongs on the worker queue.

    Returns a dict describing what was found and done.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    out = {'fire': fire.fire_numbe, 'actions': [], 'ok': True,
           'needs_full_rebuild': False}

    cache_dir = getattr(fire, 'cache_dir', '') or ''
    if not cache_dir or not os.path.isdir(cache_dir):
        out['ok'] = False
        out['needs_full_rebuild'] = True
        out['actions'].append('cache directory missing')
        emit(f'[verify] {fire.fire_numbe}: cache dir missing '
             f'({cache_dir or "unset"}) -- needs a full re-prepare')
        return out

    crop = getattr(fire, 'crop_bin', '') or ''
    if not crop or not os.path.isfile(crop):
        out['ok'] = False
        out['needs_full_rebuild'] = True
        out['actions'].append('AOI stack missing')
        emit(f'[verify] {fire.fire_numbe}: AOI stack missing '
             f'({crop or "unset"}) -- needs a full re-prepare. This is '
             f'expected after a reboot if the stack lived on /ram.')
        return out

    prev_dir = os.path.join(cache_dir, 'previews')
    have_post = os.path.isfile(os.path.join(prev_dir, 'post.png'))

    if not have_post:
        emit(f'[verify] {fire.fire_numbe}: previews missing -- '
             f'regenerating from {os.path.basename(crop)}')
        try:
            views = generate_all_previews(
                crop, cache_dir, fire.fire_numbe)
            stamp_previews_product(fire)
            try:
                from .mapping import record_base_preview_geo
                record_base_preview_geo(cache_dir, crop)
            except Exception:
                pass
            fire.available_views = list(views or [])
            out['actions'].append(
                f'regenerated previews ({len(views or [])} view(s))')
        except Exception as exc:
            out['ok'] = False
            out['actions'].append(f'preview regeneration failed: {exc}')
            emit(f'[verify] {fire.fire_numbe}: preview regeneration '
                 f'failed: {type(exc).__name__}: {exc}')
            return out

    # Re-derive the view list from what is actually on disk. Cheap, and
    # it is the field the client validates against -- an empty list is
    # what produces "View ... not available" even when the images exist.
    try:
        whitelist = ('pre', 'post', 'diff1', 'diff2', 'diff3', 'hint',
                     'result', 'result_prebrush')
        names = [os.path.splitext(f)[0]
                 for f in sorted(os.listdir(prev_dir))
                 if f.endswith('.png')]
        found = [n for n in names if n in whitelist]
        if ('hint' not in found
                and any(n.startswith('hint_') for n in names)):
            found.append('hint')
        if found and set(found) != set(fire.available_views or []):
            before = len(fire.available_views or [])
            fire.available_views = found
            out['actions'].append(
                f'view list rebuilt: {before} -> {len(found)}')
            emit(f'[verify] {fire.fire_numbe}: view list rebuilt '
                 f'({before} -> {len(found)}): {", ".join(found)}')
    except OSError as exc:
        out['actions'].append(f'could not list previews: {exc}')

    # Rebuild previews/result.png when a run is recorded but its
    # overlay is gone.
    #
    # serial_results is persisted, so a fire can come back from a
    # restart knowing it has a result while the image that shows it has
    # been lost (a stale per-source stash used to do exactly that). The
    # classified raster is still on disk, so the overlay can simply be
    # re-rendered -- and the "ML classification" view is available
    # again without the user having to re-run or re-accept anything.
    try:
        results = list(getattr(fire, 'serial_results', None) or [])
        if results and 'result' not in (fire.available_views or []):
            newest = results[-1]
            clf = newest.get('classified') or ''
            if not clf or not os.path.isfile(clf):
                from .state import find_classified
                clf = find_classified(fire, [cache_dir]) or ''
            if clf and os.path.isfile(clf):
                from .mapping import _overlay_mask_on_post
                _overlay_mask_on_post(fire, clf, 'result',
                                      (0.9, 0.1, 0.0))
                rp = os.path.join(prev_dir, 'result.png')
                if os.path.isfile(rp):
                    if 'result' not in fire.available_views:
                        fire.available_views.append('result')
                    out['actions'].append(
                        'rebuilt the ML classification overlay')
                    emit(f'[verify] {fire.fire_numbe}: rebuilt '
                         f'previews/result.png from '
                         f'{os.path.basename(clf)}')
            else:
                emit(f'[verify] {fire.fire_numbe}: a run is recorded '
                     f'but no classified raster was found; the ML '
                     f'classification view cannot be rebuilt')
    except Exception as exc:
        out['actions'].append(f'result overlay rebuild failed: {exc}')

    # Repair map info on the classification if it was lost. This runs
    # on open and in the startup sweep, so a mask written by an older
    # build -- before every writer set the geotransform -- is corrected
    # once rather than plotting in the wrong place forever.
    try:
        from .erase import active_classified, ensure_geo
        _clf = active_classified(fire)
        if _clf and fire.crop_bin:
            if ensure_geo(_clf, fire.crop_bin):
                out['actions'].append('restored map info on the '
                                      'classification')
    except Exception as exc:
        out['actions'].append(f'geo check failed: {exc}')

    # Render "ML Classification - before brushing" for fires whose
    # result predates the layer. The pre-brush mask (_raw.bin) has been
    # written by every brushing path all along, so the layer can be
    # produced retroactively -- without this it only ever appeared on
    # fires mapped AFTER the change, which looks like the feature is
    # missing.
    try:
        from .erase import render_prebrush_overlay, prebrush_path, \
            active_classified
        _clf2 = active_classified(fire)
        if _clf2 and os.path.isfile(prebrush_path(_clf2)):
            _png = os.path.join(prev_dir, 'result_prebrush.png')
            if not os.path.isfile(_png):
                if render_prebrush_overlay(fire, _clf2):
                    out['actions'].append(
                        'rendered the pre-brush classification layer')
    except Exception as exc:
        out['actions'].append(f'pre-brush layer skipped: {exc}')

    # A hint the CLI would consume must still exist, or mapping fails
    # at run time with a less obvious message.
    hb = getattr(fire, 'hint_bin', '') or ''
    if hb and not os.path.isfile(hb):
        mode = getattr(fire, 'hint_mode', 'redwins_post') or 'redwins_post'
        if mode in DERIVED_HINT_MODES:
            path, err = build_derived_hint_for_fire(fire, mode)
            if path:
                fire.hint_bin = path
                out['actions'].append(f'rebuilt {mode} hint')
                emit(f'[verify] {fire.fire_numbe}: rebuilt {mode} hint')
            else:
                out['actions'].append(f'hint rebuild failed: {err}')

    if out['actions']:
        try:
            _save_fire_state()
        except Exception:
            pass
    return out


def restrict_hint_to_bcws(fire: FireInfo, hint_path: str,
                          log=None) -> str:
    """Clip a hint mask to the BCWS perimeter polygons.

    Written to a SEPARATE file rather than edited in place: the
    unrestricted hint is still the right answer when the checkbox is
    off, and rebuilding it from scratch on every toggle would be slow
    and would lose the VIIRS mask, which is downloaded rather than
    derived.

    Returns the restricted path, or the original when there is nothing
    to clip against -- an empty hint would fail the run, and silently
    substituting one is worse than ignoring the setting.
    """
    def emit(msg):
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log(msg)
            except Exception:
                pass

    try:
        import numpy as np
        from osgeo import gdal

        if not hint_path or not os.path.isfile(hint_path):
            return hint_path
        per_path, err = build_bcws_hint_for_fire(fire)
        if not per_path or not os.path.isfile(per_path):
            emit(f'[hint] restrict to BCWS skipped: '
                 f'{err or "no perimeter"}')
            return hint_path

        out = (os.path.splitext(hint_path)[0] + '_bcws.bin')
        try:
            if (os.path.isfile(out)
                    and os.path.getmtime(out) >= os.path.getmtime(hint_path)
                    and os.path.getmtime(out) >= os.path.getmtime(per_path)):
                return out
        except OSError:
            pass

        hds = gdal.Open(hint_path, gdal.GA_ReadOnly)
        pds = gdal.Open(per_path, gdal.GA_ReadOnly)
        if hds is None or pds is None:
            return hint_path
        harr = hds.GetRasterBand(1).ReadAsArray()
        parr = pds.GetRasterBand(1).ReadAsArray()
        if harr is None or parr is None or harr.shape != parr.shape:
            emit('[hint] restrict to BCWS skipped: shapes differ')
            hds = None
            pds = None
            return hint_path

        keep = np.nan_to_num(parr) > 0
        res = np.where(keep, np.nan_to_num(harr), 0.0).astype('float32')
        before = int(np.count_nonzero(np.nan_to_num(harr) > 0))
        after = int(np.count_nonzero(res > 0))
        if after == 0:
            emit(f'[hint] restrict to BCWS skipped: it would empty the '
                 f'hint ({before:,} px, none inside the perimeter)')
            hds = None
            pds = None
            return hint_path

        drv = gdal.GetDriverByName('ENVI')
        ods = drv.Create(out, hds.RasterXSize, hds.RasterYSize, 1,
                         gdal.GDT_Float32, options=['INTERLEAVE=BSQ'])
        ods.SetGeoTransform(hds.GetGeoTransform())
        pr = hds.GetProjection()
        if pr:
            ods.SetProjection(pr)
        b = ods.GetRasterBand(1)
        b.WriteArray(res)
        b.SetDescription('hint restricted to BCWS perimeter')
        b = None
        ods = None
        hds = None
        pds = None
        # One header per raster, at <stem>.hdr.
        #
        # This used to be "rename the appended header only if the stem one
        # is missing", which did nothing in the single case that matters:
        # when BOTH exist, GDAL reads the appended one, and the tool's
        # appended header carries no map info -- so the raster reported
        # origin (0, 0) and was rejected from its own AOI.
        from .aoi_stack import normalize_envi_header
        normalize_envi_header(out)
        emit(f'[hint] restricted to BCWS perimeter: {before:,} -> '
             f'{after:,} px')
        return out
    except Exception as exc:
        emit(f'[hint] restrict to BCWS failed ({exc}); using the '
             f'unrestricted hint')
        return hint_path


def derived_hint_path(fire: FireInfo, mode: str) -> str:
    """Where this product's hint for *mode* lives, if it has one.

    Same naming the builders use, so "do we already have it?" can be
    answered without deriving anything.
    """
    try:
        pkey = (product_key_for_path(getattr(fire, 'crop_bin', '') or '')
                or product_key(getattr(fire, 'post_source', 'l2') or 'l2',
                               getattr(fire, 'l2_start_date', '') or ''))
        return os.path.join(fire.cache_dir, '_redwins',
                            f'{mode}_{pkey}_hint.bin')
    except Exception:
        return ''


# ---------------------------------------------------------------
# Preview warming queue
# ---------------------------------------------------------------
# Rendering a product's preview PNGs is what stands between "the stack
# is on disk" and "switching to it is instant". It used to happen only
# when a build finished in this process, so anything built by an
# earlier run -- or restored from the durable store -- sat unrendered
# until someone selected it and waited.
#
# The work is idempotent and its state lives on disk: a preview either
# exists or it does not, and preview.py writes through a temporary file
# and os.replace, so a process killed mid-render leaves either the old
# file or nothing, never a half one. That makes a persistent journal
# unnecessary -- and less reliable than the filesystem it would
# describe. Resuming is simply sweeping again, which is what happens on
# every startup.
_PREVIEW_WORKERS = 3
_preview_q: 'queue.Queue' = queue.Queue()
_preview_inflight: set = set()      # idents queued or running
_preview_lock = threading.Lock()
_preview_workers_started = False
_preview_stats = {'done': 0, 'failed': 0, 'skipped': 0}


# ---------------------------------------------------------------------
# Per-product state notes
# ---------------------------------------------------------------------
#
# Keyed by FIRE **and** PRODUCT, so a note can only ever describe the
# one product it was written for. The fire-wide ``fire.progress`` is
# deliberately NOT used here: it is a single slot shared by every
# product of a fire (and by a mapping run), so a note taken from it
# would show one product's work against another's row.
#
# Every note expires. A message that outlives the thing it describes
# is worse than no message, so each state carries the longest time it
# can still be true for; a stale entry is dropped on read as well as
# on write.
_product_state = {}
_product_state_lock = threading.Lock()

# Seconds a note stays believable. Work that is actively re-reported
# (building, rendering) gets a short life and is refreshed by its own
# progress; a conclusion that persists until something changes the
# product (withheld, retired) gets a long one.
PRODUCT_STATE_TTL = {
    'building': 1800.0,
    'queued': 600.0,
    'rendering': 900.0,
    'build_failed': 900.0,
    'restoring': 60.0,
    'repointing': 20.0,
    'busy': 20.0,
    'rerendering': 60.0,
    'preview_failed': 900.0,
    'preview_skipped': 60.0,
    'hint_empty': 900.0,
    'withheld': 3600.0,
    'retired': 600.0,
    'grid_deviates': 3600.0,
}
_PRODUCT_STATE_DEFAULT_TTL = 60.0
_PRODUCT_STATE_MAX = 4000        # a hard ceiling; oldest go first


def _product_state_ident(fire_numbe: str, key: str) -> str:
    return '%s:%s' % (fire_numbe or '', key or '')


# States that mean "work is in progress on this product right now". The
# Sources panel keeps polling while any row is in one of them, and stops
# once every row is settled -- rather than polling forever because some
# product simply has not been built.
ACTIVE_PRODUCT_STATES = frozenset((
    'queued', 'building', 'restoring', 'repointing', 'busy',
    'rendering', 'rerendering'))


def note_product_state(fire_numbe: str, key: str, state: str,
                       detail: str = '', frac: float = None,
                       stage: str = None) -> None:
    """Record what is happening to ONE product, for its own row.

    With a *stage* (one of PREP_STAGES) and a fraction through it, the
    note carries the same progress model the fire list uses for a fire
    being prepared: stage label and number, weighted overall fraction,
    a smoothed ETA, the time the work started and the time anything
    last changed -- so a row can say "~1m 20s left (45%)" and, when
    nothing moves, "no change for 5m", exactly as the fire list does.
    """
    if not fire_numbe or not key or not state:
        return
    try:
        now = time.time()
        ttl = PRODUCT_STATE_TTL.get(state, _PRODUCT_STATE_DEFAULT_TTL)
        ident = _product_state_ident(fire_numbe, key)
        with _product_state_lock:
            prev = _product_state.get(ident) or {}
            continuing = (state in ACTIVE_PRODUCT_STATES
                          and prev.get('state') in ACTIVE_PRODUCT_STATES
                          and now - prev.get('at', 0) <= prev.get('ttl', 0))
            started = (prev.get('started_at') if continuing else None) or now
            n = {'state': state, 'detail': detail or '', 'at': now,
                 'ttl': ttl, 'started_at': started}
            if stage in _PREP_INDEX:
                idx = _PREP_INDEX[stage]
                f = max(0.0, min(1.0, float(frac or 0.0)))
                done = sum(w for _k, _l, w in PREP_STAGES[:idx])
                overall = max(0.0, min(0.999, done + PREP_STAGES[idx][2] * f))
                elapsed = max(0.0, now - float(started))
                eta = None
                if overall >= 0.04 and elapsed >= 5.0:
                    raw = max(0.0, elapsed * (1.0 - overall) / overall)
                    pe = prev.get('eta_s') if continuing else None
                    if isinstance(pe, (int, float)) and pe >= 0:
                        alpha = min(0.9, 0.2 + 0.7 * overall)
                        eta = (1.0 - alpha) * pe + alpha * raw
                    else:
                        eta = raw
                n.update({'stage': stage,
                          'stage_label': prep_stage_label(stage),
                          'stage_idx': idx + 1,
                          'total_stages': len(PREP_STAGES),
                          'fraction': overall, 'stage_fraction': f,
                          'eta_s': eta})
            changed = (not continuing
                       or prev.get('detail') != n['detail']
                       or prev.get('stage') != n.get('stage')
                       or round(float(prev.get('fraction') or 0), 3)
                       != round(float(n.get('fraction') or 0), 3))
            n['last_change_at'] = (now if changed
                                   else prev.get('last_change_at', now))
            _product_state[ident] = n
            if len(_product_state) > _PRODUCT_STATE_MAX:
                dead = [i for i, x in _product_state.items()
                        if now - x['at'] > x['ttl']]
                for i in dead:
                    _product_state.pop(i, None)
                while len(_product_state) > _PRODUCT_STATE_MAX:
                    oldest = min(_product_state,
                                 key=lambda i: _product_state[i]['at'])
                    _product_state.pop(oldest, None)
    except Exception:
        pass                      # a note is never worth an exception


def clear_product_state(fire_numbe: str, key: str) -> None:
    """Forget this product's note -- it has reached a settled state."""
    try:
        with _product_state_lock:
            _product_state.pop(
                _product_state_ident(fire_numbe, key), None)
    except Exception:
        pass


def product_state_note(fire_numbe: str, key: str):
    """This product's note, or None if there is none or it expired."""
    try:
        ident = _product_state_ident(fire_numbe, key)
        now = time.time()
        with _product_state_lock:
            n = _product_state.get(ident)
            if not n:
                return None
            if now - n['at'] > n['ttl']:
                _product_state.pop(ident, None)
                return None
            return dict(n)
    except Exception:
        return None


def forget_fire_product_states(fire_numbe: str) -> int:
    """Drop every note belonging to one fire.

    A deleted fire's notes must not outlive it. The key is the fire
    NUMBER, which is reused the moment the operator recreates a fire
    with the same name, so a surviving note would be read as the new
    fire's -- which is exactly how a freshly created fire came to show
    'withheld' rows describing the deleted one's products.
    """
    gone = 0
    try:
        pre = '%s:' % (fire_numbe or '')
        with _product_state_lock:
            for ident in [i for i in _product_state if i.startswith(pre)]:
                _product_state.pop(ident, None)
                gone += 1
        # Its deletions go with it (the manifest holding them lives in
        # the fire's cache, which is purged with the fire).
        with _tomb_lock:
            _tombs.pop(fire_numbe or '', None)
            _tombs_loaded.discard(fire_numbe or '')
    except Exception:
        pass
    return gone


# ---- Deleted products -------------------------------------------------
#
# A product the operator deletes must stay deleted. Several paths build a
# product on their own initiative -- the on-demand preview path switching
# the fire back to the operator's remembered product, the refresh of each
# fire's default products at startup -- and each of them resurrected a
# product that had just been deleted. A tombstone records the deletion;
# those implicit paths skip a tombstoned product, and only an explicit
# request to build it again (Date select) lifts the tombstone.
#
# Kept in memory, and in the fire's manifest so a restart honours it. The
# manifest lives in the fire's cache, so a removed or recreated fire never
# inherits another fire's tombstones.

_tomb_lock = threading.Lock()
_tombs = {}              # fire number -> {product key: deleted at}
_tombs_loaded = set()    # fires whose manifest tombstones have been read


def _tombs_for(fire) -> dict:
    """This fire's tombstones (loaded from its manifest once per run)."""
    fn = getattr(fire, 'fire_numbe', '') or ''
    with _tomb_lock:
        if fn in _tombs_loaded:
            return _tombs.setdefault(fn, {})
    got = {}
    try:
        from .manifest import load as _mload
        got = dict((_mload(fire) or {}).get('deleted') or {})
    except Exception:
        got = {}
    with _tomb_lock:
        d = _tombs.setdefault(fn, {})
        for k, v in got.items():
            try:
                d.setdefault(str(k), float(v))
            except (TypeError, ValueError):
                pass
        _tombs_loaded.add(fn)
        return d


def _save_tombs(fire) -> None:
    try:
        from .manifest import load as _mload, save as _msave
        with _tomb_lock:
            snap = dict(_tombs.get(fire.fire_numbe) or {})
        man = _mload(fire) or {}
        man['deleted'] = snap
        _msave(fire, man)
    except Exception as exc:
        sys.stderr.write(f'[sources] could not record deletions in the '
                         f'manifest: {exc}\n')


def tombstone_products(fire, keys) -> None:
    """Record that the operator deleted these products."""
    keys = [k for k in (keys or []) if k]
    if not keys:
        return
    _tombs_for(fire)
    now = time.time()
    with _tomb_lock:
        d = _tombs.setdefault(fire.fire_numbe, {})
        for k in keys:
            d[k] = now
    _save_tombs(fire)


def untombstone_product(fire, key: str) -> None:
    """The operator asked for this product again: lift its tombstone."""
    if not key or key not in _tombs_for(fire):
        return
    with _tomb_lock:
        (_tombs.get(fire.fire_numbe) or {}).pop(key, None)
    _save_tombs(fire)
    sys.stderr.write(f'[sources] {fire.fire_numbe}: {key} requested '
                     f'again; its deletion no longer applies\n')


def product_tombstone(fire, key: str) -> float:
    """When this product was deleted, or 0.0 if it was not."""
    if not key:
        return 0.0
    try:
        return float(_tombs_for(fire).get(key) or 0.0)
    except Exception:
        return 0.0


def product_state_deleted_since(fire, key: str, path: str) -> bool:
    """Tombstoned, and not rebuilt since (no file newer than the deletion).

    A copy of the deleted stack that survives -- the durable mirror
    finishing a copy it had already started, say -- keeps the old file's
    time, so it is still treated as deleted. A genuine rebuild writes a
    new file and is not.
    """
    ts = product_tombstone(fire, key)
    if not ts:
        return False
    try:
        return not (path and os.path.getmtime(path) > ts)
    except OSError:
        return True


def product_states_for_fire(fire_numbe: str) -> dict:
    """Every live note for one fire, keyed by product key."""
    out = {}
    try:
        pre = '%s:' % (fire_numbe or '')
        now = time.time()
        with _product_state_lock:
            for ident, n in list(_product_state.items()):
                if not ident.startswith(pre):
                    continue
                if now - n['at'] > n['ttl']:
                    _product_state.pop(ident, None)
                    continue
                out[ident[len(pre):]] = dict(n)
    except Exception:
        pass
    return out


# What each state says in the Sources column. Kept beside the states
# so the wording and the producers cannot drift apart.
PRODUCT_STATE_TEXT = {
    'building': 'building',
    'restoring': 'restoring from the durable store',
    'repointing': 'repointing to imagery already built',
    'busy': 'waiting: another switch holds this fire',
    'rerendering': 'previews stale for this grid; re-rendering',
    'preview_failed': 'preview render failed; will retry on use',
    'preview_skipped': 'preview render skipped',
    'hint_empty': 'no hint pixels for this product',
    'withheld': 'withheld: not on this AOI grid; rebuilds on use',
    'retired': 'retired: built on an older grid; rebuilding',
    'grid_deviates': 'grid differs from this AOI',
    'queued': 'queued',
    'rendering': 'rendering',
    'build_failed': 'build failed',
}

# The fire list's stage names (templates/fire_list.html,
# LIST_STAGE_LABELS), so a product row and the fire list describe the
# same work in the same words.
PROGRESS_STAGE_LABELS = {
    'downloading_viirs': 'VIIRS',
    'accumulating': 'VIIRS accumulate',
    'cropping': 'Build AOI stack',
    'locating': 'Locating imagery',
    'extracting': 'Reading Sentinel-2 data',
    'compositing': 'Building the AOI composite',
    'previews': 'Rendering preview imagery',
    'hint': 'Computing the hint layer',
}

_STALL_S = 180.0              # the fire list's "no change for" threshold


def fmt_dur(sec) -> str:
    """Same rendering as the fire list's fmtDur()."""
    try:
        sec = float(sec)
    except (TypeError, ValueError):
        return '--'
    if sec != sec or sec < 0:
        return '--'
    if sec < 60:
        return f'{int(round(sec))}s'
    if sec < 3600:
        m = int(sec // 60)
        r = int(round(sec - m * 60))
        return f'{m}m {r}s' if r else f'{m}m'
    h = int(sec // 3600)
    m = int(round((sec - h * 3600) / 60))
    return f'{h}h {m}m' if m else f'{h}h'


def progress_line(prog, now: float = None) -> str:
    """One progress snapshot as text, in the fire list's format.

    "<stage> (i/n) — <detail> · ~<eta> left (NN%)", or "<elapsed>
    elapsed" when there is no estimate yet, then "no change for <t>"
    once nothing has moved for three minutes. Works for a fire's
    snapshot (fire.progress) and a product note alike.
    """
    if not prog:
        return ''
    now = now or time.time()
    stage = prog.get('stage') or ''
    label = (prog.get('stage_label')
             or PROGRESS_STAGE_LABELS.get(stage, stage) or '')
    idx = prog.get('stage_idx') or 0
    tot = prog.get('total_stages') or 0
    head = label + (f' ({idx}/{tot})' if (label and tot) else '')
    tail = []
    det = (prog.get('detail') or '').strip()
    if det:
        tail.append(det)
    eta = prog.get('eta_s')
    frac = prog.get('fraction')
    started = prog.get('started_at')
    if isinstance(eta, (int, float)) and eta >= 0:
        tail.append(f'~{fmt_dur(eta)} left'
                    + (f' ({int(round(float(frac) * 100))}%)'
                       if isinstance(frac, (int, float)) else ''))
    elif isinstance(started, (int, float)) and started > 0:
        tail.append(f'{fmt_dur(now - started)} elapsed')
    lca = prog.get('last_change_at')
    if isinstance(lca, (int, float)) and now - lca > _STALL_S:
        tail.append(f'no change for {fmt_dur(now - lca)}')
    body = ' · '.join(tail)
    if head and body:
        return f'{head} — {body}'
    return head or body


def product_state_text(note, now: float = None) -> str:
    """The Sources-column wording for one note.

    A note with a stage is progress, and reads like the fire list's
    progress line. Any other note is its state's wording plus detail,
    with the same "no change for" flag while the work is active.
    """
    if not note:
        return ''
    now = now or time.time()
    if note.get('stage'):
        return progress_line(note, now)
    base = PRODUCT_STATE_TEXT.get(note.get('state'),
                                  note.get('state') or '')
    d = (note.get('detail') or '').strip()
    text = f'{base} \u2014 {d}' if (base and d) else (base or d)
    lca = note.get('last_change_at')
    if (note.get('state') in ACTIVE_PRODUCT_STATES
            and isinstance(lca, (int, float)) and now - lca > _STALL_S):
        text += f' \u00b7 no change for {fmt_dur(now - lca)}'
    return text


def preview_queue_status() -> dict:
    """What the warming queue is doing, for the Sources panel."""
    with _preview_lock:
        return {'queued': _preview_q.qsize(),
                'in_flight': len(_preview_inflight),
                'idents': sorted(_preview_inflight)[:40],
                'done': _preview_stats['done'],
                'failed': _preview_stats['failed'],
                'skipped': _preview_stats['skipped']}


def preview_dir_grid_ok(fire: FireInfo, pdir: str,
                        view: str = 'post') -> bool:
    """Was this preview directory rendered on the fire's grid?

    False only when BOTH are known and they differ: the directory
    records its grid (geo.json, written by every render) and the fire
    has a pinned grid. Unrecorded or unpinned cannot be judged, and is
    treated as fine rather than thrown away.
    """
    try:
        from .preview_fs import read_geo, geo_entry_matches
        entry = read_geo(pdir).get(view)
        if not entry:
            return True
        from .aoi_stack import load_pinned_grid_for
        pin = load_pinned_grid_for(
            fire.fire_numbe, getattr(state, 'shared_root', '') or '')
        if not pin:
            return True
        return geo_entry_matches(entry, pin['width'], pin['height'],
                                 pin['gt'], tol_px=0.01)
    except Exception:
        return True


# A hint mask that cannot be made -- the red-wins rule matching no pixel
# of this product, no VIIRS data, no BCWS polygons -- is recorded beside
# the product's previews (hintmask_<mode>.none) with the reason. Until the
# record expires the mode counts as settled: the queue does not retry it
# every few seconds, and a pane asking for it gets the reason at once
# instead of "still being made" for ever. A transient failure is recorded
# the same way, briefly, so it is retried soon but not in a tight loop.
_HINT_EMPTY_TTL_S = 1800.0
_HINT_RETRY_TTL_S = 60.0


def available_hint_modes(fire: FireInfo) -> list:
    """The hint modes this fire can offer a mask for right now."""
    modes = ['redwins_post', 'redwins_diff']
    try:
        if fire.viirs_bin and os.path.isfile(fire.viirs_bin):
            modes.insert(0, 'viirs')
    except Exception:
        pass
    try:
        from .bcws import _overlay_json_path
        _bp = _overlay_json_path(state)
        if _bp and os.path.isfile(_bp):
            modes.append('bcws_perimeter')
    except Exception:
        pass
    return modes


def _hint_marker(pdir: str, mode: str) -> str:
    return os.path.join(pdir, f'hintmask_{mode}.none')


def hint_mask_problem(pdir: str, mode: str):
    """The recorded reason this mode has no mask here, if still current.

    Returns ``(reason, permanent)`` or ``None``.
    """
    try:
        with open(_hint_marker(pdir, mode), encoding='utf-8') as fh:
            rec = json.load(fh) or {}
        perm = bool(rec.get('permanent'))
        ttl = _HINT_EMPTY_TTL_S if perm else _HINT_RETRY_TTL_S
        if time.time() - float(rec.get('at') or 0) <= ttl:
            return (str(rec.get('reason') or ''), perm)
    except (OSError, ValueError, TypeError):
        pass
    return None


def _record_hint_problem(pdir: str, mode: str, reason: str,
                         permanent: bool) -> None:
    try:
        from .preview_fs import merge_json
        merge_json(_hint_marker(pdir, mode),
                   {'reason': (reason or '')[:300], 'at': time.time(),
                    'permanent': bool(permanent)})
    except Exception:
        pass


def hint_masks_complete(fire: FireInfo, key: str) -> bool:
    """Is every available hint mode's mask made (or settled) for *key*?"""
    try:
        d = os.path.join(fire.cache_dir, f'previews_{key}')
        if not os.path.isdir(d):
            return False
        for mode in available_hint_modes(fire):
            if os.path.isfile(os.path.join(d, f'hintmask_{mode}.png')):
                continue
            if hint_mask_problem(d, mode):
                continue
            return False
        return True
    except Exception:
        return True          # never let a check keep a product queued


def previews_complete(fire: FireInfo, key: str) -> bool:
    """Has this product's post-fire preview been rendered, on its grid?"""
    try:
        d = os.path.join(fire.cache_dir, f'previews_{key}')
        return (os.path.isfile(os.path.join(d, 'post.png'))
                and preview_dir_grid_ok(fire, d))
    except Exception:
        return False


def _preview_worker() -> None:
    while True:
        item = None
        try:
            item = _preview_q.get()
        except Exception:
            continue
        ident = ''
        # Reset with ident: these persist across loop iterations, so a
        # failed unpack would otherwise leave the PREVIOUS product's
        # fire and key in scope and the error below would be recorded
        # against the wrong row.
        fire = None
        key = None
        try:
            fire, key, stack_path, ident = item
            # Check again at the moment of doing it: the product may
            # have been warmed by the interactive path, or deleted,
            # while this item waited its turn.
            if (previews_complete(fire, key)
                    and hint_masks_complete(fire, key)):
                with _preview_lock:
                    _preview_stats['skipped'] += 1
                # Already rendered: this product has nothing pending.
                clear_product_state(fire.fire_numbe, key)
                continue
            if not stack_path or not os.path.isfile(stack_path):
                sys.stderr.write(
                    f'[warmq] {ident}: stack is gone; nothing to '
                    f'render\n')
                with _preview_lock:
                    _preview_stats['skipped'] += 1
                note_product_state(fire.fire_numbe, key,
                                   'preview_skipped', 'stack is gone')
                continue
            t0 = time.time()
            # A render can lose its files to a directory cleared while it
            # ran (see preview_fs) -- nothing wrong with the product. Try
            # again before calling it a failure, and if it still fails,
            # say why rather than "produced no preview".
            _res, ok = {}, False
            for _attempt in range(3):
                _res = warm_product_artifacts(fire, stack_path) or {}
                ok = previews_complete(fire, key)
                if ok or not os.path.isfile(stack_path):
                    break
                time.sleep(0.5 * (_attempt + 1))
            with _preview_lock:
                _preview_stats['done' if ok else 'failed'] += 1
            # The counters above are process-wide and cannot say WHICH
            # product failed; this note can, because it is keyed.
            if ok:
                clear_product_state(fire.fire_numbe, key)
            else:
                note_product_state(
                    fire.fire_numbe, key, 'preview_failed',
                    ('; '.join(_res.get('errors') or [])
                     or 'no preview was produced')[:160])
            sys.stderr.write(
                '[warmq] %s: %s in %.1fs (%d still queued)\n'
                % (ident, 'rendered' if ok else 'produced no preview',
                   time.time() - t0, _preview_q.qsize()))
        except Exception as exc:
            with _preview_lock:
                _preview_stats['failed'] += 1
            try:
                if fire is not None and key:
                    note_product_state(fire.fire_numbe, key,
                                       'preview_failed', str(exc)[:80])
            except Exception:
                pass
            sys.stderr.write(f'[warmq] {ident or "?"}: failed: {exc}\n')
        finally:
            if ident:
                with _preview_lock:
                    _preview_inflight.discard(ident)
            try:
                _preview_q.task_done()
            except Exception:
                pass


def _ensure_preview_workers() -> None:
    """Start the pool once, lazily."""
    global _preview_workers_started
    with _preview_lock:
        if _preview_workers_started:
            return
        _preview_workers_started = True
        for i in range(_PREVIEW_WORKERS):
            threading.Thread(target=_preview_worker, daemon=True,
                             name=f'warmq-{i + 1}').start()
    sys.stderr.write(
        f'[warmq] {_PREVIEW_WORKERS} preview worker(s) started\n')


def enqueue_preview_warm(fire: FireInfo, key: str,
                         stack_path: str) -> bool:
    """Queue one product for warming. True if it was added.

    Idempotent and cheap to call repeatedly: a product already warmed,
    already queued, or already being rendered is not queued again.
    """
    if not fire or not key or not stack_path:
        return False
    ident = f'{fire.fire_numbe}:{key}'
    # Queued until the previews AND every available hint mode's mask are
    # made (or recorded as impossible) -- not just the previews, which
    # left products that already had previews without any hint mask.
    if previews_complete(fire, key) and hint_masks_complete(fire, key):
        return False
    with _preview_lock:
        if ident in _preview_inflight:
            return False
        _preview_inflight.add(ident)
    _ensure_preview_workers()
    _preview_q.put((fire, key, stack_path, ident))
    return True


def _fire_stack_prefix(fire: FireInfo) -> str:
    """``<safe>_<hash>`` for this fire, from the stack it is loaded on."""
    base = os.path.basename(getattr(fire, 'crop_bin', '') or '')
    m = re.match(r'^\d{8}_stack_(.+?_[0-9a-fA-F]{6,})(?:_|\.)', base)
    return m.group(1) if m else ''


def warm_outstanding_previews(fire: FireInfo) -> int:
    """Queue every product of this fire whose previews are missing."""
    pfx = _fire_stack_prefix(fire)
    if not pfx:
        return 0
    ram = os.path.dirname(getattr(fire, 'crop_bin', '') or '')
    if not ram or not os.path.isdir(ram):
        return 0
    added = 0
    for cand in sorted(glob.glob(
            os.path.join(ram, f'*_stack_{pfx}*.bin'))):
        bn = os.path.basename(cand)
        # Clustering scratch and post-fire buffers are inputs, not
        # products: they have no previews and never will.
        if ('_nob8' in bn or '.kgc' in bn or '.post.' in bn
                or '_selected' in bn or bn.endswith('.part')):
            continue
        try:
            key = product_key_for_path(cand)
        except Exception:
            key = ''
        if not key:
            continue
        if enqueue_preview_warm(fire, key, cand):
            added += 1
    if added:
        sys.stderr.write(
            f'[warmq] {fire.fire_numbe}: queued {added} product(s) '
            f'for preview rendering\n')
    return added


def warm_outstanding_previews_all(delay_s: float = 0.0) -> None:
    """Sweep every fire, in the background. Safe to call repeatedly."""
    def _run():
        if delay_s > 0:
            time.sleep(delay_s)
        total = 0
        try:
            names = list(state.fires.keys())
        except Exception:
            names = []
        for fn in names:
            fire = state.fires.get(fn)
            if fire is None:
                continue
            try:
                total += warm_outstanding_previews(fire)
            except Exception as exc:
                sys.stderr.write(
                    f'[warmq] {fn}: sweep failed: {exc}\n')
        sys.stderr.write(
            f'[warmq] startup sweep queued {total} product(s) across '
            f'{len(names)} fire(s)\n')

    threading.Thread(target=_run, daemon=True, name='warmq-sweep').start()


_hint_jobs: dict = {}
_hint_jobs_lock = threading.Lock()


def _defer_hint_build(fire: FireInfo, mode: str) -> None:
    """Derive one hint in the background, once per (fire, product)."""
    key = f'{fire.fire_numbe}:{derived_hint_path(fire, mode)}'

    def _run():
        try:
            path, err = build_derived_hint_for_fire(fire, mode)
            if path and getattr(fire, 'restrict_hint_bcws', False):
                path = restrict_hint_to_bcws(fire, path)
            if path:
                fire.hint_bin = path
                fire.perimeter_type = mode
                fire.hint_mode = mode
                try:
                    from .mapping import _overlay_mask_on_post
                    _overlay_mask_on_post(fire, path, 'hint',
                                          (0.0, 0.8, 0.2))
                    if 'hint' not in fire.available_views:
                        fire.available_views.append('hint')
                except Exception:
                    pass
                sys.stderr.write(
                    f'[hint] {fire.fire_numbe}: {mode} ready '
                    f'({os.path.basename(path)})\n')
            else:
                sys.stderr.write(
                    f'[hint] {fire.fire_numbe}: {mode} failed: {err}\n')
        except Exception as exc:
            sys.stderr.write(
                f'[hint] {fire.fire_numbe}: {mode} failed: {exc}\n')
        finally:
            with _hint_jobs_lock:
                _hint_jobs.pop(key, None)

    with _hint_jobs_lock:
        t = _hint_jobs.get(key)
        if t is not None and t.is_alive():
            return
        th = threading.Thread(target=_run, daemon=True,
                              name=f'hint-{fire.fire_numbe}')
        _hint_jobs[key] = th
        th.start()


def _defer_pregenerate_hints(fire: FireInfo) -> None:
    """Render every hint mode for this product, off the hot path."""
    def _run():
        try:
            pregenerate_all_hints(fire)
            sys.stderr.write(
                f'[hint] {fire.fire_numbe}: all hint modes '
                f'pre-rendered\n')
        except Exception as exc:
            sys.stderr.write(
                f'[hint] {fire.fire_numbe}: pregenerate failed: '
                f'{exc}\n')

    threading.Thread(target=_run, daemon=True,
                     name=f'hints-{fire.fire_numbe}').start()


def build_derived_hint_for_fire(fire: FireInfo, mode: str,
                                stack_path: str = ''):
    """Build whichever derived hint *mode* names.

    Single entry point so the call sites -- switch, prepare, re-prepare,
    pregenerate -- do not each need to know which builder handles which
    mode. Adding a mode means adding it here and to DERIVED_HINT_MODES,
    not editing five branches.
    """
    if mode == 'bcws_perimeter':
        return build_bcws_hint_for_fire(fire)
    # stack_path names the product to read when it is not the one the
    # fire currently points at -- rendering a hint for a layer in the
    # Sources list without switching to it. build_redwins_hint_for_fire
    # has always taken it; this entry point dropped it, so those calls
    # died with TypeError and the hint silently fell back to the plain
    # post-fire image. That is why hint.png equals post.png in several
    # preview directories.
    return build_redwins_hint_for_fire(fire, mode, stack_path)


def build_bcws_hint_for_fire(fire: FireInfo):
    """Hint mask from BCWS fire polygons intersecting the AOI.

    Every BCWS polygon overlapping the AOI is burned into one mask --
    deliberately NOT filtered to a particular fire number. This system
    detects fire/burn; deciding which perimeter belongs to which
    incident is somebody else's job, and filtering here would silently
    drop burn that belongs to a neighbouring fire.

    Written to the same place, in the same format, and with the same
    invalidation rule as the red-wins masks, so everything downstream
    -- the mapping CLI, the hint preview, agreement scoring -- consumes
    it without knowing the difference.

    Returns ``(path, None)`` or ``(None, error_message)``.
    """
    if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
        return None, 'Fire has no crop raster.'

    out_dir = os.path.join(fire.cache_dir, '_redwins')
    os.makedirs(out_dir, exist_ok=True)
    src = getattr(fire, 'post_source', 'l2') or 'l2'
    # Per source like the others: the mask must match the dimensions of
    # whichever stack is current, and switching source repoints
    # crop_bin at a different raster.
    # Product-keyed for the same reason as the red-wins masks: the
    # rasterisation follows the crop it is burned onto.
    # Key the hint by the PRODUCT the stack actually is.
    #
    # product_key(src, start_date) returns a bare 'l2' for a default
    # build, while every other part of the app identifies that same
    # product as 'l2_d<newest-date>'. The hint was therefore written
    # under one name and looked for under another, so it was recomputed
    # on every switch -- the "Computing the hint layer" that appeared
    # even for products prepared hours earlier.
    _pkey = (product_key_for_path(getattr(fire, 'crop_bin', '') or '')
             or product_key(src, getattr(fire, 'l2_start_date', '') or ''))
    out_path = os.path.join(out_dir,
                            f'bcws_perimeter_{_pkey}_hint.bin')

    try:
        if (os.path.isfile(out_path)
                and os.path.getmtime(out_path)
                >= os.path.getmtime(fire.crop_bin)):
            return out_path, None
    except OSError:
        pass

    ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
    if ds is None:
        return None, 'Could not open the AOI stack.'
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    w, h = ds.RasterXSize, ds.RasterYSize
    ds = None

    # Same province-wide overlay JSON the map overlays use: already in
    # the raster's native CRS, so no reprojection, and the hint cannot
    # disagree with the perimeter the user sees drawn on screen.
    from .bcws import _overlay_json_path
    path = _overlay_json_path(state)
    if not path or not os.path.isfile(path):
        return None, ('BCWS perimeters have not been downloaded yet. '
                      'They are fetched at startup; check the server '
                      'log for the [bcws] lines.')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return None, f'Could not read the BCWS overlay data: {exc}'

    rings = data.get('polygons') or []
    if not rings:
        return None, 'No BCWS fire polygons are currently available.'

    try:
        import numpy as np
        from osgeo import ogr, osr

        srs = osr.SpatialReference()
        if proj:
            srs.ImportFromWkt(proj)

        mem_drv = ogr.GetDriverByName('Memory')
        mem_ds = mem_drv.CreateDataSource('bcws_hint')
        layer = mem_ds.CreateLayer('polys', srs, ogr.wkbPolygon)

        # AOI rectangle, for the intersection test.
        x0, y0 = gt[0], gt[3]
        x1 = gt[0] + w * gt[1] + h * gt[2]
        y1 = gt[3] + w * gt[4] + h * gt[5]
        aoi_ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
            aoi_ring.AddPoint_2D(float(x), float(y))
        aoi_ring.CloseRings()
        aoi_poly = ogr.Geometry(ogr.wkbPolygon)
        aoi_poly.AddGeometry(aoi_ring)

        n_used = 0
        for ring in rings:
            if not ring or len(ring) < 3:
                continue
            r = ogr.Geometry(ogr.wkbLinearRing)
            for pt in ring:
                r.AddPoint_2D(float(pt[0]), float(pt[1]))
            r.CloseRings()
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(r)
            if not poly.IsValid():
                poly = poly.Buffer(0)          # repair self-touching rings
            if poly is None or poly.IsEmpty():
                continue
            if not poly.Intersects(aoi_poly):
                continue
            feat = ogr.Feature(layer.GetLayerDefn())
            feat.SetGeometry(poly)
            layer.CreateFeature(feat)
            feat = None
            n_used += 1

        if n_used == 0:
            return None, (
                'No BCWS fire polygon intersects this AOI. The '
                'perimeter layer only covers currently-reported fires, '
                'so a new or unreported fire will not appear in it -- '
                'use "Red wins (post)" instead.')

        # Rasterise to a 1-band float mask, matching what the red-wins
        # path produces so the CLI and previews need no special case.
        drv = gdal.GetDriverByName('ENVI')
        out_ds = drv.Create(out_path, w, h, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(gt)
        if proj:
            out_ds.SetProjection(proj)
        gdal.RasterizeLayer(out_ds, [1], layer, burn_values=[1.0])
        band = out_ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        n_px = int(np.count_nonzero(arr > 0)) if arr is not None else 0
        band.SetDescription('bcws_perimeter')
        band = None
        out_ds = None
        mem_ds = None

        if n_px == 0:
            return None, (
                'BCWS polygons intersect this AOI but covered no '
                'pixels once rasterised -- the overlap is smaller than '
                'one pixel.')

        sys.stderr.write(
            f'[bcws_hint] bcws_perimeter [{src}]: {n_used} polygon(s), '
            f'{n_px} pixel(s) -> {out_path}\n')
        sys.stderr.flush()
        return out_path, None
    except Exception as exc:
        return None, f'Failed to rasterise BCWS perimeters: {exc}'


def warm_product_artifacts(fire: FireInfo, stack_path: str,
                           log=None) -> dict:
    """Render everything a product needs, without switching the fire.

    A built stack is not enough to switch to instantly: the pane wants
    preview PNGs, and the hint layer wants its mask. Producing those
    only on first switch is what made a date change sit behind
    "Rendering preview imagery" and "Computing the hint layer" for
    imagery that had been on disk for hours.

    Doing it here, at build time, means every later switch is a file
    read. Nothing in this function touches ``fire.crop_bin`` or any
    other shared field, so it is safe to run for several products at
    once and while the operator works on a different one.
    """
    out = {'previews': 0, 'hints': 0, 'skipped': 0, 'errors': []}
    if not stack_path or not os.path.isfile(stack_path):
        return out
    key = product_key_for_path(stack_path)
    if not key:
        return out

    # --- previews ----------------------------------------------------
    try:
        from .preview import generate_all_previews
        outdir = os.path.join(fire.cache_dir, f'previews_{key}')
        # Rendered means the post-fire preview exists -- the same test as
        # previews_complete(). "Any PNG" was satisfied by the hint masks
        # written into the same directory, so a stash whose preview
        # render had been lost was skipped for ever and reported as a
        # failed render on every retry.
        have = os.path.isfile(os.path.join(outdir, 'post.png'))
        if have and not preview_dir_grid_ok(fire, outdir):
            # Rendered on a grid this fire no longer has: never shown,
            # so re-rendered from this product's stack.
            sys.stderr.write(
                f'[warm] {fire.fire_numbe}: previews_{key} is not on '
                f'the fire\'s grid; re-rendering\n')
            from .preview_fs import rmtree as _pf_rmtree
            _pf_rmtree(outdir)
            have = False
        if not have:
            note_product_state(fire.fire_numbe, key, 'rendering',
                               'post-fire, pre-fire and difference',
                               frac=0.0, stage='previews')
        if have:
            out['skipped'] += 1
        else:
            os.makedirs(outdir, exist_ok=True)
            views = generate_all_previews(stack_path, fire.cache_dir,
                                          fire.fire_numbe,
                                          preview_dir=outdir)
            out['previews'] = len(views or [])
            if not views:
                out['errors'].append('no preview views were rendered')
            try:
                with open(os.path.join(outdir, '.product'), 'w',
                          encoding='utf-8') as f:
                    f.write(key)
            except OSError:
                pass
    except Exception as exc:
        out['errors'].append(f'previews: {exc}')
        sys.stderr.write(
            f'[warm] {fire.fire_numbe}: previews for {key}: {exc}\n')

    # --- hint masks --------------------------------------------------
    #
    # Only the red-wins modes: the BCWS perimeter hint is derived from
    # the incident polygon rather than the imagery, and the VIIRS hint
    # does not vary by product.
    # Every hint mode the fire can offer -- VIIRS and BCWS too, not just
    # the two red-wins rules -- so a pane showing "Hint mask" finds this
    # product's mask already made, whichever mode is selected. A mode
    # already made, or recorded as impossible, is skipped.
    _outdir = os.path.join(fire.cache_dir, f'previews_{key}')
    _modes = available_hint_modes(fire)
    _labels = {'redwins_post': 'red-wins, post-fire',
               'redwins_diff': 'red-wins, difference',
               'viirs': 'VIIRS hotspots',
               'bcws_perimeter': 'BCWS perimeters'}
    for _mi, mode in enumerate(_modes):
        if (os.path.isfile(os.path.join(_outdir, f'hintmask_{mode}.png'))
                or hint_mask_problem(_outdir, mode)):
            continue
        note_product_state(fire.fire_numbe, key, 'rendering',
                           _labels.get(mode, mode),
                           frac=_mi / float(len(_modes)), stage='hint')
        if mode not in ('redwins_post', 'redwins_diff'):
            # Not a red-wins rule: the mask comes straight from the
            # fire's VIIRS raster or the BCWS polygons.
            try:
                if os.path.isdir(_outdir) and render_hint_mask_for_product(
                        fire, mode, stack_path, _outdir):
                    out['hints'] += 1
                else:
                    _pr = hint_mask_problem(_outdir, mode)
                    out['errors'].append(
                        f'{mode}: {_pr[0] if _pr else "no mask"}')
            except Exception as _vexc:
                out['errors'].append(f'{mode}: {_vexc}')
            continue
        try:
            path, err = build_redwins_hint_for_fire(
                fire, mode, stack_path=stack_path)
            if err:
                out['errors'].append(f'{mode}: {err}')
                if os.path.isdir(_outdir):
                    _record_hint_problem(_outdir, mode, err, permanent=True)
            elif path:
                out['hints'] += 1
                # The mask layer too, while the derived hint is warm.
                #
                # It is the artifact the pane actually draws now, and
                # making it here means selecting a hint mode is a
                # cached image swap rather than a render. Cheap: the
                # expensive part -- deriving the mask raster -- has
                # just been done and is reused.
                try:
                    _pd = os.path.join(fire.cache_dir,
                                       f'previews_{key}')
                    if os.path.isdir(_pd):
                        render_hint_mask_for_product(
                            fire, mode, stack_path, _pd)
                except Exception as _mexc:
                    sys.stderr.write(
                        f'[warm] {fire.fire_numbe}: hint mask {mode} '
                        f'for {key}: {_mexc}\n')
        except Exception as exc:
            out['errors'].append(f'{mode}: {exc}')
            sys.stderr.write(
                f'[warm] {fire.fire_numbe}: {mode} for {key}: '
                f'{exc}\n')

    # Record what exists now, so deletion never has to guess.
    #
    # The manifest is the authoritative list of this fire's files; a
    # filename pattern is not, because names and identity hashes are
    # shared by more than one thing.
    try:
        from . import manifest as _mf
        _items = []
        for _sfx in ('.bin', '.hdr', '_dates.json', '_overlays.json'):
            _p = os.path.splitext(stack_path)[0] + _sfx
            if os.path.isfile(_p):
                _items.append((_mf.KIND_STACK, _p, key))
        _pd = os.path.join(fire.cache_dir, f'previews_{key}')
        if os.path.isdir(_pd):
            _items.append((_mf.KIND_PREVIEW, _pd, key))
        _rw = os.path.join(fire.cache_dir, '_redwins')
        if os.path.isdir(_rw):
            import glob as _g2
            for _h in _g2.glob(os.path.join(_rw, f'*_{key}_hint.*')):
                _items.append((_mf.KIND_HINT, _h, key))
        _cv = os.path.join(fire.cache_dir, 'coverage',
                           f'{key}_dates.json')
        if os.path.isfile(_cv):
            _items.append((_mf.KIND_COVERAGE, _cv, key))
        if _items:
            _mf.record_many(fire, _items)
    except Exception as _mexc:
        sys.stderr.write(
            f'[manifest] {fire.fire_numbe}: record failed: {_mexc}\n')

    msg = (f'[warm] {fire.fire_numbe}: {key} ready to switch '
           f'({out["previews"]} preview(s), {out["hints"]} hint(s)'
           + (f', {len(out["errors"])} error(s)' if out['errors'] else '')
           + ')')
    sys.stderr.write(msg + '\n')
    if log:
        log(msg)
    return out


def build_redwins_hint_for_fire(fire: FireInfo, mode: str,
                                stack_path: str = ''):
    """Generate the red-wins hint for *mode*.

    Works against ``fire.crop_bin`` by default, or against
    *stack_path* when given -- which lets a product's hint be computed
    at BUILD time, without the fire being switched to it. Computing it
    on first switch instead is what put "Computing the hint layer" in
    front of an operator who had merely changed date.

    Returns ``(path, None)`` on success or ``(None, error_message)``.

    Shared by :func:`switch_hint_mode` (user picks a hint mode) and by
    the re-prepare path (crop changed, so the mask must be rebuilt to
    match the new crop's dimensions -- a hint raster generated for a
    different crop would not align with it).
    """
    if mode not in ('redwins_post', 'redwins_diff'):
        return None, f'Not a red-wins mode: {mode}'
    _crop = stack_path or fire.crop_bin
    if not _crop or not os.path.isfile(_crop):
        return None, 'Fire has no crop raster.'

    band_names = parse_envi_band_names(_crop)
    if not band_names:
        ds = gdal.Open(_crop, gdal.GA_ReadOnly)
        if ds:
            try:
                n = ds.RasterCount
            finally:
                ds = None
            band_names = [f'band {i + 1}' for i in range(n)]
    groups = detect_band_groups(band_names)

    if mode == 'redwins_post':
        indices = groups.get('post', [])
    else:
        indices = groups.get('diff1', [])

    if len(indices) < 3:
        return None, (f'Not enough bands for {mode} '
                      f'(need 3, found {len(indices)}).')

    out_dir = os.path.join(fire.cache_dir, '_redwins')
    os.makedirs(out_dir, exist_ok=True)
    # Per SOURCE as well as per mode. Both stacks previously wrote to
    # <mode>_hint.bin, so switching post source silently overwrote the
    # other source's mask -- and because fire.hint_bin is what the
    # mapping CLI consumes, a run could be seeded with the wrong
    # source's hint. The rendered PNGs hid this (they are stashed per
    # source), so it would only have shown up in the mapping result.
    src = getattr(fire, 'post_source', 'l2') or 'l2'
    # Keyed by PRODUCT, not just source.
    #
    # Every L2 composite shared one filename, so the hint built for the
    # most-recent product was reused for a dated one and vice versa. On
    # a cloudy composite red-wins finds almost nothing, so selecting a
    # clear dated product showed an empty hint (and fed that emptiness
    # to the clustering), while red-wins-diff showed the other
    # product's features. The date is part of what the mask is derived
    # from, so it has to be part of its name.
    # Key the hint by the PRODUCT the stack actually is.
    #
    # product_key(src, start_date) returns a bare 'l2' for a default
    # build, while every other part of the app identifies that same
    # product as 'l2_d<newest-date>'. The hint was therefore written
    # under one name and looked for under another, so it was recomputed
    # on every switch -- the "Computing the hint layer" that appeared
    # even for products prepared hours earlier.
    _pkey = (product_key_for_path(_crop)
             or product_key(src, getattr(fire, 'l2_start_date', '') or ''))
    out_path = os.path.join(out_dir, f'{mode}_{_pkey}_hint.bin')

    # Reuse an existing mask when it is newer than the stack it was
    # derived from. Six call sites reach this function and several run
    # back-to-back during preparation, so without this the same mask is
    # recomputed repeatedly (visible in the log as the identical
    # "[redwins] redwins_post: 104792 fire pixel(s)" line twice).
    #
    # The stack's mtime is the correct invalidation signal: switching
    # post source repoints crop_bin at a different file, and a
    # re-prepare rewrites it, so either genuinely forces a rebuild.
    try:
        if (os.path.isfile(out_path)
                and os.path.getmtime(out_path)
                >= os.path.getmtime(_crop)):
            return out_path, None
    except OSError:
        pass

    n_fire = generate_redwins_hint(_crop, indices, out_path)
    if n_fire < 0:
        return None, f'Failed to generate {mode} hint mask.'
    if n_fire == 0:
        # A valid-but-empty mask would be rejected by the mapping CLI
        # (or silently produce nothing), so fail here with a message
        # that says which rule came up empty and what to try instead.
        other = ('redwins_diff' if mode == 'redwins_post'
                 else 'redwins_post')
        return None, (
            f'The {mode} rule matched no pixels in this crop -- the '
            f'first of its three bands never exceeded the other two, '
            f'so the hint would be empty. Try {other} instead, or use '
            f'VIIRS if data is available for this fire.')
    sys.stderr.write(
        f'[redwins] {mode} [{src}]: {n_fire} fire pixel(s) '
        f'-> {out_path}\n')
    sys.stderr.flush()
    return out_path, None


# switch_post_source() swaps the contents of <cache>/previews, and the
# background prebuild calls it too. Without a lock a user switch and the
# prebuild can interleave and leave previews/ holding a mix of both
# sources' images. One lock per fire keeps each fire's swap atomic while
# letting different fires proceed in parallel.
_SOURCE_SWITCH_LOCKS = {}
_SOURCE_SWITCH_LOCKS_GUARD = threading.Lock()


def _source_switch_lock(fire_numbe: str) -> threading.Lock:
    with _SOURCE_SWITCH_LOCKS_GUARD:
        lk = _SOURCE_SWITCH_LOCKS.get(fire_numbe)
        if lk is None:
            lk = threading.Lock()
            _SOURCE_SWITCH_LOCKS[fire_numbe] = lk
        return lk


def product_key_for_path(path: str) -> str:
    """Identity of the product a stack FILE represents.

    Stack names are ``<postdate>_stack_<name>_<hash>[_l2[_d<start>]].bin``
    where <postdate> is the imagery the composite was built from. That
    date is what makes last night's product different from tonight's,
    so it is part of the identity:

        20260823_stack_F_h.bin              -> mrap_p20260823
        20260823_stack_F_h_l2.bin           -> l2_p20260823
        20260805_stack_F_h_l2_d20260805.bin -> l2_d20260805

    Without the post date every nightly rebuild collided with the one
    before it -- same key, same preview stash, same cache entry -- so
    yesterday's imagery was unreachable the moment today's arrived.

    A start-date L2 build keeps its start-date key: it is pinned to
    that acquisition window and does not change when new imagery
    lands, so it needs no post date to stay distinct.
    """
    base = os.path.basename(path or '')
    m = re.match(
        r'^(?P<post>\d{8})_stack_.+?_[0-9a-fA-F]{6,}'
        r'(?P<l2>_l2(_d(?P<start>\d{8}))?)?\.bin$', base)
    if not m:
        return ''
    if m.group('l2'):
        # ONE identity for L2, however it was built.
        #
        # A default build is "newest-first from whatever exists", which
        # is the same thing as "start from the newest date available".
        # They were keyed differently -- l2_p<newest> for the automatic
        # one, l2_d<start> for a chosen one -- so on a day when the
        # newest acquisition WAS the chosen date the selector listed
        # the same imagery twice under two names. The start date is the
        # identity; for a default build that is its newest acquisition,
        # which is exactly what the filename prefix records.
        return f"l2_d{m.group('start') or m.group('post')}"
    return f"mrap_p{m.group('post')}"


def product_label(key: str) -> str:
    """Human name for a product key, as the selectors show it."""
    m = re.fullmatch(r'(mrap|l2)_p(\d{8})', key or '')
    if m:
        return (('MRAP composite ' if m.group(1) == 'mrap'
                 else 'L2 recent ') + m.group(2))
    m = re.fullmatch(r'l2_d(\d{8})', key or '')
    if m:
        # Same product, same name. Whether the operator chose this
        # start date or the builder took the newest available, the
        # result is an L2 composite starting from that date.
        return f'L2 recent {m.group(1)}'
    return 'MRAP composite' if key == 'mrap' else 'L2 recent tile'


def product_parts(key: str):
    """('mrap'|'l2', l2_start_date, post_date) for a product key."""
    m = re.fullmatch(r'(mrap|l2)_p(\d{8})', key or '')
    if m:
        return m.group(1), '', m.group(2)
    m = re.fullmatch(r'l2_d(\d{8})', key or '')
    if m:
        return 'l2', m.group(1), ''
    return ('mrap' if key == 'mrap' else 'l2'), '', ''


def product_key(source: str, l2_date: str = '') -> str:
    """Identity of one displayable product: source AND, for L2, date.

    'l2' and 'mrap' were enough while a fire had one L2 composite at a
    time. Now that several dated L2 products coexist and can be shown
    side by side, the date is part of what distinguishes one product
    from another -- so it belongs in the stash name, the cache key and
    the preview URL, all of which derive from here.
    """
    source = (source or 'l2').lower()
    if source != 'l2':
        return source
    d = (l2_date or '').strip()
    return f'l2_d{d}' if d else 'l2'


def parse_product_key(key: str):
    """'l2_d20260805' -> ('l2', '20260805'); 'mrap' -> ('mrap', '')."""
    key = (key or 'l2').strip()
    m = re.fullmatch(r'l2_d(\d{8})', key)
    if m:
        return 'l2', m.group(1)
    return ('mrap' if key == 'mrap' else 'l2'), ''


def stamp_previews_product(fire: FireInfo, path: str = None) -> None:
    """Record which product the live previews were rendered from.

    ``previews/`` is a single directory reused by every product, so its
    contents can only be identified by remembering what wrote them. A
    switch that repoints crop_bin but fails to re-render leaves last
    product's pictures in place, and everything downstream then agrees
    they belong to the new one -- the stack says MRAP, the previews are
    L2, and nothing can tell. This marker makes that detectable.
    """
    try:
        key = product_key_for_path(
            path or getattr(fire, 'crop_bin', '') or '')
        d = os.path.join(fire.cache_dir, 'previews')
        if not key or not os.path.isdir(d):
            return
        with open(os.path.join(d, '.product'), 'w',
                  encoding='utf-8') as f:
            f.write(key)
    except OSError as exc:
        sys.stderr.write(f'[prepare] could not stamp previews: '
                         f'{exc}\n')


def previews_product(cache_dir: str) -> str:
    """Which product a preview directory holds, or '' if unmarked."""
    try:
        with open(os.path.join(cache_dir, '.product'),
                  encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return ''


def _preview_stash_dir(fire: FireInfo, source: str = None,
                       l2_date: str = None, path: str = None) -> str:
    """Per-PRODUCT copy of the rendered previews.

    generate_all_previews() always writes to ``<cache>/previews``, so
    products overwrite each other there. Without a stash, every switch
    re-renders every preview even though the stack is already cached --
    which is what made switching slow. A copy per product turns a
    switch into a handful of file copies.

    ``l2_date`` defaults to the fire's current date, which keeps every
    existing caller correct: they stash the product that is loaded.
    """
    # Prefer the identity of the STACK FILE the previews came from.
    #
    # (source, date) cannot tell last night's MRAP composite from
    # tonight's, so both wrote to previews_mrap and the newer render
    # overwrote the older -- stomping the very product the operator
    # wants to go back to.
    if path is None:
        path = getattr(fire, 'crop_bin', '') or ''
    key = product_key_for_path(path) if path else ''
    if not key:
        if l2_date is None:
            l2_date = getattr(fire, 'l2_start_date', '') or ''
        key = product_key(source or getattr(fire, 'post_source', 'l2'),
                          l2_date)
    return os.path.join(fire.cache_dir, f'previews_{key}')


def _stash_previews(fire: FireInfo, source: str,
                    l2_date: str = None, path: str = None) -> None:
    src = os.path.join(fire.cache_dir, 'previews')
    if not os.path.isdir(src):
        return
    dst = _preview_stash_dir(fire, source, l2_date, path=path)

    # The stamp decides where these images may be filed, not the
    # caller.
    #
    # This copy is DESTRUCTIVE -- it removes the destination first --
    # so filing it under the wrong key does not merely add a wrong
    # copy, it destroys the right one. That is what happened: a
    # background render finished holding one product's images while
    # the fire had already been switched to another, the destination
    # was resolved from crop_bin, and a September product ended up
    # showing August imagery with no error reported anywhere.
    #
    # previews/.product records which product actually rendered the
    # images now sitting in that directory. If it disagrees with where
    # they are about to be filed, the copy is refused. Refusing costs
    # one re-render; proceeding silently corrupts a product.
    #
    # The check and the copy happen under ONE hold of the fire's preview
    # lock: checked outside it, previews/ could change product between
    # the check and the copy (see preview_fs).
    from .preview_fs import lock_for, replace_tree
    with lock_for(src):
        if not os.path.isdir(src):
            return
        stamped = previews_product(src)
        want = os.path.basename(dst)
        want = (want[len('previews_'):] if want.startswith('previews_')
                else '')
        if stamped and want and stamped != want:
            sys.stderr.write(
                f'[prepare] {getattr(fire, "fire_numbe", "?")}: REFUSING '
                f'to stash previews rendered from {stamped} under {want} '
                f'-- the live previews belong to a different product\n')
            return
        try:
            replace_tree(src, dst)
        except OSError as exc:
            sys.stderr.write(f'[prepare] preview stash failed: {exc}\n')


def _restore_previews(fire: FireInfo, source: str,
                      path: str = None) -> bool:
    """Put *source*'s stashed previews back in place. True if restored.

    Refuses a stash older than the stack it came from: a re-prepare can
    resize the crop, which makes every stashed PNG the wrong dimensions
    and would misregister the vector overlays drawn on top of them.
    """
    src = _preview_stash_dir(fire, source, path=path)
    if not os.path.isdir(src):
        return False
    try:
        # NOTE: no mtime comparison against the stack.
        #
        # It used to refuse any stash older than the stack file, which
        # is wrong now that products are switched rather than rebuilt:
        # returning to a composite touches its stack, so every stash
        # made before that moment looked stale and every switch back
        # re-rendered from scratch. The grid check below answers the
        # real question -- does this stash describe THIS product's
        # grid -- and answers it exactly.
        # The mtime test is circumstantial; the GRID test is decisive.
        # A stash whose recorded geotransform does not match the stack
        # being switched to would put a different extent on screen --
        # the shorter, diagonally-shifted pane -- so it is deleted and
        # re-rendered rather than restored. Once. After that the stash
        # carries the current grid for ever.
        try:
            import json as _json
            from osgeo import gdal
            gj = os.path.join(src, 'geo.json')
            entry = None
            if os.path.isfile(gj):
                with open(gj, encoding='utf-8') as fh:
                    entry = (_json.load(fh) or {}).get('post')
            ds = gdal.Open(fire.crop_bin, gdal.GA_ReadOnly)
            if ds is not None:
                rw, rh = ds.RasterXSize, ds.RasterYSize
                gt = ds.GetGeoTransform()
                ds = None
                ok = bool(entry) and (
                    int(entry.get('rw', -1)) == rw
                    and int(entry.get('rh', -1)) == rh
                    and all(abs(float(a) - float(b)) < 1e-6
                            for a, b in zip(entry.get('gt', []), gt)))
                if not ok:
                    sys.stderr.write(
                        f'[prepare] previews_{source} stash is on a '
                        f'different grid (or predates geo recording); '
                        f'deleting it and re-rendering\n')
                    from .preview_fs import rmtree as _pf_rmtree
                    _pf_rmtree(src)
                    return False
        except Exception as exc:
            sys.stderr.write(f'[prepare] stash grid check skipped: '
                             f'{exc}\n')
        dst = os.path.join(fire.cache_dir, 'previews')
        # Replaced under the fire's preview lock; a render still
        # claiming previews/ loses its claim (see preview_fs).
        from .preview_fs import replace_tree
        replace_tree(src, dst)
        return True
    except OSError as exc:
        sys.stderr.write(f'[prepare] preview restore failed: {exc}\n')
        return False


def prebuild_other_source(fire: FireInfo) -> None:
    """Build the post source the user is NOT currently viewing.

    Run in the background right after a fire becomes READY so the first
    toggle is instant instead of paying for a full stack build and
    preview render. Purely opportunistic: any failure is logged and
    dropped, because the on-demand path in switch_post_source() still
    works.
    """
    other = 'mrap' if getattr(fire, 'post_source', 'l2') == 'l2' else 'l2'
    if os.path.isdir(_preview_stash_dir(fire, other)):
        return
    current = getattr(fire, 'post_source', 'l2')

    # The prebuild temporarily flips fire.post_source to the other
    # source and back. Anything reading it during that window -- most
    # importantly /prepare, which the page calls on open -- saw the
    # WRONG source and rendered the wrong default in the dropdown, and
    # could be served previews mid-rewrite (the truncated image).
    #
    # user_post_source records what the USER is on. It never changes
    # during a prebuild, so readers have a stable answer no matter when
    # they land.
    fire.user_post_source = current
    fire.prebuilding = True
    try:
        fire.console_log.append(
            f'  Pre-building the {other.upper()} stack in the '
            f'background so switching is instant ...')
        res = switch_post_source(fire, other)
        if res.get('ok'):
            # Switch back so the user keeps the source they were on;
            # the stash built above makes this second switch cheap.
            switch_post_source(fire, current)
            fire.console_log.append(
                f'  {other.upper()} stack ready -- switching is now '
                f'instant.')
        else:
            fire.console_log.append(
                f'  Pre-build of {other.upper()} failed: '
                f'{res.get("error", "unknown")}')
    except Exception as exc:
        sys.stderr.write(f'[prepare] prebuild failed: {exc}\n')
    finally:
        # Always restore, even if the switch raised: leaving the fire
        # on the other source would change what the user sees.
        fire.prebuilding = False
        if getattr(fire, 'post_source', current) != current:
            try:
                switch_post_source(fire, current)
            except Exception:
                fire.post_source = current
        fire.user_post_source = current


def render_hint_for_product(fire: FireInfo, mode: str,
                            stack_path: str, preview_dir: str) -> bool:
    """Render one product's hint image WITHOUT switching the fire.

    The old route rendered a hint by switching the fire onto the
    product, rendering into the live previews directory, and switching
    back. A display request thus mutated server state twice, and if the
    second switch lost the fire lock -- which happens under any
    concurrency -- the fire was left on the other product and every
    later render used its imagery. That is how the picture behind a
    hint became a different source while the selector still named the
    one that had been chosen.

    Everything needed is already per product: the mask builder takes
    the stack to read, and the compositor takes the directory holding
    that product's post.png. Nothing here touches fire.crop_bin.
    """
    if not stack_path or not os.path.isfile(stack_path):
        return False
    if not preview_dir or not os.path.isdir(preview_dir):
        return False
    out = os.path.join(preview_dir, f'hint_{mode}.png')

    if mode == 'viirs':
        # VIIRS detections are a property of the fire, not of a
        # product, so the same mask serves every product.
        mask = fire.viirs_bin
        if not mask or not os.path.isfile(mask):
            return False
    else:
        mask, err = build_derived_hint_for_fire(fire, mode,
                                                stack_path=stack_path)
        if mask and getattr(fire, 'restrict_hint_bcws', False):
            mask = restrict_hint_to_bcws(fire, mask)
        if not mask:
            sys.stderr.write(
                f'[prepare] hint {mode} for '
                f'{os.path.basename(stack_path)}: {err}\n')
            return False

    try:
        _overlay_mask_on_post(fire, mask, f'hint_{mode}',
                              (0.0, 0.8, 0.2),
                              preview_dir=preview_dir)
    except Exception as exc:
        sys.stderr.write(
            f'[prepare] hint overlay {mode} failed: {exc}\n')
        return False
    ok = os.path.isfile(out)
    if ok:
        sys.stderr.write(
            '[prepare] %s: rendered hint %s for %s (no switch)\n'
            % (fire.fire_numbe, mode, os.path.basename(preview_dir)))
    return ok


def render_hint_mask_for_product(fire: FireInfo, mode: str,
                                 stack_path: str,
                                 preview_dir: str) -> bool:
    """Write hintmask_<mode>.png: the mask alone, transparent elsewhere.

    The composited hint (hint_<mode>.png) stays exactly as it was and
    remains the fallback. This is the cheaper artifact: the browser
    already holds the product's post-fire image, so sending only the
    mask lets it draw the two as layers, and switching hint modes
    becomes an image swap instead of a server-side composite per mode.
    """
    if not stack_path or not os.path.isfile(stack_path):
        return False
    if not preview_dir or not os.path.isdir(preview_dir):
        return False
    out = os.path.join(preview_dir, f'hintmask_{mode}.png')

    if mode == 'viirs':
        mask = fire.viirs_bin
        if not mask or not os.path.isfile(mask):
            _record_hint_problem(preview_dir, mode,
                                 'there is no VIIRS hotspot data for this '
                                 'fire', permanent=True)
            return False
    else:
        mask, err = build_derived_hint_for_fire(fire, mode,
                                                stack_path=stack_path)
        if mask and getattr(fire, 'restrict_hint_bcws', False):
            mask = restrict_hint_to_bcws(fire, mask)
        if not mask:
            sys.stderr.write(
                f'[prepare] hint mask {mode} for '
                f'{os.path.basename(stack_path)}: {err}\n')
            _record_hint_problem(preview_dir, mode,
                                 err or 'the hint could not be derived',
                                 permanent=True)
            return False
    # A mask with no pixel set would be drawn as a fully transparent
    # layer: "Hint mask" selected, nothing green, and no word why. That
    # happens when an earlier derivation found nothing and a later call
    # reuses its raster. Say so instead of drawing an empty layer.
    try:
        import numpy as _np
        _mds = gdal.Open(mask, gdal.GA_ReadOnly)
        _any = True
        if _mds is not None:
            _arr = _mds.GetRasterBand(1).ReadAsArray()
            _any = bool(_np.any(_np.nan_to_num(_arr) > 0))
        _mds = None
    except Exception:
        _any = True                      # cannot judge: draw it as before
    if not _any:
        _record_hint_problem(preview_dir, mode,
                             'the hint mask is empty for this product: '
                             'no pixel matched', permanent=True)
        return False
    try:
        _overlay_mask_on_post(fire, mask, f'hintmask_{mode}',
                              (0.0, 0.8, 0.2),
                              preview_dir=preview_dir, mask_only=True)
    except Exception as exc:
        sys.stderr.write(
            f'[prepare] hint mask {mode} failed: {exc}\n')
        _record_hint_problem(preview_dir, mode, f'rendering failed: {exc}',
                             permanent=False)
        return False
    ok = os.path.isfile(out)
    if ok:
        try:
            os.remove(_hint_marker(preview_dir, mode))
        except OSError:
            pass
    return ok


def render_hint_for_mode(fire: FireInfo, mode: str) -> bool:
    """Render previews/hint_<mode>.png for the CURRENT post source.

    Each (post source x hint mode) pair is a distinct image, because the
    red-wins rule reads the stack's post bands (mode 'redwins_post') or
    its anomaly bands (mode 'redwins_diff'), and those bands differ
    between the L2 and MRAP stacks. Storing them under one filename is
    what made the masks appear identical.
    """
    if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
        return False
    out = os.path.join(fire.cache_dir, 'previews', f'hint_{mode}.png')

    if mode == 'viirs':
        mask = fire.viirs_bin
        if not mask or not os.path.isfile(mask):
            return False
    else:
        mask, err = build_derived_hint_for_fire(fire, mode)
        if mask and getattr(fire, 'restrict_hint_bcws', False):
            # Clip the chosen hint to the BCWS perimeter, so
            # the preview, the agreement score and the
            # clustering all use the same restricted mask.
            mask = restrict_hint_to_bcws(fire, mask)
        if not mask:
            sys.stderr.write(
                f'[prepare] hint {mode}: {err}\n')
            return False

    # _overlay_mask_on_post writes previews/<name>.png, so render under
    # the per-mode name directly.
    try:
        _overlay_mask_on_post(fire, mask, f'hint_{mode}', (0.0, 0.8, 0.2))
    except Exception as exc:
        sys.stderr.write(f'[prepare] hint overlay {mode} failed: {exc}\n')
        return False
    return os.path.isfile(out)


def pregenerate_all_hints(fire: FireInfo) -> list:
    """Render every hint mode available for the current post source.

    Called after a source's stack is built so that switching hint modes
    is a cache hit rather than a render. Returns the modes rendered.
    """
    done = []
    modes = list(DERIVED_HINT_MODES)
    if fire.viirs_bin and os.path.isfile(fire.viirs_bin):
        modes.append('viirs')
    for m in modes:
        try:
            if render_hint_for_mode(fire, m):
                done.append(m)
        except Exception as exc:
            sys.stderr.write(
                f'[prepare] pregenerate hint {m} failed: {exc}\n')
    src = getattr(fire, 'post_source', 'l2')
    fire.console_log.append(
        f'  Pre-rendered hint mask(s) for {src.upper()}: '
        f'{", ".join(done) if done else "none"}')
    return done


def set_user_post_source(fire: FireInfo, source: str) -> None:
    """Record a source the USER chose (as opposed to a prebuild)."""
    fire.user_post_source = source


def _l2_selection_is_current(fire, source: str) -> bool:
    """Is the product the fire is USING already the requested one?

    The question is about what is loaded, not what exists on disk.
    Testing existence was wrong in the one case that matters: switching
    back to a date whose product had already been built returned "there
    is nothing to do" while crop_bin still pointed at a different
    composite -- so the visualisation, and the clustering input, stayed
    on the wrong date.

    The loaded stack's own path carries the answer: it ends in
    ``_l2_d<YYYYMMDD>.bin`` for a dated product and ``_l2.bin`` for the
    default (most recent).
    """
    if source != 'l2':
        return True
    cur = getattr(fire, 'crop_bin', '') or ''
    if not cur or not os.path.isfile(cur):
        return False
    # The REQUESTED date, which during a switch is the pending one.
    #
    # switch_post_source() records the caller's date in
    # _pending_l2_date and deliberately does not apply it until inside
    # the lock (so the outgoing product can be stashed under its own
    # name). This check runs BEFORE that, so reading l2_start_date
    # compared the loaded date against itself: always equal, always
    # "nothing to do". The switch returned success instantly, having
    # done nothing, and the old composite stayed on screen -- which is
    # exactly why selecting a dated product changed nothing.
    _pend = getattr(fire, '_pending_l2_date', None)
    want = (getattr(fire, 'l2_start_date', '') or '') if _pend is None \
        else (_pend or '')
    m = re.search(r'_l2_d(\d{8})\.bin$', cur)
    if m:
        have = m.group(1)
    elif cur.endswith('_l2.bin'):
        have = ''                     # the default, most-recent product
    else:
        return False                  # not an L2 stack at all
    if have != want:
        sys.stderr.write(
            f'[switch] L2 date differs: loaded '
            f'{have or "most-recent"}, requested '
            f'{want or "most-recent"} -- rebuilding/repointing\n')
        return False
    return True


def stack_path_for_product(fire, key: str) -> str:
    """Existing stack file for a product key, or '' if not built."""
    import glob as _g
    cb = getattr(fire, 'crop_bin', '') or ''
    ram = os.path.dirname(cb) or '/ram'
    base = os.path.basename(cb)
    m = re.match(r'^\d{8}_stack_(?P<safe>.+?)_(?P<h>[0-9a-fA-F]{6,})'
                 r'(_l2(_d\d{8})?)?\.bin$', base)
    if not m:
        return ''
    pat = os.path.join(
        ram, f'*_stack_{m.group("safe")}_{m.group("h")}*.bin')
    for cand in sorted(_g.glob(pat), reverse=True):
        if product_key_for_path(cand) == key:
            return cand

    # Not on the ramdisk -- try the durable store, and bring it back.
    #
    # Rebuilding is not an alternative here. A stack's filename carries
    # the POST date it was built against ('20260913_stack_..._l2_d
    # 20260901.bin'), and a rebuild today would compute a different
    # prefix -- so the restore-by-name path inside ensure_aoi_stack
    # cannot find it, and for a dated MRAP whose mosaic has rolled off
    # there is nothing to rebuild from at all. Copying the stored file
    # back under its own name is both faster and the only thing that
    # works.
    try:
        from .durable import durable_products
        import shutil as _sh
        for cand in durable_products(fire):
            if product_key_for_path(cand) != key:
                continue
            dst = os.path.join(ram, os.path.basename(cand))
            stem_s = os.path.splitext(cand)[0]
            stem_d = os.path.splitext(dst)[0]
            for sfx in ('.bin', '.hdr', '_dates.json',
                        '_overlays.json'):
                s_p, d_p = stem_s + sfx, stem_d + sfx
                if os.path.isfile(s_p) and not os.path.isfile(d_p):
                    tmp = d_p + '.part'
                    _sh.copy2(s_p, tmp)
                    os.replace(tmp, d_p)
            if os.path.isfile(dst):
                sys.stderr.write(
                    '[persist] %s: %s restored from the durable store '
                    'on selection (%s)\n'
                    % (getattr(fire, 'fire_numbe', '?'), key,
                       os.path.basename(dst)))
                return dst
    except Exception as exc:
        sys.stderr.write(
            f'[persist] durable lookup for {key} failed: {exc}\n')
    return ''


def switch_post_source(fire: FireInfo, source: str,
                       l2_date: str = None,
                       product: str = None) -> dict:
    """Switch the fire to a product: a source, and for L2 a date.

    ``l2_date`` of None means "keep the fire's current date", which is
    what every pre-existing caller wants. An explicit date (or '' for
    the most-recent composite) selects a specific L2 product, so the
    left-pane selector can choose one directly.
    """
    # A product key names one specific composite. If its file exists,
    # the locked switch repoints instead of rebuilding -- which is the
    # only way to reach an older MRAP mosaic, since the builder always
    # takes the newest.
    if product:
        _psrc, _pstart, _ppost = product_parts(product)
        source = _psrc
        existing = stack_path_for_product(fire, product)
        if existing:
            fire._pending_product_path = existing
            fire._pending_l2_date = _pstart
        else:
            # Not built: build it. For L2 the key's start date decides
            # the composite; for MRAP the key's date names which
            # province-wide mosaic to clip, so an earlier day can be
            # produced rather than silently getting the newest.
            fire._pending_l2_date = _pstart
            if _psrc == 'mrap' and _ppost:
                fire._pending_mrap_date = _ppost
        l2_date = None          # the key has already decided

    if l2_date is not None and (source or '').lower() == 'l2':
        # Only RECORD the request here. Applying it now would corrupt
        # the outgoing product's identity: the locked switch stashes
        # the current previews under the date they actually describe,
        # and it has to read that date before the new one lands.
        fire._pending_l2_date = l2_date or ''

    if source not in ('l2', 'mrap'):
        return {'ok': False, 'error': f'Unknown post source: {source}'}
    if not getattr(fire, 'bbox_native', None):
        return {'ok': False, 'error': 'Fire has no bbox on record.'}

    # Refuse rather than queue when a switch for this fire is already
    # running.
    #
    # Blocking on the lock meant a second request sat there until the
    # first finished -- which, on a first-time stack build, can be
    # minutes -- with nothing to distinguish it from a hang. Callers
    # that legitimately want to wait can retry; the UI reports it.
    lk = _source_switch_lock(fire.fire_numbe)
    if not lk.acquire(blocking=False):
        # Say WHAT holds it: on a freshly prepared fire this is almost
        # always the background prebuild of the other source, which the
        # caller should simply wait out.
        why = ('the background prebuild of the other source'
               if getattr(fire, 'prebuilding', False)
               else 'another source switch')
        # Busy: this call does nothing, so its date request dies here
        # rather than being applied by whatever switches next.
        try:
            if hasattr(fire, '_pending_l2_date'):
                del fire._pending_l2_date
        except Exception:
            pass
        return {'ok': False, 'busy': True,
                'holder': ('prebuild'
                           if getattr(fire, 'prebuilding', False)
                           else 'switch'),
                'error': f'Busy: {why} is using this fire. '
                         f'Retrying automatically.'}
    try:
        # Already there: nothing to do, and saying so instantly is much
        # better than repeating the work.
        #
        # "There" now includes the L2 START DATE. Without that check,
        # applying a different date to a fire already on L2 matched this
        # branch and returned success instantly, having built nothing --
        # the menu closed, no progress appeared, and the old composite
        # stayed on screen. The date is part of the product's identity,
        # so it has to be part of this comparison.
        # Compare the PRODUCT, not the source and start date.
        #
        # Neither of those can tell last night's MRAP composite from
        # tonight's, nor two L2-recent products from different nights:
        # for all of them the source matches and the start date is
        # empty, so this branch declared "already there" and returned
        # success having done nothing. The selector snapped back, the
        # imagery never changed, and trying again eventually worked
        # only because something else had moved on in between. That is
        # the whole "takes two or three attempts" symptom.
        _cur_prod = product_key_for_path(
            getattr(fire, 'crop_bin', '') or '')
        _want_prod = ''
        if product:
            _want_prod = product
            if _want_prod in ('mrap', 'l2'):
                # A bare source means "the newest of that source",
                # which is only a no-op if that is what is loaded AND
                # nothing newer has been built since.
                _want_prod = ''
        if _want_prod:
            _same = bool(_cur_prod) and _cur_prod == _want_prod
        else:
            _same = ((getattr(fire, 'post_source', '') or 'l2') == source
                     and _l2_selection_is_current(fire, source))
        if _same:
            prev_dir = os.path.join(fire.cache_dir, 'previews')
            if os.path.isdir(prev_dir) and os.listdir(prev_dir):
                return {'ok': True, 'unchanged': True,
                        'post_source': source,
                        'product_key': _cur_prod}
            sys.stderr.write(
                f'[prepare] {fire.fire_numbe}: on {_cur_prod or source} '
                f'already but its previews are missing; re-rendering\n')
        return _switch_post_source_locked(fire, source)
    finally:
        # The pending date must not outlive this call.
        #
        # The locked switch consumes it, but the short-circuit and
        # busy paths above return without ever reaching it. A leftover
        # value would then be read by the NEXT switch -- including a
        # switch to MRAP -- and silently retarget it at a date the
        # caller never asked for.
        try:
            if hasattr(fire, '_pending_l2_date'):
                del fire._pending_l2_date
        except Exception:
            fire._pending_l2_date = None
        lk.release()


def _switch_post_source_locked(fire: FireInfo, source: str) -> dict:
    from .aoi_stack import ensure_aoi_stack, AoiStackError

    # Capture the OUTGOING product first, then apply any requested
    # date.
    #
    # Order matters twice over: ensure_aoi_stack() below reads
    # fire.l2_start_date to decide WHICH composite to build, so the new
    # date has to be in place before it runs -- but the previews now on
    # disk describe the OLD date, so that identity has to be recorded
    # before it is overwritten. Getting this backwards stashes one
    # product's previews under another product's name, which is
    # unrecoverable once it happens.
    _out_src = getattr(fire, 'post_source', '') or ''
    _out_date = getattr(fire, 'l2_start_date', '') or ''
    # The file the previews on disk were rendered from.
    _out_path = getattr(fire, 'crop_bin', '') or ''
    _pending_req = getattr(fire, '_pending_l2_date', None)
    if _pending_req is not None:
        if (source or '').lower() == 'l2':
            fire.l2_start_date = _pending_req or ''
        try:
            del fire._pending_l2_date
        except AttributeError:
            fire._pending_l2_date = None

    ref_raster = (state.rasters_by_year.get(fire.fire_year)
                  or state.raster_path)

    def _cb(detail, frac):
        set_prep_stage(fire, stage_for_stack_detail(detail),
                       detail=f'{source.upper()}: {detail}', frac=frac)

    # Selecting a product that already exists is a repoint, not a build.
    #
    # An older MRAP composite cannot be rebuilt at all -- the builder
    # always takes the newest mosaic -- so navigating back to last
    # night's imagery is only possible by pointing at the file that is
    # still on disk. Doing the same for any existing product also makes
    # switching between them instant.
    _want_path = getattr(fire, '_pending_product_path', None)
    if _want_path:
        try:
            del fire._pending_product_path
        except AttributeError:
            fire._pending_product_path = None
    if _want_path and os.path.isfile(_want_path):
        sys.stderr.write(
            f'[prepare] {fire.fire_numbe}: repointing to '
            f'{os.path.basename(_want_path)} (already built)\n')
        try:
            _rk = product_key_for_path(_want_path)
            if _rk:
                note_product_state(fire.fire_numbe, _rk, 'repointing',
                                   os.path.basename(_want_path))
        except Exception:
            pass
        info = {'path': _want_path}
        _src2, _start2, _post2 = product_parts(
            product_key_for_path(_want_path))
        fire.post_source = _src2
        fire.l2_start_date = _start2
    else:
        info = None

    try:
        if info is None:
            _mrap_req = getattr(fire, '_pending_mrap_date', '') or ''
            if _mrap_req:
                try:
                    del fire._pending_mrap_date
                except AttributeError:
                    fire._pending_mrap_date = ''
            info = ensure_aoi_stack(
                fire.fire_numbe, fire.bbox_native, progress_cb=_cb,
                mrap_date=_mrap_req,
                instance_key=getattr(state, 'shared_root', '') or '',
                post_source=source, ref_raster=ref_raster,
                log_cb=lambda m: fire.console_log.append(m.rstrip()),
                # Per-date L2 composites: empty means 'most recent',
                # which is the historical behaviour.
                l2_start_date=getattr(fire, 'l2_start_date', ''))
    except AoiStackError as exc:
        fire.progress = {}
        return {'ok': False, 'error': str(exc)}
    except Exception as exc:
        fire.progress = {}
        return {'ok': False, 'error': f'stack build failed: {exc}'}

    fire.progress = {}
    # Snapshot the OUTGOING source's previews before repointing.
    #
    # Only the "no stash existed" branch below stashed anything, so a
    # switch to a source that already had a stash discarded whatever
    # had been produced since the outgoing source was last stashed --
    # including previews/result.png from a mapping or KGC run. The
    # symptom is that "ML classification" silently disappears from the
    # view list after any source switch (and the on-demand stash build
    # in the preview handler performs two switches, so simply opening
    # a fire could do it).
    # Identity captured at the top of this function, before the new
    # date was applied.
    prev_src = _out_src
    prev_date = _out_date
    # Compare the products by the identity of their FILES: two MRAP
    # composites from different nights are different products even
    # though (source, date) says they are the same.
    _prev_key = (product_key_for_path(_out_path)
                 or (product_key(prev_src, prev_date) if prev_src else ''))
    _new_key = ''          # filled in after the stack is resolved
    if prev_src and _out_path:
        try:
            # Always stash the outgoing product. It is keyed by its own
            # file, so this cannot overwrite anything, and it is what
            # makes returning to it a copy rather than a re-render.
            _stash_previews(fire, prev_src, prev_date, path=_out_path)
        except Exception as exc:
            sys.stderr.write(
                f'[prepare] could not stash {prev_src} previews before '
                f'switching to {source}: {exc}\n')

    fire.post_source = source
    fire.crop_bin = info['path']
    fire.crop_w = info.get('width', fire.crop_w)
    fire.crop_h = info.get('height', fire.crop_h)

    # Previews are derived from the stack, so they must match the new
    # post bands. Restore this source's stash when one exists (cheap
    # copies) and only re-render when it does not.
    # A DATE change is an l2 -> l2 switch, and the stash for 'l2' holds
    # the PREVIOUS date's imagery. Restoring it would put the old
    # composite straight back on screen -- which is exactly what
    # happened: the build ran, and the display never changed. Drop the
    # stale stash so the previews are re-rendered from the new stack.
    _new_key = product_key_for_path(getattr(fire, 'crop_bin', '') or '')
    if _prev_key and _new_key and _prev_key != _new_key:
        try:
            # The LIVE previews describe the OUTGOING product, so they
            # cannot stay -- but they are no longer thrown away: they
            # were stashed under that product's own key above, and the
            # incoming product's stash (if it has one) is restored
            # below. Moving between two built products is therefore a
            # file copy, not a re-render.
            live = os.path.join(fire.cache_dir, 'previews')
            if os.path.isdir(live):
                from .preview_fs import rmtree as _pf_rmtree
                _pf_rmtree(live)
                sys.stderr.write(
                    f'[prepare] cleared live previews: moving from '
                    f'{_prev_key} to {_new_key}\n')
        except Exception as exc:
            sys.stderr.write(f'[prepare] live preview clear failed: '
                             f'{exc}\n')

    # Keep the durable copy current. Background: nothing waits on it.
    try:
        from .durable import mirror_in_background
        mirror_in_background()
    except Exception:
        pass

    restored = _restore_previews(fire, source,
                                 path=getattr(fire, 'crop_bin', ''))
    sys.stderr.write(
        '[persist] previews %s: %s\n'
        % (fire.fire_numbe,
           'restored from the stash' if restored
           else 'no stash; rendering'))
    if restored:
        stamp_previews_product(fire)
    if restored:
        try:
            # Only real VIEWS belong in this list. The previews dir
            # also holds the per-mode hint renders (hint_redwins_post
            # .png etc.), which are selected via the Hint buttons and
            # served through ?hint= -- listing them blindly put
            # "hint_redwins_post" into the view dropdown and, because
            # the list is also what the client validates against, left
            # legitimate views looking unavailable ("View 'Post-fire'
            # not available").
            _VIEW_WHITELIST = ('pre', 'post', 'diff1', 'diff2', 'diff3',
                               'hint', 'result', 'result_prebrush')
            names = [os.path.splitext(f)[0]
                     for f in sorted(os.listdir(
                         os.path.join(fire.cache_dir, 'previews')))
                     if f.endswith('.png')]
            fire.available_views = [n for n in names
                                    if n in _VIEW_WHITELIST]
            # A per-mode hint render implies the 'hint' view is usable
            # even if the generic hint.png is absent.
            if ('hint' not in fire.available_views
                    and any(n.startswith('hint_') for n in names)):
                fire.available_views.append('hint')
        except OSError:
            pass
    else:
        try:
            set_prep_stage(fire, 'previews',
                           detail='rendering the display layers',
                           frac=0.1)
            views = generate_all_previews(
                fire.crop_bin, fire.cache_dir, fire.fire_numbe)
            stamp_previews_product(fire)
            try:
                from .mapping import record_base_preview_geo
                record_base_preview_geo(fire.cache_dir, fire.crop_bin)
            except Exception:
                pass
            fire.available_views = views
        except Exception as exc:
            sys.stderr.write(
                f'[prepare] preview regeneration failed after '
                f'post-source switch: {exc}\n')

    # The red-wins hints are computed FROM the stack bands, so a hint
    # built against the old source is stale. Rebuild whichever mode is
    # active against the new bands.
    #
    # If the fire has no usable hint at all (e.g. no VIIRS on disk, or
    # a previous build failed), fall back to red-wins here rather than
    # leaving hint_bin empty -- an empty hint is what makes the fire
    # unmappable and leaves the UI parked on "preparing".
    mode = getattr(fire, 'hint_mode', 'redwins_post') or 'redwins_post'
    if mode == 'viirs' and not (
            fire.viirs_bin and os.path.isfile(fire.viirs_bin)):
        mode = 'redwins_post'
    if mode in DERIVED_HINT_MODES:
        # Reuse this product's hint when it exists, and otherwise get
        # OUT OF THE WAY.
        #
        # Deriving a hint takes seconds to minutes. Doing it inside the
        # switch meant that asking for post-fire imagery -- the thing
        # the operator actually clicked -- waited on a mask they had
        # not asked to see. Stepping through dates felt like building,
        # because it was.
        _hp = derived_hint_path(fire, mode)
        if _hp and os.path.isfile(_hp):
            rw_path, rw_err = _hp, None
            sys.stderr.write(
                '[persist] hint %s: reusing %s\n'
                % (fire.fire_numbe, os.path.basename(_hp)))
        else:
            rw_path, rw_err = None, 'deferred'
            sys.stderr.write(
                '[persist] hint %s: %s not on disk; deriving in the '
                'background (the switch does not wait)\n'
                % (fire.fire_numbe, os.path.basename(_hp or mode)))
            _defer_hint_build(fire, mode)
        if rw_path and getattr(fire, 'restrict_hint_bcws', False):
            # Clip the chosen hint to the BCWS perimeter, so
            # the preview, the agreement score and the
            # clustering all use the same restricted mask.
            rw_path = restrict_hint_to_bcws(fire, rw_path)
        if rw_path:
            fire.hint_bin = rw_path
            fire.perimeter_type = mode
            fire.hint_mode = mode
        elif rw_err != 'deferred':
            sys.stderr.write(
                f'[prepare] red-wins rebuild after source switch '
                f'failed: {rw_err}\n')
    elif fire.viirs_bin and os.path.isfile(fire.viirs_bin):
        fire.hint_bin = fire.viirs_bin
        fire.perimeter_type = 'viirs'

    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        try:
            _overlay_mask_on_post(
                fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))
            if 'hint' not in fire.available_views:
                fire.available_views.append('hint')
        except Exception:
            pass

    # Overlays are cached per crop_bin, and the switch just repointed
    # it at the other source's stack -- so this source needs its own
    # cache entry. Building it here means the background prebuild also
    # warms the overlays for the source the user has not opened yet,
    # instead of that cost landing on the first switch.
    try:
        from .fire_overlays import build_fire_overlays
        build_fire_overlays(state, fire)
    except Exception as exc:
        sys.stderr.write(
            f'[prepare] overlay build after source switch failed: '
            f'{exc}\n')

    # Same grid change as a re-prepare: the crop now points at the
    # other source's stack, so run overlays must be re-rendered onto
    # it or they keep the previous source's extent.
    try:
        from .mapping import rerender_run_overlays
        rerender_run_overlays(
            fire, log=lambda m: fire.console_log.append(m))
    except Exception as _rexc:
        sys.stderr.write(
            f'[prepare] run overlay re-render skipped: {_rexc}\n')

    if not restored:
        # Render every hint mode for this source, but IN THE
        # BACKGROUND. The stash still ends up with all of them; the
        # difference is that the operator's switch returns now rather
        # than after the last mask is derived.
        _defer_pregenerate_hints(fire)
        # Snapshot now that previews/ holds this source's images AND
        # all of its hint overlays, so a later switch back restores
        # everything.
        # Name the product explicitly rather than letting the stash
        # resolve it from crop_bin. The value is the same at this
        # point, but stating it makes the intent checkable.
        _stash_previews(fire, source, path=info.get('path'))

    # Return the fire to READY. The switch rebuilds the same artifacts
    # preparation produces, so a fire that was mid-prepare (or errored
    # on the previous source) is usable again -- without this the badge
    # stays on "preparing"/"error" even though everything succeeded.
    #
    # But do NOT demote a fire that already has a result: MAPPED and
    # ACCEPTED are states the user reached by mapping and accepting,
    # and a source switch (including the automatic one the preview
    # handler performs to build the other source's stash) must not
    # quietly undo them. Only the states that mean "not usable yet"
    # are cleared.
    if fire.status not in (FireStatus.MAPPED, FireStatus.ACCEPTED):
        fire.status = FireStatus.READY
    fire.error_msg = ''
    fire.progress = {}

    if _save_fire_state is not None:
        try:
            _save_fire_state()
        except Exception:
            pass

    if info.get('filled_fraction') is not None:
        fire.console_log.append(
            f"  AOI coverage: {info.get('filled_px', 0):,}/"
            f"{info.get('total_px', 0):,} px "
            f"({info['filled_fraction']:.1%}) filled with non-nodata.")

    return {'ok': True, 'post_source': source,
            'filled_fraction': info.get('filled_fraction'),
            'status': fire.status.value if hasattr(fire.status, 'value')
                      else str(fire.status),
            'hint_mode': fire.hint_mode,
            'tiles': info.get('tiles', []),
            'tile_dates': info.get('tile_dates', {}),
            'post_date': info.get('post_date', '')}


def switch_hint_mode(fire: FireInfo, mode: str) -> dict:
    """Switch a fire's hint mask between viirs / redwins_post / redwins_diff.

    Regenerates ``fire.hint_bin``, the hint overlay preview PNG, and
    updates ``fire.perimeter_type`` / ``fire.hint_mode``.

    Returns a dict with 'ok' (bool) and 'error' (str, if not ok).
    """
    if mode not in ALL_HINT_MODES:
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
        out_path, err = build_derived_hint_for_fire(fire, mode)
        if out_path and getattr(fire, 'restrict_hint_bcws', False):
            # Clip the chosen hint to the BCWS perimeter, so
            # the preview, the agreement score and the
            # clustering all use the same restricted mask.
            out_path = restrict_hint_to_bcws(fire, out_path)
        if err:
            return {'ok': False, 'error': err}
        fire.hint_bin = out_path
        fire.perimeter_type = mode
        fire.hint_mode = mode

    # Regenerate the hint overlay preview PNG.
    if fire.hint_bin and os.path.isfile(fire.hint_bin):
        # Render BOTH the generic hint.png and the per-mode
        # hint_<mode>.png.
        #
        # The preview endpoint prefers the per-mode file and falls back
        # to the generic one. Rendering only the generic file meant the
        # image you got depended on whether a background
        # pregenerate_all_hints() had already produced a per-mode file
        # for some OTHER mode -- so a freshly selected hint sometimes
        # appeared only after toggling views, which happened to force a
        # different resolution path. Writing both makes the result the
        # same regardless of what has run in the background.
        _overlay_mask_on_post(fire, fire.hint_bin, 'hint', (0.0, 0.8, 0.2))
        try:
            _overlay_mask_on_post(fire, fire.hint_bin, f'hint_{mode}',
                                  (0.0, 0.8, 0.2))
        except Exception as exc:
            sys.stderr.write(
                f'[prepare] per-mode hint render failed for {mode}: '
                f'{exc}\n')
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


def ensure_fire_stack_present(fire: FireInfo) -> dict:
    """Make sure this fire's AOI stack still exists on the ramdisk.

    /ram is tmpfs, so a reboot silently empties it while the fire's
    state (on real disk) still points at the stack. Rather than letting
    that surface as a file-not-found deep inside the mapping CLI, every
    entry point that is about to use ``fire.crop_bin`` calls this first
    and rebuilds from the source mosaics if needed.

    Progress is reported through the fire's console log and progress
    snapshot -- the same channels the prepare stages already use, so
    the existing UI picks it up with no popup.

    Returns {'rebuilt': bool, 'path': str}; raises nothing on the happy
    path. On failure the fire is left untouched and the exception
    propagates to the caller, which already knows how to report it.
    """
    from .aoi_stack import ensure_aoi_stack, stack_is_valid

    if not getattr(fire, 'bbox_native', None):
        # Nothing to rebuild from; leave whatever is on record alone.
        return {'rebuilt': False, 'path': fire.crop_bin}

    if fire.crop_bin and stack_is_valid(fire.crop_bin):
        return {'rebuilt': False, 'path': fire.crop_bin}

    fire.console_log.append(
        '  AOI stack missing from ramdisk (server or machine restarted) '
        '-- regenerating from source imagery ...')

    def _cb(detail, frac):
        set_prep_stage(fire, stage_for_stack_detail(detail),
                       detail=f'rebuilding: {detail}', frac=frac)

    info = ensure_aoi_stack(
        fire.fire_numbe, fire.bbox_native, progress_cb=_cb,
        instance_key=getattr(state, 'shared_root', '') or '',
        post_source=getattr(fire, 'post_source', 'l2') or 'l2',
        ref_raster=(state.rasters_by_year.get(fire.fire_year)
                    or state.raster_path),
            # Per-date L2 composites: empty means 'most recent',
            # which is the historical behaviour.
            l2_start_date=getattr(fire, 'l2_start_date', ''))
    fire.crop_bin = info['path']
    if info.get('width'):
        fire.crop_w = info['width']
        fire.crop_h = info['height']
    fire.progress = {}
    if info.get('rebuilt'):
        fire.console_log.append('  AOI stack regenerated.')
    if _save_fire_state is not None:
        try:
            _save_fire_state()
        except Exception:
            pass
    return info


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
    from .viirs_worker import (
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

    # PADDING IS REMOVED. It is pinned to 0 regardless of what the
    # caller or the saved settings ask for.
    #
    # Padding was the only thing that changed the AOI window after a
    # fire was created, and every crop change put previews on a
    # different grid. Reconciling those grids in the split view was a
    # persistent source of misalignment: each mechanism for tracking
    # which grid a preview belonged to (view names, a sidecar, copied
    # entries, HTTP headers, re-rendering) fixed one path and left
    # another. With padding fixed at 0 the AOI window is exactly the
    # bbox the user drew, for every run and every source, so all
    # previews share one grid permanently and there is nothing to
    # reconcile.
    #
    # The parameter is still accepted so old callers, saved settings
    # and persisted state load without error -- it is simply ignored.
    if padding not in (None, 0, 0.0):
        sys.stderr.write(
            f'[prepare] ignoring padding={padding}: padding has been '
            f'removed; the AOI is always the drawn bbox\n')
    pad = 0.0
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
    # Best-effort: if no VIIRS data exists for this fire's bbox/dates,
    # the re-prepare continues without a VIIRS hint (user can switch to
    # "red wins" on the fire mapping page).
    acc_shp = None
    try:
        acc_shp = accumulate_for_fire(fire, cache_dir, ref_raster)
    except WorkerError as exc:
        sys.stderr.write(
            f'[prepare] [{fire_numbe}] VIIRS accumulate returned no '
            f'data ({exc}) — proceeding without VIIRS hint.\n')
        sys.stderr.flush()
    except Exception as exc:
        sys.stderr.write(
            f'[prepare] [{fire_numbe}] accumulate failed ({exc}) '
            f'— proceeding without VIIRS hint.\n')
        sys.stderr.flush()

    # ---- Crop bounds: the user's drawn AOI rectangle (bbox_native),
    # optionally expanded by the padding fraction.
    if not getattr(fire, 'bbox_native', None):
        _set_fire_status(
            fire, FireStatus.ERROR,
            'Cannot re-prepare: fire has no bbox on record.')
        return

    bx0, by0, bx1, by1 = fire.bbox_native
    if pad > 0:
        bw = bx1 - bx0
        bh = by1 - by0
        bx0 -= pad * bw
        by0 -= pad * bh
        bx1 += pad * bw
        by1 += pad * bh
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = bx0, by0, bx1, by1

    # crop_w/crop_h are set from the AOI stack's real dimensions once it
    # has been built (below) rather than estimated from the bbox here --
    # the stack clips its window to the source raster, so an AOI hanging
    # over the mosaic edge would otherwise report dimensions larger than
    # the raster that actually exists, and sample_size would be computed
    # from a pixel count that was never there.
    old_pad = fire.padding_used
    fire.padding_used = pad

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
        from .preview_fs import rmtree as _pf_rmtree
        _pf_rmtree(previews_dir)

    # Build the AOI stack for the (possibly padded) bounds. The stack
    # is regenerated rather than cropped because there is no longer a
    # province-wide stack to cut from -- and because a padding change
    # alters the window, so the previous stack would be the wrong size
    # regardless.
    from .aoi_stack import ensure_aoi_stack, AoiStackError

    def _stack_progress(detail, frac):
        # The builder's own message decides which named step this is,
        # so the header and the detail always describe the same thing.
        set_prep_stage(fire, stage_for_stack_detail(detail),
                       detail=str(detail or ''), frac=frac)

    # An explicit request wins; otherwise the newest acquisition over
    # this AOI, which is what "L2 recent" means.
    _creation_l2_date = getattr(fire, 'l2_start_date', '') or ''
    if (not _creation_l2_date
            and (getattr(fire, 'post_source', 'l2') or 'l2') == 'l2'):
        _creation_l2_date = l2_reference_date(
            fire, bbox=(crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    try:
        stack_info = ensure_aoi_stack(
            fire_numbe,
            (crop_xmin, crop_ymin, crop_xmax, crop_ymax),
            progress_cb=_stack_progress, force=True,
            instance_key=getattr(state, 'shared_root', '') or '',
            post_source=getattr(fire, 'post_source', 'l2') or 'l2',
            ref_raster=ref_raster,
            # Date the L2 product by the DATA it contains.
            #
            # An empty start date produced <prefix>_l2.bin, whose key
            # is taken from the mosaic date in the filename -- so a
            # composite built from the 21 September acquisition was
            # listed as 23 September, the date of the mosaic that
            # happened to be newest. Naming the acquisition makes the
            # file <prefix>_l2_d<acq>.bin and the listed date the one
            # the imagery actually came from.
            l2_start_date=_creation_l2_date)
    except AoiStackError as exc:
        _set_fire_status(fire, FireStatus.ERROR,
                         f'AOI stack build failed: {exc}')
        return
    crop_bin = stack_info['path']

    # Sanity-check the built window against what was requested. The
    # stack clips to the source raster, so an AOI hanging off the edge
    # of the mosaic yields fewer rows/cols than asked for -- which
    # renders as a correct-width, short-height image. Silently using it
    # produced exactly that symptom, so say so loudly instead.
    try:
        # pixel_size is not part of the stack_info contract, so derive
        # it from the built raster rather than assuming a key exists.
        _px = 20.0
        try:
            from osgeo import gdal as _g
            _ds = _g.Open(crop_bin, _g.GA_ReadOnly)
            if _ds is not None:
                _px = abs(_ds.GetGeoTransform()[1]) or 20.0
                _ds = None
        except Exception:
            pass
        _want_w = max(1, int(round((crop_xmax - crop_xmin) / _px)))
        _want_h = max(1, int(round((crop_ymax - crop_ymin) / _px)))
        _got_w = int(stack_info.get('width') or 0)
        _got_h = int(stack_info.get('height') or 0)
        if _got_w and _got_h and (abs(_got_w - _want_w) > 1
                                  or abs(_got_h - _want_h) > 1):
            msg = (f'  WARNING: AOI stack is {_got_w}x{_got_h} px but '
                   f'the drawn AOI implies {_want_w}x{_want_h} -- the '
                   f'window was clipped to the source raster. Part of '
                   f'the AOI has no imagery for this source; try the '
                   f'other post source or move the AOI inside '
                   f'coverage.')
            fire.console_log.append(msg)
            sys.stderr.write('[prepare]' + msg + '\n')
    except Exception as _sexc:
        sys.stderr.write(f'[prepare] window check skipped: {_sexc}\n')
    fire.crop_bin = crop_bin
    fire.crop_w = stack_info['width']
    fire.crop_h = stack_info['height']
    fire.perim_bin = ''

    # Sample size follows the stack's real pixel count, now that it is
    # known.
    sample_size = int(round(
        fire.crop_w * fire.crop_h * state.sample_rate))
    fire.sample_size = max(state.min_samples,
                           min(state.max_samples, sample_size))

    # -- Re-rasterize the cumulative VIIRS shapefile onto the crop frame --
    # Best-effort: if acc_shp is None (no VIIRS data), skip rasterize.
    viirs_bin = None
    if acc_shp and os.path.isfile(acc_shp):
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
        _invalidate_stale_rasterize(acc_shp, crop_rast_dir)
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

    if viirs_bin and os.path.isfile(viirs_bin):
        fire.viirs_bin = viirs_bin
    else:
        fire.viirs_bin = ''

    # -- Re-establish the hint mask for the NEW crop ------------------
    # The crop frame just changed, so any existing hint raster is sized
    # for the *old* crop and no longer aligns. Rebuild it according to
    # whichever hint mode the fire is actually using.
    #
    # This is what makes a serial sweep work with a red-wins hint: the
    # sweep re-prepares on every padding change, and this path used to
    # unconditionally fall back to VIIRS -- clearing hint_bin whenever
    # VIIRS was unavailable, which then failed the whole run with
    # "No hint mask available" even though the user had explicitly
    # selected Red wins (post) or Red wins (diff).
    _mode = getattr(fire, 'hint_mode', 'redwins_post') or 'redwins_post'
    if _mode in DERIVED_HINT_MODES:
        _rw_path, _rw_err = build_derived_hint_for_fire(fire, _mode)
        if _rw_err:
            _set_fire_status(
                fire, FireStatus.ERROR,
                f'Cannot rebuild {_mode} hint for the new crop: {_rw_err}')
            return
        fire.hint_bin = _rw_path
        fire.perimeter_type = _mode
    elif fire.viirs_bin:
        fire.hint_bin = fire.viirs_bin
        fire.perimeter_type = 'viirs'
    else:
        # No VIIRS for this AOI (common now that downloading is
        # disabled). Fall back to red-wins rather than leaving an
        # empty hint, which would fail at map time.
        _rw_path, _rw_err = build_redwins_hint_for_fire(
            fire, 'redwins_post')
        if _rw_path:
            fire.hint_bin = _rw_path
            fire.perimeter_type = 'redwins_post'
            fire.hint_mode = 'redwins_post'
        else:
            fire.hint_bin = ''
            fire.perimeter_type = 'none'
            sys.stderr.write(
                f'[prepare] [{fire_numbe}] no VIIRS and red-wins '
                f'fallback failed: {_rw_err}\n')

    if fire.viirs_start_date:
        fire.acc_start = fire.viirs_start_date
    if fire.viirs_end_date:
        fire.acc_end = fire.viirs_end_date

    # -- Generate preview images --
    set_prep_stage(fire, 'previews',
                   detail='rendering the display layers',
                   frac=0.05)
    views = generate_all_previews(crop_bin, cache_dir, fire_numbe)
    stamp_previews_product(fire)
    set_prep_stage(fire, 'previews',
                   detail='previews written', frac=1.0)
    try:
        from .mapping import record_base_preview_geo
        record_base_preview_geo(cache_dir, crop_bin)
    except Exception:
        pass
    # The AOI grid just changed. Put every existing run overlay back
    # onto it, so all views in this fire share one geotransform and
    # the split view cannot misalign.
    try:
        from .mapping import rerender_run_overlays
        rerender_run_overlays(
            fire, log=lambda m: fire.console_log.append(m))
    except Exception as _rexc:
        sys.stderr.write(
            f'[prepare] run overlay re-render skipped: {_rexc}\n')
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
    # `fire` is not in scope in this function -- only fire_numbe and
    # crop_bin are -- so build the candidate list from those directly
    # rather than through state.classified_names().
    _clf_patterns = []
    if crop_bin:
        _clf_patterns.append(
            os.path.basename(crop_bin) + '_classified.bin')
    _clf_patterns += [f'{fire_numbe}_crop.bin_classified.bin',
                      f'{fire_numbe}_crop_classified.bin',
                      f'{fire_numbe}_classified.bin']
    for pattern in _clf_patterns:
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

    # Clear the progress line: a finished fire must not keep
    # showing the last step it happened to be on.
    try:
        fire.progress = {}
    except Exception:
        pass
    fire.status = FireStatus.READY
    _save_fire_state()

    # Build the OTHER default source in the background.
    #
    # Preparation builds the source the fire is on; the second default
    # product used to appear only at the next start-up, or when the
    # operator switched and waited for it. Building it here means a new
    # AOI offers both defaults -- L2 recent and the MRAP composite --
    # as soon as it is ready.
    #
    # Deliberately via ensure_aoi_stack rather than a source switch:
    # switching would clear the live previews and re-render them, which
    # is exactly the churn that made a freshly prepared fire announce
    # that it was being prepared all over again.
    def _build_other():
        # Named for what it used to do -- build "the other source".
        # It now ensures BOTH default products, because which one is
        # missing depends on the two reference dates, not on which
        # source the fire happens to be pointing at.
        try:
            if fire_numbe not in state.fires:
                return                      # deleted while preparing
            # Both default products, each dated by its own reference
            # date -- the same routine the start-up refresh uses, so a
            # fire created today and a fire refreshed tomorrow end up
            # with the same rule applied.
            #
            # This built only "the other source" with no date, which
            # is why a new fire could end up with an L2 product and no
            # MRAP composite for the newest mosaic.
            _res = ensure_default_products(fire)
            sys.stderr.write(
                f'[prepare] {fire_numbe}: default products -- '
                f'built {_res["built"] or "none"}, '
                f'already present {_res["skipped"] or "none"} '
                f'(MRAP ref {_res["mrap"] or "?"}, '
                f'L2 ref {_res["l2"] or "?"})\n')
            from .durable import mirror_in_background
            mirror_in_background()
        except AoiStackError as exc:
            sys.stderr.write(
                f'[prepare] {fire_numbe}: default products not '
                f'available: {exc}\n')
        except Exception as exc:
            sys.stderr.write(
                f'[prepare] {fire_numbe}: default products failed: '
                f'{type(exc).__name__}: {exc}\n')

    threading.Thread(target=_build_other, daemon=True,
                     name=f'other-src-{fire_numbe}').start()


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
    # Remember that THIS record produced this directory. Deletion uses
    # it to tell "results this fire wrote" from "results some earlier
    # fire of the same name wrote", which a name match cannot do.
    try:
        fire.accepted_dir = fire_dir
    except Exception:
        pass

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
        # Vectorize the accepted mask first, so the copy loop below
        # picks up the shapefile parts and the KML. The accept step
        # already copied *.shp/*.dbf/*.shx/*.prj -- nothing was
        # producing them, which is why exports lost the perimeter.
        try:
            vres = vectorize_classified(fire)
            if vres.get('error'):
                fire.console_log.append(
                    f'  Perimeter vectorization skipped: '
                    f'{vres["error"]}')
            else:
                fire.console_log.append(
                    f'  Perimeter vectorized: {vres["polygons"]} '
                    f'polygon(s) -> shapefile'
                    + (' + KML' if vres.get('kml') else ''))
        except Exception as _vexc:
            fire.console_log.append(
                f'  Perimeter vectorization failed: {_vexc}')

        for pattern in ('*.bin', '*.hdr', '*.png', '*.shp', '*.dbf',
                         '*.shx', '*.prj', '*.cpg', '*.kml'):
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
                # Low-resolution variants exist only so the browser can
                # paint something while the full image loads. They are
                # the same picture at a fraction of the pixels, so they
                # are noise in a delivered archive; the full-resolution
                # file beside them is always shipped.
                if '.low.' in fname:
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
        # Which source layer this run consumed. Looked up here, once,
        # so the deliverable's own record names its input.
        try:
            from .workers import result_attribution
            _attrib = result_attribution(fire)
        except Exception:
            _attrib = {}

        # Write params YAML
        params_dict = {
            'fire': {
                'fire_numbe': fire_numbe,
                'fire_size_ha': fire.fire_size_ha,
                # Recorded separately, because these are different
                # measurements: fire_size_ha is the BCWS perimeter
                # area for the incident, hint_size_ha is the area of
                # the mask this run was seeded with.
                'hint_size_ha': getattr(fire, 'hint_size_ha', 0.0),
                # The source layer this run consumed, named the same
                # way the selector and the manifest name it, so the
                # deliverable says what it was derived from.
                'source_product': _attrib.get('product', ''),
                'source_stack': _attrib.get('stack', ''),
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
                # Recorded separately, because these are different
                # measurements: fire_size_ha is the BCWS perimeter
                # area for the incident, hint_size_ha is the area of
                # the mask this run was seeded with.
                'hint_size_ha': getattr(fire, 'hint_size_ha', 0.0),
                # The source layer this run consumed, named the same
                # way the selector and the manifest name it, so the
                # deliverable says what it was derived from.
                'source_product': _attrib.get('product', ''),
                'source_stack': _attrib.get('stack', ''),
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


def s2_acquired_on(fire, day: str):
    """Did Sentinel-2 image ANY tile over this AOI on *day*?

    True, False, or None when it cannot be determined. The caller
    treats None as "carry on": skipping a build on ignorance would be
    worse than building one product too many.

    The nightly refresh builds today's L2 composite and today's MRAP
    composite for every fire. On a day with no pass over the AOI, both
    are rebuilt from exactly the imagery of the previous product and
    differ from it only in their timestamp -- a duplicate layer, with
    no cloud figure because there was no acquisition to report one
    for. Asking first is cheap: one object listing for the day,
    restricted to the tiles that intersect this AOI, and the answer is
    already cached per tile-day from the cloud-cover work.
    """
    try:
        from . import cloud_cover as _cc
        from .l2_recent import tiles_intersecting_bbox
    except Exception:
        return None
    if not re.fullmatch(r'\d{8}', day or ''):
        return None

    crs = ''
    try:
        ref = (state.rasters_by_year.get(fire.fire_year)
               or state.raster_path)
        if ref:
            ds = gdal.Open(ref, gdal.GA_ReadOnly)
            if ds is not None:
                crs = ds.GetProjection()
                ds = None
    except Exception:
        crs = ''
    try:
        tiles = sorted(set(
            tiles_intersecting_bbox(fire.bbox_native, crs)))
    except Exception as exc:
        sys.stderr.write(
            f'[refresh] {fire.fire_numbe}: tiles unknown ({exc}); '
            f'building as usual\n')
        return None
    if not tiles:
        return None

    # The cache first: the cloud-cover work already records, per tile
    # and day, either a percentage (a product existed) or an explicit
    # empty. A day whose tiles are ALL recorded empty is a definite no.
    try:
        root = os.path.join(state.output_root, '.cloud_cover')
        data = _cc._load(root)
        seen = 0
        for t in tiles:
            v = data.get(_cc._key(_cc.canon_tile(t), day))
            if not isinstance(v, dict):
                continue
            seen += 1
            if isinstance(v.get('pct'), (int, float)):
                return True             # something was imaged
        if seen == len(tiles):
            return False                # every tile recorded, all empty
    except Exception:
        pass

    # Not cached yet: ask the mirror directly for this one day.
    try:
        products = _cc._products_for_day(day, tiles)
        return bool(products)
    except Exception as exc:
        sys.stderr.write(
            f'[refresh] {fire.fire_numbe}: could not list {day} '
            f'({exc}); building as usual\n')
        return None


# ---------------------------------------------------------------
# Reference dates
# ---------------------------------------------------------------
# A product's date is the date of the DATA it was built from, and the
# two kinds of product get their dates from two independent places:
#
#   MRAP composite : the date of the newest <date>_mrap.bin in
#                    /data/mrap_bc -- a province-wide mosaic produced
#                    by a cron job outside this application.
#   L2 recent      : the newest Sentinel-2 acquisition for which a zip
#                    exists on a tile intersecting THIS AOI.
#
# They routinely differ: the mosaic turns over nightly whether or not
# Sentinel-2 passed over a particular fire. The refresh used to ask
# only "was this AOI imaged on the mosaic's date?" and skip BOTH
# products when the answer was no -- so a genuinely new province-wide
# mosaic was never clipped for the fire, and when a build did happen
# the L2 product was labelled with the mosaic's date rather than the
# acquisition it actually contains.

def mrap_reference_date() -> str:
    """Date of the newest province-wide mosaic, or '' if unknown."""
    try:
        from .aoi_stack import find_latest_mrap, MRAP_DIR
        d = (find_latest_mrap() or ('', ''))[0] or ''
        if not d:
            sys.stderr.write(
                f'[refdate] find_latest_mrap returned no date from '
                f'{MRAP_DIR}\n')
        return d
    except Exception as exc:
        try:
            from .aoi_stack import MRAP_DIR as _md
        except Exception:
            _md = '?'
        sys.stderr.write(
            f'[refdate] no MRAP mosaic available in {_md}: '
            f'{type(exc).__name__}: {exc}\n')
        return ''


def l2_reference_date(fire: FireInfo, bbox=None) -> str:
    """Newest Sentinel-2 acquisition available over this AOI, or ''.

    '' means "could not be determined" -- not "none exists" -- so the
    caller leaves the L2 product alone rather than building one with a
    date it cannot justify.

    *bbox* overrides the fire's own: during creation the crop box is
    known before it has been stored on the fire.
    """
    try:
        from .l2_recent import available_acq_dates
        ref = (state.rasters_by_year.get(fire.fire_year)
               or state.raster_path)
        box = bbox if bbox is not None else fire.bbox_native
        dates = available_acq_dates(box, ref_raster=ref)
        return (dates[0]['date'] if dates else '')
    except Exception as exc:
        sys.stderr.write(
            f'[refdate] {getattr(fire, "fire_numbe", "?")}: Sentinel-2 '
            f'dates unavailable ({exc})\n')
        return ''


def existing_product_keys(fire: FireInfo) -> set:
    """Product keys already on disk for this fire."""
    keys = set()
    try:
        cb = getattr(fire, 'crop_bin', '') or ''
        d = os.path.dirname(cb)
        m = re.match(r'^\d{8}_stack_(.+?_[0-9a-fA-F]{6,})(?:_|\.)',
                     os.path.basename(cb))
        if not d or not m or not os.path.isdir(d):
            return keys
        for cand in glob.glob(os.path.join(
                d, f'*_stack_{m.group(1)}*.bin')):
            bn = os.path.basename(cand)
            if any(t in bn for t in ('_nob8', '.kgc', '.post.',
                                     '_selected')):
                continue
            k = product_key_for_path(cand)
            if k:
                keys.add(k)
    except Exception as exc:
        sys.stderr.write(
            f'[refdate] {getattr(fire, "fire_numbe", "?")}: could not '
            f'enumerate products ({exc})\n')
    return keys


def ensure_default_products(fire: FireInfo, log=None,
                            ref_raster: str = '',
                            instance_key: str = '') -> dict:
    """Give this fire the newest product of BOTH kinds, each dated by
    its own reference date. Builds only what is missing.

    Used by fire creation and by the start-up refresh, so the two can
    never disagree about what "the newest products" means. Every case
    is handled the same way, by asking each kind separately:

      new mosaic, no new acquisition -> MRAP built, L2 left alone
      new acquisition, no new mosaic -> L2 built, MRAP left alone
      both new                       -> both built
      neither new                    -> nothing built

    A product whose key already exists is never rebuilt, which is what
    keeps a quiet week from accumulating identical layers under
    different dates.
    """
    out = {'mrap': '', 'l2': '', 'built': [], 'skipped': []}
    if not fire or not getattr(fire, 'bbox_native', None):
        return out

    def _say(msg):
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

    fn = getattr(fire, 'fire_numbe', '?')
    from .aoi_stack import ensure_aoi_stack, AoiStackError
    # Take the caller's reference raster and instance key when it has
    # them. The worker already resolved both to run its own build, and
    # resolving them a second time from module state is a second chance
    # to get a different answer -- or to raise, before a single product
    # has been considered.
    try:
        inst = instance_key or (getattr(state, 'shared_root', '') or '')
        ref = ref_raster or (state.rasters_by_year.get(fire.fire_year)
                             or state.raster_path)
    except Exception as exc:
        sys.stderr.write(
            f'[refresh] {fn}: could not resolve the reference raster '
            f'({type(exc).__name__}: {exc}); using the caller\'s\n')
        inst, ref = instance_key, ref_raster
    have = existing_product_keys(fire)
    sys.stderr.write(
        f'[refresh] {fn}: products already present: '
        f'{sorted(have) or "none"}\n')

    out['mrap'] = mrap_reference_date()
    out['l2'] = l2_reference_date(fire)

    wanted = []
    if out['mrap']:
        wanted.append(('mrap', f'mrap_p{out["mrap"]}',
                       {'mrap_date': out['mrap']}))
    else:
        # Say so. A reference date that cannot be determined silently
        # drops its whole product kind, and that is indistinguishable
        # from "it built fine" unless it is reported.
        _say(f'[refresh] {fn}: no MRAP reference date -- no readable '
             f'<date>_mrap.bin was found, so no MRAP composite can be '
             f'built')
    if out['l2']:
        wanted.append(('l2', f'l2_d{out["l2"]}',
                       {'l2_start_date': out['l2']}))
    else:
        _say(f'[refresh] {fn}: no L2 reference date -- no Sentinel-2 '
             f'zip was found on a tile intersecting this AOI, so no '
             f'L2 recent product can be built')

    for src, key, kw in wanted:
        if key in have:
            out['skipped'].append(key)
            _say(f'[refresh] {fn}: {key} is already present; not '
                 f'rebuilding it')
            continue
        if product_tombstone(fire, key):
            out['skipped'].append(key)
            _say(f'[refresh] {fn}: {key} was deleted by the operator; '
                 f'not rebuilding it')
            continue
        try:
            info = ensure_aoi_stack(
                fire.fire_numbe, fire.bbox_native,
                instance_key=inst, post_source=src,
                ref_raster=ref, **kw)
            path = (info or {}).get('path', '')
            out['built'].append(key)
            _say(f'[refresh] {fn}: built {key} from '
                 f'{os.path.basename(path)}')
            try:
                warm_product_artifacts(fire, path)
            except Exception as wexc:
                sys.stderr.write(f'[refresh] {fn}: warm failed: {wexc}\n')
        except AoiStackError as exc:
            _say(f'[refresh] {fn}: {key} could not be built: {exc}')
        except Exception as exc:
            _say(f'[refresh] {fn}: {key} could not be built: '
                 f'{type(exc).__name__}: {exc}')

    if not wanted:
        _say(f'[refresh] {fn}: neither reference date could be '
             f'determined; nothing built')
    return out


def refresh_products_for_all_fires(delay_s: float = 20.0) -> None:
    """Build today's MRAP and L2 composites for every existing fire.

    Runs in the background at start-up. The province-wide MRAP mosaic
    turns over nightly, so on the morning after a rebuild every fire's
    newest product is one the server has never built. Without this the
    first analyst to open each fire pays that build while they wait;
    with it the new date is already in the source menu.

    Deliberately additive: older products are left on disk, so the
    operator can still switch back to the composite they were working
    with yesterday. That is the whole point of dating them.

    Never fatal -- a fire that cannot be prepared is logged and
    skipped, because start-up must not depend on imagery being
    available.
    """
    import threading

    def _run():
        time.sleep(max(0.0, delay_s))     # let the server finish booting
        try:
            with state.lock:
                names = list(state.fires.keys())
        except Exception:
            return
        if not names:
            return
        sys.stderr.write(
            f'[startup] refreshing products for {len(names)} fire(s)\n')
        built = skipped = failed = 0
        for fn in names:
            try:
                fire = state.fires.get(fn)
                if fire is None or not getattr(fire, 'bbox_native', None):
                    skipped += 1
                    continue
                # Remember EXACTLY what the analyst was on, and put it
                # back: this is a background refresh, not a change of
                # their view.
                #
                # By product key, not by source. "Switch back to MRAP"
                # would land on the NEWEST MRAP composite -- the one this
                # loop has just built -- silently moving a fire off the
                # night's imagery its analyst had chosen. The product key
                # names the specific file, and repointing to it costs a
                # file copy.
                # Build what is missing WITHOUT switching the fire.
                #
                # This used to switch to MRAP, then to L2, then back --
                # three switches per fire, each of which clears the live
                # previews and re-renders them. That is why a fire whose
                # imagery was already on disk greeted the operator with
                # "being prepared" after a restart, and why products
                # appeared to move on their own. ensure_aoi_stack()
                # builds the stack and returns; the fire stays exactly
                # where its operator left it.
                if fn not in state.fires:
                    # Deleted while this loop was running. Rebuilding
                    # would restore files the delete had just purged,
                    # and they would then appear as products of
                    # whatever is next created with that name.
                    skipped += 1
                    continue
                # One routine decides what "the newest products" are,
                # shared with fire creation so the two cannot drift.
                #
                # This replaces a single gate that asked "was this AOI
                # imaged on the MRAP mosaic's date?" and skipped BOTH
                # products when the answer was no. A new province-wide
                # mosaic was then never clipped for the fire, which is
                # exactly the 20260923 MRAP composite that never
                # appeared.
                _res = ensure_default_products(fire)
                if not _res['built']:
                    skipped += 1
                    continue
                built += 1
            except Exception as exc:
                failed += 1
                sys.stderr.write(
                    f'[startup] {fn}: product refresh failed: '
                    f'{type(exc).__name__}: {exc}\n')
        sys.stderr.write(
            f'[startup] product refresh done: {built} refreshed, '
            f'{skipped} skipped, {failed} failed\n')
        try:
            from .persistence import _save_fire_state
            _save_fire_state()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True,
                     name='startup-products').start()


# ---------------------------------------------------------------------
# Preparation progress
# ---------------------------------------------------------------------
# One named stage list, with weights reflecting how long each step
# actually takes. Reporting every step as "5 of 5" told the operator
# nothing about where in the process a fire was, and left whatever text
# the last callback happened to write sitting there while something
# else ran -- which is how the detail line came to describe a step that
# had finished minutes earlier.

PREP_STAGES = [
    ('locating', 'Locating imagery', 0.04),
    ('extracting', 'Reading Sentinel-2 data', 0.56),
    ('compositing', 'Building the AOI composite', 0.20),
    ('previews', 'Rendering preview imagery', 0.12),
    ('hint', 'Computing the hint layer', 0.08),
]
_PREP_INDEX = {k: i for i, (k, _l, _w) in enumerate(PREP_STAGES)}


def prep_stage_label(key: str) -> str:
    for k, label, _w in PREP_STAGES:
        if k == key:
            return label
    return key or ''


def set_prep_stage(fire, key: str, detail: str = '', frac: float = 0.0,
                   kind: str = 'prepare') -> None:
    """Record which preparation step a fire is on, and how far in.

    ``detail`` is written fresh every call, so it can never outlive the
    step that produced it. The overall fraction is the weighted
    position across all stages, which makes the ETA reflect the WHOLE
    job rather than the current step -- the previous version restarted
    its estimate at each step and so promised "nearly done" repeatedly.
    """
    try:
        idx = _PREP_INDEX.get(key, 0)
        frac = max(0.0, min(1.0, float(frac or 0.0)))
        done = sum(w for _k, _l, w in PREP_STAGES[:idx])
        overall = done + PREP_STAGES[idx][2] * frac
        overall = max(0.0, min(0.999, overall))

        prev = getattr(fire, 'progress', None) or {}
        started = prev.get('started_at')
        if not started or prev.get('kind') != kind:
            started = time.time()
        now = time.time()
        elapsed = max(0.0, now - float(started))

        # ETA from the weighted position. Held back until enough of the
        # job has happened to mean anything: a guess made two seconds
        # in is noise, and showing it invites the operator to trust it.
        eta = None
        if overall >= 0.04 and elapsed >= 5.0:
            raw = max(0.0, elapsed * (1.0 - overall) / overall)
            prev_eta = prev.get('eta_s')
            if isinstance(prev_eta, (int, float)) and prev_eta >= 0 \
                    and prev.get('kind') == kind:
                # Blend with the previous figure, weighted by how far
                # through the job we are.
                #
                # The raw estimate is elapsed/position, and position
                # jumps at every stage boundary -- which is what made
                # the countdown fall from two minutes to six seconds in
                # a single poll. Early on the raw number is mostly
                # noise, so it is damped heavily; as the job completes
                # it becomes reliable and is trusted almost fully, so
                # the estimate still converges to zero at the end
                # instead of hanging above it.
                alpha = min(0.9, 0.2 + 0.7 * overall)
                eta = (1.0 - alpha) * prev_eta + alpha * raw
            else:
                eta = raw

        changed = (prev.get('stage') != key
                   or prev.get('detail') != detail)
        fire.progress = {
            'kind': kind,
            'stage': key,
            'stage_label': prep_stage_label(key),
            'stage_idx': idx + 1,
            'total_stages': len(PREP_STAGES),
            'detail': detail or '',
            'fraction': overall,
            'stage_fraction': frac,
            'started_at': started,
            'updated_at': now,
            'elapsed_s': elapsed,
            'eta_s': eta,
            'last_change_at': (now if changed
                               else prev.get('last_change_at', now)),
        }
    except Exception:
        pass


def stage_for_stack_detail(detail: str) -> str:
    """Map an aoi_stack progress message onto a preparation stage.

    The builder reports what it is doing in prose; this is where that
    prose becomes a step the operator can recognise. Unknown text keeps
    the composite stage rather than inventing a new one.
    """
    d = (detail or '').lower()
    if ('tile' in d and ('find' in d or 'intersect' in d)) \
            or 'searching' in d or 'locating' in d:
        return 'locating'
    # 'ready' contains 'read': "AOI stack ready" used to jump a finished
    # build back to the Sentinel-2 reading stage.
    if 'ready' in d:
        return 'compositing'
    if ('zip' in d or 'extract' in d or 'read' in d
            or 'jp2' in d or 'band' in d or 'download' in d):
        return 'extracting'
    if ('warp' in d or 'mosaic' in d or 'composite' in d
            or 'stack' in d or 'header' in d or 'writ' in d):
        return 'compositing'
    return 'compositing'


# ---------------------------------------------------------------------
# One-time migration: unified L2 product keys
# ---------------------------------------------------------------------

_L2P_RE = re.compile(r'(?<![A-Za-z0-9])l2_p(\d{8})(?![0-9])')


def _unified_key(text: str) -> str:
    """'…l2_p20260909…' -> '…l2_d20260909…'."""
    return _L2P_RE.sub(lambda m: f'l2_d{m.group(1)}', text or '')


def migrate_l2_product_keys() -> dict:
    """Rename artefacts left under the old split L2 keys.

    The identity change itself needs no migration: keys are derived
    from stack FILENAMES, so existing stacks resolve to the unified key
    the moment the new code runs. What does need moving is everything
    NAMED after the old key -- preview stashes, hint masks, coverage
    sidecars -- and the remembered selections in fire_state.yaml.

    Without this they are simply not found: previews re-render, hints
    rebuild, coverage is re-fetched from the mirror, and the operator's
    saved product falls back to whatever is loaded. Nothing breaks, but
    a good deal of work is repeated and a selection is lost, so the old
    names are moved rather than abandoned.

    Idempotent: a second run finds nothing to do. Never fatal.
    """
    stats = {'previews': 0, 'hints': 0, 'coverage': 0, 'fires': 0,
             'skipped': 0}

    def _rename(old_path: str, new_path: str) -> bool:
        if old_path == new_path or not os.path.exists(old_path):
            return False
        if os.path.exists(new_path):
            # The unified name already holds something -- the newer
            # build. Leave it alone and drop the stale duplicate rather
            # than overwrite work that is already correct.
            try:
                if os.path.isdir(old_path):
                    shutil.rmtree(old_path, ignore_errors=True)
                else:
                    os.remove(old_path)
            except OSError:
                pass
            stats['skipped'] += 1
            return False
        try:
            os.replace(old_path, new_path)
            return True
        except OSError as exc:
            sys.stderr.write(f'[migrate] {old_path}: {exc}\n')
            return False

    try:
        with state.lock:
            fires = list(state.fires.values())
    except Exception:
        fires = []

    for fire in fires:
        cache = getattr(fire, 'cache_dir', '') or ''
        if cache and os.path.isdir(cache):
            # Preview stashes: previews_l2_p<date> -> previews_l2_d<date>
            try:
                for name in os.listdir(cache):
                    if not name.startswith('previews_l2_p'):
                        continue
                    if _rename(os.path.join(cache, name),
                               os.path.join(cache, _unified_key(name))):
                        stats['previews'] += 1
            except OSError:
                pass

            # Derived hint masks under _redwins/
            rw = os.path.join(cache, '_redwins')
            if os.path.isdir(rw):
                try:
                    for name in os.listdir(rw):
                        if 'l2_p' not in name:
                            continue
                        if _rename(os.path.join(rw, name),
                                   os.path.join(rw, _unified_key(name))):
                            stats['hints'] += 1
                except OSError:
                    pass

            # Per-product coverage sidecars
            cov = os.path.join(cache, 'coverage')
            if os.path.isdir(cov):
                try:
                    for name in os.listdir(cov):
                        if not name.startswith('l2_p'):
                            continue
                        if _rename(os.path.join(cov, name),
                                   os.path.join(cov, _unified_key(name))):
                            stats['coverage'] += 1
                except OSError:
                    pass

        # Remembered selections, so a reload restores what the operator
        # actually chose instead of falling back to what is loaded.
        changed = False
        try:
            up = getattr(fire, 'user_product', '') or ''
            if 'l2_p' in up:
                fire.user_product = _unified_key(up)
                changed = True
            ui = getattr(fire, 'ui_state', None)
            if isinstance(ui, dict):
                for k in ('left_key', 'right_key'):
                    v = ui.get(k)
                    if isinstance(v, str) and 'l2_p' in v:
                        ui[k] = _unified_key(v)
                        changed = True
        except Exception:
            pass
        if changed:
            stats['fires'] += 1

    if any(stats.values()):
        sys.stderr.write(
            f'[migrate] unified L2 keys: {stats["previews"]} preview '
            f'stash(es), {stats["hints"]} hint(s), '
            f'{stats["coverage"]} coverage file(s), '
            f'{stats["fires"]} fire record(s)'
            + (f', {stats["skipped"]} stale duplicate(s) removed'
               if stats['skipped'] else '') + '\n')
        try:
            from .persistence import _save_fire_state
            _save_fire_state()
        except Exception:
            pass
    return stats
