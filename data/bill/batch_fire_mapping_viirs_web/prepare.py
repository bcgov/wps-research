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

    # Write as single-band ENVI Byte, 0/1 only.
    driver = gdal.GetDriverByName('ENVI')
    out_ds = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)
    if out_ds is None:
        return -1
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_ds.GetRasterBand(1).WriteArray(result)
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
            from .state import find_classified
            clf_path = find_classified(
                fire.fire_numbe, [fire.cache_dir])
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
                     'result', 'comparison', 'brush_comparison')
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


def build_derived_hint_for_fire(fire: FireInfo, mode: str):
    """Build whichever derived hint *mode* names.

    Single entry point so the call sites -- switch, prepare, re-prepare,
    pregenerate -- do not each need to know which builder handles which
    mode. Adding a mode means adding it here and to DERIVED_HINT_MODES,
    not editing five branches.
    """
    if mode == 'bcws_perimeter':
        return build_bcws_hint_for_fire(fire)
    return build_redwins_hint_for_fire(fire, mode)


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
    out_path = os.path.join(out_dir, f'bcws_perimeter_{src}_hint.bin')

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


def build_redwins_hint_for_fire(fire: FireInfo, mode: str):
    """Generate the red-wins hint for *mode* against ``fire.crop_bin``.

    Returns ``(path, None)`` on success or ``(None, error_message)``.

    Shared by :func:`switch_hint_mode` (user picks a hint mode) and by
    the re-prepare path (crop changed, so the mask must be rebuilt to
    match the new crop's dimensions -- a hint raster generated for a
    different crop would not align with it).
    """
    if mode not in ('redwins_post', 'redwins_diff'):
        return None, f'Not a red-wins mode: {mode}'
    if not fire.crop_bin or not os.path.isfile(fire.crop_bin):
        return None, 'Fire has no crop raster.'

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
    out_path = os.path.join(out_dir, f'{mode}_{src}_hint.bin')

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
                >= os.path.getmtime(fire.crop_bin)):
            return out_path, None
    except OSError:
        pass

    n_fire = generate_redwins_hint(fire.crop_bin, indices, out_path)
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


def _preview_stash_dir(fire: FireInfo, source: str) -> str:
    """Per-source copy of the rendered previews.

    generate_all_previews() always writes to ``<cache>/previews``, so
    the two post sources overwrite each other there. Without a stash,
    every switch has to re-render every preview even though the stack
    it renders from is already cached -- which is what made switching
    slow. Keeping a copy per source turns a switch into a handful of
    file copies.
    """
    return os.path.join(fire.cache_dir, f'previews_{source}')


def _stash_previews(fire: FireInfo, source: str) -> None:
    src = os.path.join(fire.cache_dir, 'previews')
    if not os.path.isdir(src):
        return
    dst = _preview_stash_dir(fire, source)
    try:
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    except OSError as exc:
        sys.stderr.write(f'[prepare] preview stash failed: {exc}\n')


def _restore_previews(fire: FireInfo, source: str) -> bool:
    """Put *source*'s stashed previews back in place. True if restored.

    Refuses a stash older than the stack it came from: a re-prepare can
    resize the crop, which makes every stashed PNG the wrong dimensions
    and would misregister the vector overlays drawn on top of them.
    """
    src = _preview_stash_dir(fire, source)
    if not os.path.isdir(src):
        return False
    try:
        if fire.crop_bin and os.path.isfile(fire.crop_bin):
            if os.path.getmtime(src) < os.path.getmtime(fire.crop_bin):
                return False
        dst = os.path.join(fire.cache_dir, 'previews')
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
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


def switch_post_source(fire: FireInfo, source: str) -> dict:
    """Switch a fire between the L2-recent and MRAP post imagery.

    Only the POST bands change. The pre bands are the same median
    composite in both cases, and the anomaly is recomputed from
    (new post, same pre) with the identical formula -- so 'Pre-fire'
    looks the same either way while 'Post-fire' and 'Diff 1' follow the
    selection, which is exactly the intended behaviour.

    The stacks live at different paths per source, so switching back and
    forth reuses whichever is already built rather than rebuilding.
    """
    if source not in ('l2', 'mrap'):
        return {'ok': False, 'error': f'Unknown post source: {source}'}
    if not getattr(fire, 'bbox_native', None):
        return {'ok': False, 'error': 'Fire has no bbox on record.'}

    with _source_switch_lock(fire.fire_numbe):
        return _switch_post_source_locked(fire, source)


def _switch_post_source_locked(fire: FireInfo, source: str) -> dict:
    from .aoi_stack import ensure_aoi_stack, AoiStackError

    ref_raster = (state.rasters_by_year.get(fire.fire_year)
                  or state.raster_path)

    def _cb(detail, frac):
        try:
            fire.progress = {
                'stage': 'cropping', 'stage_idx': 5, 'total_stages': 5,
                'detail': f'{source.upper()} stack: {detail}',
                'fraction': max(0.0, min(1.0, float(frac))),
                'updated_at': time.time(),
            }
        except Exception:
            pass

    try:
        info = ensure_aoi_stack(
            fire.fire_numbe, fire.bbox_native, progress_cb=_cb,
            instance_key=getattr(state, 'shared_root', '') or '',
            post_source=source, ref_raster=ref_raster,
            log_cb=lambda m: fire.console_log.append(m.rstrip()))
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
    prev_src = getattr(fire, 'post_source', '') or ''
    if prev_src and prev_src != source:
        try:
            _stash_previews(fire, prev_src)
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
    restored = _restore_previews(fire, source)
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
                               'hint', 'result', 'comparison',
                               'brush_comparison')
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
            views = generate_all_previews(
                fire.crop_bin, fire.cache_dir, fire.fire_numbe)
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
        rw_path, rw_err = build_derived_hint_for_fire(fire, mode)
        if rw_path:
            fire.hint_bin = rw_path
            fire.perimeter_type = mode
            fire.hint_mode = mode
        else:
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
        # Render EVERY hint mode for this source before stashing, so
        # the stash carries all of them and later hint toggles never
        # need a render.
        pregenerate_all_hints(fire)
        # Snapshot now that previews/ holds this source's images AND
        # all of its hint overlays, so a later switch back restores
        # everything.
        _stash_previews(fire, source)

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
        try:
            fire.progress = {
                'stage': 'cropping',
                'stage_idx': 5,
                'total_stages': 5,
                'detail': f'Rebuilding AOI stack: {detail}',
                'fraction': max(0.0, min(1.0, float(frac))),
                'updated_at': time.time(),
            }
        except Exception:
            pass

    info = ensure_aoi_stack(
        fire.fire_numbe, fire.bbox_native, progress_cb=_cb,
        instance_key=getattr(state, 'shared_root', '') or '',
        post_source=getattr(fire, 'post_source', 'l2') or 'l2',
        ref_raster=(state.rasters_by_year.get(fire.fire_year)
                    or state.raster_path))
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
        shutil.rmtree(previews_dir, ignore_errors=True)

    # Build the AOI stack for the (possibly padded) bounds. The stack
    # is regenerated rather than cropped because there is no longer a
    # province-wide stack to cut from -- and because a padding change
    # alters the window, so the previous stack would be the wrong size
    # regardless.
    from .aoi_stack import ensure_aoi_stack, AoiStackError

    def _stack_progress(detail, frac):
        try:
            fire.progress = {
                'stage': 'cropping',
                'stage_idx': 5,
                'total_stages': 5,
                'detail': f'AOI stack: {detail}',
                'fraction': max(0.0, min(1.0, float(frac))),
                'updated_at': time.time(),
            }
        except Exception:
            pass

    try:
        stack_info = ensure_aoi_stack(
            fire_numbe,
            (crop_xmin, crop_ymin, crop_xmax, crop_ymax),
            progress_cb=_stack_progress, force=True,
            instance_key=getattr(state, 'shared_root', '') or '',
            post_source=getattr(fire, 'post_source', 'l2') or 'l2',
            ref_raster=ref_raster)
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
    views = generate_all_previews(crop_bin, cache_dir, fire_numbe)
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
