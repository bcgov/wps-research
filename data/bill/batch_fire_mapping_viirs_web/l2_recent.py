"""l2_recent.py — most-recent L2A composite over an AOI, straight from SAFE zips.

Why this exists
---------------
The MRAP mosaic is a cloud-screened running composite, so its "post"
imagery can lag the present by days wherever cloud persisted. For an
actively burning fire that lag matters. This module builds the
alternative: for each Sentinel-2 tile intersecting the AOI, take the
single most recent L2A acquisition available on disk and mosaic those
tiles together over the AOI window.

The result may well contain cloud -- that is accepted and expected. It
trades cloud-freeness for recency, which is the whole point.

Pipeline
--------
1. Intersect the AOI with ``Sentinel_BC_Tiles.shp`` to get tile IDs.
   The shapefile is WGS84 (EPSG:4326) while AOIs are in the raster's
   CRS (BC Albers), so the AOI is transformed before comparison.
2. For each tile, list ``/data/mrap_bc/L2_T<tile>/*.zip`` and pick the
   most recent by SENSING date parsed from the filename.
3. Extract B12/B11/B9/B8 from each chosen zip (B9/B8 resampled to the
   20 m grid, exactly as sentinel2_extract_swir_nir.py does).
4. Warp each tile's 4-band result into the AOI's grid and paste it into
   a shared buffer.

Tile-ID conventions
-------------------
The shapefile stores ``09VUC``; the directories are ``L2_T09VUC`` and
the zips embed ``_T09VUC_``. The leading ``T`` is a filename convention,
not part of the tile ID, so :func:`tile_dir_name` and
:func:`normalize_tile_id` translate in both directions rather than
assuming either form.
"""

import glob
import json
import os
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

MRAP_DIR = '/data/mrap_bc'
# Repo-relative default; overridable so the module stays testable.
TILES_SHP = os.environ.get(
    'SENTINEL_BC_TILES_SHP',
    '/home/ash/GitHub/wps-research/py/sentinel2_bc_tiles_shp/'
    'Sentinel_BC_Tiles.shp')

BANDS = ('B12', 'B11', 'B9', 'B8')
TARGET_RES_M = 20.0

# S2A_MSIL2A_20260720T191831_..._T10UFB_20260721T032655.zip
#                ^acquisition                ^tile   ^processing
# The tile token is captured WITH its leading zone character (usually
# 'T') so it can be displayed verbatim; the character is not assumed,
# it is read from the filename.
_ZIP_RE = re.compile(
    r'^S2[A-Z]_MSIL2A_(\d{8}T\d{6})_.*?_([A-Z][0-9]{2}[A-Z]{3})_'
    r'(\d{8}T\d{6})\.zip$', re.IGNORECASE)

# Stop pulling in older acquisitions for a tile once this fraction of
# its AOI footprint carries real data. Trades a little completeness for
# a lot of extraction time -- each additional zip is a full ~1 GB read.
FILL_TARGET = 0.95


class L2RecentError(RuntimeError):
    pass


def normalize_tile_id(tile: str) -> str:
    """``T09VUC`` or ``09VUC`` -> ``09VUC`` (shapefile form)."""
    t = str(tile).strip().upper()
    if re.match(r'^T[0-9]{2}[A-Z]{3}$', t):
        return t[1:]
    return t


def tile_dir_name(tile: str) -> str:
    """``09VUC`` -> ``L2_T09VUC`` (on-disk directory form)."""
    return f'L2_T{normalize_tile_id(tile)}'


# ----------------------------------------------------------------------
# 1. AOI -> intersecting tile IDs
# ----------------------------------------------------------------------

def tiles_intersecting_bbox(bbox_native, crs_wkt: str,
                            tiles_shp: str = TILES_SHP) -> list:
    """Tile IDs whose footprint intersects the AOI.

    *bbox_native* is (xmin, ymin, xmax, ymax) in the CRS given by
    *crs_wkt* (the AOI raster's CRS, i.e. BC Albers). The tile
    shapefile is WGS84, so the AOI rectangle is reprojected into the
    layer's CRS before testing -- comparing raw coordinates across two
    different CRSs would silently match the wrong tiles or none at all.
    """
    if not os.path.isfile(tiles_shp):
        raise L2RecentError(f'tile shapefile not found: {tiles_shp}')

    ds = ogr.Open(tiles_shp)
    if ds is None:
        raise L2RecentError(f'could not open {tiles_shp}')
    try:
        layer = ds.GetLayer()
        layer_srs = layer.GetSpatialRef()

        src_srs = osr.SpatialReference()
        if crs_wkt:
            src_srs.ImportFromWkt(crs_wkt)
        else:
            raise L2RecentError('AOI CRS is unknown; cannot match tiles')

        # Both sides are forced to lon/lat axis order. GDAL 3 honours
        # the authority's axis order by default, which for EPSG:4326 is
        # lat,lon -- that silently transposes every coordinate and the
        # intersection then finds nothing.
        try:
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if layer_srs is not None:
                layer_srs.SetAxisMappingStrategy(
                    osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            pass

        xmin, ymin, xmax, ymax = (float(v) for v in bbox_native)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax),
                     (xmin, ymax), (xmin, ymin)):
            ring.AddPoint(x, y)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        if layer_srs is not None and not src_srs.IsSame(layer_srs):
            poly.Transform(osr.CoordinateTransformation(src_srs, layer_srs))

        layer.SetSpatialFilter(poly)
        out = []
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is None or not geom.Intersects(poly):
                continue
            # 'Name' is the tile ID column in Sentinel_BC_Tiles.shp;
            # 'Row_Labels' carries the same value as a fallback.
            name = feat.GetField('Name') if feat.GetFieldIndex('Name') >= 0 \
                else None
            if not name and feat.GetFieldIndex('Row_Labels') >= 0:
                name = feat.GetField('Row_Labels')
            if name:
                out.append(normalize_tile_id(name))
        layer.SetSpatialFilter(None)
    finally:
        ds = None
    return sorted(set(out))


# ----------------------------------------------------------------------
# 2. tile -> most recent zip
# ----------------------------------------------------------------------

def zips_for_tile(tile: str, mrap_dir: str = MRAP_DIR) -> list:
    """All L2A zips for *tile*, newest first.

    Ordered by ``<acquisition>_<processing>`` concatenated, e.g.
    ``20260716T191909_20260716T225545``. Acquisition dominates, and
    within one acquisition the later-PROCESSED product wins -- these
    directories really do carry two processings of the same overpass
    (``20260728T190911`` appears with both ``...T223612`` and
    ``...T233402``), and the later one supersedes the earlier.

    Neither mtime nor processing time alone would order these
    correctly: mtime reflects download order, and processing time alone
    would rank a late reprocessing of an old scene above a fresh one.

    Returns ``[(sort_key, acq_yyyymmdd, tile_token, path), ...]`` where
    ``tile_token`` keeps its zone character (``T10UEA``).
    """
    d = os.path.join(mrap_dir, tile_dir_name(tile))
    if not os.path.isdir(d):
        return []
    out = []
    try:
        names = os.listdir(d)
    except OSError:
        return []
    for name in names:
        if not name.lower().endswith('.zip'):
            continue
        m = _ZIP_RE.match(name)
        if not m:
            continue
        acq, ztile, proc = m.groups()
        if normalize_tile_id(ztile) != normalize_tile_id(tile):
            continue
        out.append((f'{acq}_{proc}', acq[:8], ztile.upper(),
                    os.path.join(d, name)))
    out.sort(key=lambda r: r[0], reverse=True)
    return out


def available_acq_dates(bbox_native, crs_wkt: str = '',
                        mrap_dir: str = MRAP_DIR,
                        tiles_shp: str = TILES_SHP,
                        ref_raster: str = '') -> list:
    """Distinct acquisition dates (YYYYMMDD) available over an AOI.

    Uses exactly the tiles the compositor would use, and the same zip
    listing, so the dates offered are precisely the ones a build could
    start from. The date is the first 8 characters of the acquisition
    timestamp -- the third underscore-separated field of the zip name --
    so several overpasses on one day collapse to a single entry.

    Newest first, which is the order the selector shows and the order
    the compositor walks backwards through.
    """
    # The tile lookup needs the AOI's CRS, and the compositor gets it by
    # reading the reference raster's projection -- do exactly the same,
    # so this listing can never disagree with the tiles a build would
    # use. (tiles_intersecting_bbox takes no ref_raster; passing one was
    # my error and is what raised the TypeError.)
    proj = crs_wkt or ''
    if not proj and ref_raster:
        ref = gdal.Open(ref_raster, gdal.GA_ReadOnly)
        if ref is None:
            raise L2RecentError(
                f'reference raster could not be opened: {ref_raster}')
        try:
            proj = ref.GetProjection()
        finally:
            ref = None
    if not proj:
        raise L2RecentError(
            'no CRS available for the AOI; cannot find intersecting '
            'Sentinel-2 tiles')

    tiles = tiles_intersecting_bbox(bbox_native, proj,
                                    tiles_shp=tiles_shp)
    seen = {}
    for t in tiles:
        for _key, acq8, _tok, path in zips_for_tile(t, mrap_dir=mrap_dir):
            if not acq8 or len(acq8) != 8 or not acq8.isdigit():
                continue
            e = seen.setdefault(acq8, {'date': acq8, 'tiles': set(),
                                       'zips': 0})
            e['tiles'].add(normalize_tile_id(t))
            e['zips'] += 1
    out = []
    for d in sorted(seen.keys(), reverse=True):
        e = seen[d]
        out.append({'date': d,
                    'tiles': sorted(e['tiles']),
                    'zips': int(e['zips'])})
    return out


def most_recent_zip_for_tile(tile: str, mrap_dir: str = MRAP_DIR):
    """Newest zip for *tile* as ``(acq_yyyymmdd, path)``, or None."""
    z = zips_for_tile(tile, mrap_dir=mrap_dir)
    if not z:
        return None
    return z[0][1], z[0][3]


# ----------------------------------------------------------------------
# 3. zip -> in-memory 4-band 20 m dataset
# ----------------------------------------------------------------------

def _open_safe_in_zip(zip_path: str):
    """Return a GDAL dataset for the SAFE metadata inside *zip_path*.

    Uses /vsizip so nothing is unpacked to disk -- these are ~1 GB each
    and only four bands are wanted.
    """
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise L2RecentError(f'cannot read {zip_path}: {exc}')

    meta = [n for n in names
            if n.endswith('MTD_MSIL2A.xml') or n.endswith('MTD_MSIL1C.xml')]
    if not meta:
        raise L2RecentError(f'no MTD_MSIL2A.xml inside {zip_path}')
    meta.sort(key=len)
    vsi = f'/vsizip/{zip_path}/{meta[0]}'
    ds = gdal.Open(vsi)
    if ds is None:
        raise L2RecentError(f'GDAL could not open {vsi}')
    return ds


def extract_bands_to_mem(zip_path: str):
    """Extract B12/B11/B9/B8 from a SAFE zip into one 20 m MEM dataset.

    Mirrors sentinel2_extract_swir_nir.py: B12/B11 are natively 20 m and
    used as the target grid; B9 (60 m) and B8 (10 m) are warped onto it
    bilinearly. Band order is fixed at B12, B11, B9, B8 to match the
    MRAP products, so the stack's band layout is identical either way.
    """
    d = _open_safe_in_zip(zip_path)
    try:
        found = {}
        for sub_path, _ in d.GetSubDatasets():
            sub = gdal.Open(sub_path)
            if sub is None:
                continue
            for i in range(1, sub.RasterCount + 1):
                b = sub.GetRasterBand(i)
                bn = (b.GetMetadata() or {}).get('BANDNAME')
                if bn in BANDS and bn not in found:
                    found[bn] = (sub, i)
            if all(b in found for b in BANDS):
                break
        missing = [b for b in BANDS if b not in found]
        if missing:
            raise L2RecentError(
                f'{os.path.basename(zip_path)}: missing band(s) '
                f'{missing}')

        tgt_sub, tgt_i = found['B12']
        gt = tgt_sub.GetGeoTransform()
        proj = tgt_sub.GetProjection()
        W, H = tgt_sub.RasterXSize, tgt_sub.RasterYSize

        mem = gdal.GetDriverByName('MEM')
        out = mem.Create('', W, H, len(BANDS), gdal.GDT_Float32)
        out.SetGeoTransform(gt)
        out.SetProjection(proj)

        for idx, bn in enumerate(BANDS, start=1):
            sub, i = found[bn]
            if (sub.RasterXSize, sub.RasterYSize) == (W, H):
                arr = sub.GetRasterBand(i).ReadAsArray().astype(np.float32)
            else:
                one = mem.Create('', sub.RasterXSize, sub.RasterYSize, 1,
                                 gdal.GDT_Float32)
                one.SetGeoTransform(sub.GetGeoTransform())
                one.SetProjection(sub.GetProjection())
                one.GetRasterBand(1).WriteArray(
                    sub.GetRasterBand(i).ReadAsArray().astype(np.float32))
                res = mem.Create('', W, H, 1, gdal.GDT_Float32)
                res.SetGeoTransform(gt)
                res.SetProjection(proj)
                gdal.Warp(res, one, xRes=abs(gt[1]), yRes=abs(gt[5]),
                          resampleAlg='bilinear')
                arr = res.GetRasterBand(1).ReadAsArray().astype(np.float32)
                one = res = None
            # 0 is L2A's nodata fill; treat it as absent so it does not
            # darken the mosaic or bias the anomaly (raster_zero_to_nan
            # does the same in the offline pipeline).
            arr[arr == 0] = np.nan
            out.GetRasterBand(idx).WriteArray(arr)
        return out
    finally:
        d = None


# ----------------------------------------------------------------------
# 4. mosaic into the AOI grid
# ----------------------------------------------------------------------

def _tile_footprint_mask(tile: str, win_gt, xsize: int, ysize: int,
                         proj: str, tiles_shp: str = TILES_SHP):
    """Boolean mask of the AOI window covered by *tile*'s footprint.

    The fill target is measured against this rather than the whole AOI:
    a tile clipping one corner of the AOI can never fill 95% of the AOI,
    so a whole-AOI test would make it read every zip it owns and still
    never stop early.

    Falls back to all-True if the footprint cannot be rasterized, which
    degrades to "measure against the whole window" -- conservative
    (more reading) rather than wrong.
    """
    try:
        ds = ogr.Open(tiles_shp)
        if ds is None:
            return np.ones((ysize, xsize), dtype=bool)
        try:
            layer = ds.GetLayer()
            wanted = normalize_tile_id(tile)
            match = None
            for feat in layer:
                nm = feat.GetField('Name') if \
                    feat.GetFieldIndex('Name') >= 0 else None
                if not nm and feat.GetFieldIndex('Row_Labels') >= 0:
                    nm = feat.GetField('Row_Labels')
                if nm and normalize_tile_id(nm) == wanted:
                    match = feat.GetGeometryRef().Clone()
                    break
            if match is None:
                return np.ones((ysize, xsize), dtype=bool)

            src_srs = layer.GetSpatialRef()
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromWkt(proj)
            try:
                dst_srs.SetAxisMappingStrategy(
                    osr.OAMS_TRADITIONAL_GIS_ORDER)
                if src_srs is not None:
                    src_srs.SetAxisMappingStrategy(
                        osr.OAMS_TRADITIONAL_GIS_ORDER)
            except AttributeError:
                pass
            if src_srs is not None and not src_srs.IsSame(dst_srs):
                match.Transform(
                    osr.CoordinateTransformation(src_srs, dst_srs))

            # 'Memory' is deprecated from GDAL 3.11; 'MEM' is the
            # replacement. Fall back so this still works on older GDAL.
            mem_drv = (ogr.GetDriverByName('MEM')
                       or ogr.GetDriverByName('Memory'))
            mds = mem_drv.CreateDataSource('mask')
            mlayer = mds.CreateLayer('m', dst_srs, ogr.wkbPolygon)
            f = ogr.Feature(mlayer.GetLayerDefn())
            f.SetGeometry(match)
            mlayer.CreateFeature(f)

            rdrv = gdal.GetDriverByName('MEM')
            rds = rdrv.Create('', xsize, ysize, 1, gdal.GDT_Byte)
            rds.SetGeoTransform(win_gt)
            rds.SetProjection(proj)
            gdal.RasterizeLayer(rds, [1], mlayer, burn_values=[1])
            arr = rds.GetRasterBand(1).ReadAsArray().astype(bool)
            rds = None
            mds = None
            return arr
        finally:
            ds = None
    except Exception:
        return np.ones((ysize, xsize), dtype=bool)


def date_polygons_path(stack_bin: str) -> str:
    """Sidecar holding the per-date coverage polygons.

    Lives beside the AOI stack on the ramdisk: same lifetime, same
    expendability, and it is rebuilt with the stack. Keeping it next to
    the stack (rather than in the fire cache) means switching post
    source or reopening a fire recalls exactly the polygons that match
    the L2 buffer currently on disk.
    """
    return os.path.splitext(stack_bin)[0] + '_dates.json'


def _polygonize_date_mask(mask, date_str: str, simplify_px: float = 1.5):
    """Outline the pixels attributed to one acquisition.

    Returns rings in CROP PIXEL coordinates, not map units: the client
    only ever draws these into a fixed-aspect thumbnail, so pixel space
    is the natural frame and needs no geotransform on the client side.

    Polygons are simplified slightly -- a per-pixel outline of a
    swath-shaped region can run to tens of thousands of vertices, which
    is pointless for a thumbnail and slow to ship.
    """
    h, w = mask.shape
    drv = gdal.GetDriverByName('MEM')
    src = drv.Create('', w, h, 1, gdal.GDT_Byte)
    src.GetRasterBand(1).WriteArray(mask.astype(np.uint8))
    # Identity geotransform => polygon coordinates ARE pixel coords.
    src.SetGeoTransform((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    ogr_drv = ogr.GetDriverByName('MEM') or ogr.GetDriverByName('Memory')
    vds = ogr_drv.CreateDataSource('poly')
    layer = vds.CreateLayer('p', None, ogr.wkbPolygon)
    fld = ogr.FieldDefn('v', ogr.OFTInteger)
    layer.CreateField(fld)
    try:
        # Band 1 doubles as its own mask so only value-1 pixels emit
        # polygons; without the mask the 0-background becomes a
        # polygon too.
        gdal.Polygonize(src.GetRasterBand(1), src.GetRasterBand(1),
                        layer, 0)
    except Exception as exc:
        sys.stderr.write(f'[l2_recent] polygonize failed: {exc}\n')
        return []

    rings = []
    for feat in layer:
        if feat.GetField('v') != 1:
            continue
        g = feat.GetGeometryRef()
        if g is None:
            continue
        try:
            g = g.Simplify(simplify_px)
        except Exception:
            pass
        if g is None or g.IsEmpty():
            continue
        # A Polygonize result may be Polygon or MultiPolygon; flatten
        # to exterior rings, which is all a thumbnail needs.
        polys = ([g.GetGeometryRef(i) for i in range(g.GetGeometryCount())]
                 if g.GetGeometryName() == 'MULTIPOLYGON' else [g])
        for poly in polys:
            if poly is None or poly.GetGeometryCount() == 0:
                continue
            ext = poly.GetGeometryRef(0)
            if ext is None or ext.GetPointCount() < 4:
                continue
            rings.append([[round(px, 1), round(py, 1)]
                          for px, py in
                          (ext.GetPoint_2D(i)
                           for i in range(ext.GetPointCount()))])
    src = None
    vds = None
    return rings


def write_date_polygons(out_json: str, date_map, date_list,
                        xsize: int, ysize: int,
                        date_sats=None,
                        acq_times=None) -> dict:
    """Build and persist the per-date coverage polygons."""
    date_sats = date_sats or {}
    acq_times = list(acq_times or [])
    entries = []
    for idx, acq in enumerate(date_list):
        m = (date_map == idx)
        n_px = int(m.sum())
        if n_px == 0:
            continue
        rings = _polygonize_date_mask(m, acq)
        entries.append({'date': acq, 'pixels': n_px, 'rings': rings,
                        'sats': date_sats.get(acq, [])})
    # Newest first so the legend reads in the same order as the
    # extraction log.
    entries.sort(key=lambda e: e['date'], reverse=True)
    # Flat, deduplicated, most recent first: one entry per
    # satellite/date pair that actually contributed to this composite.
    # Built here because this is where the satellite is already known
    # per date; the caller writes it beside the stack, so it can be read
    # back later without reopening a single zip.
    sources = sorted(
        {f'{sat}_{acq}'
         for acq, sats in (date_sats or {}).items()
         for sat in (sats or ['S2'])
         if acq},
        key=lambda v: (v.split('_')[1], v.split('_')[0]),
        reverse=True)
    payload = {'width': xsize, 'height': ysize, 'dates': entries,
               'sources': sources,
               # Newest acquisition datetime that actually went into
               # this composite, UTC, YYYYMMDDTHHMMSS as it appears in
               # the source file name. Used to date the delivered
               # products, so it must be the datetime, not just the day.
               'acq_newest_utc': (max(acq_times) if acq_times else '')}
    try:
        tmp = f'{out_json}.tmp{os.getpid()}'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f)
        os.replace(tmp, out_json)
    except OSError as exc:
        sys.stderr.write(
            f'[l2_recent] date polygon write failed: {exc}\n')
    return payload


def build_l2_recent_post(bbox_native, ref_raster: str, out_bin: str,
                         progress_cb=None,
                         mrap_dir: str = MRAP_DIR,
                         tiles_shp: str = TILES_SHP,
                         log_cb=None,
                         fill_target: float = FILL_TARGET,
                         start_date: str = '') -> dict:
    """Build the 4-band most-recent-L2 composite over the AOI.

    Output is on the SAME grid the AOI stack uses (from *ref_raster*),
    so the L2 post bands are pixel-aligned with the pre bands and the
    anomaly needs no further resampling.

    Backfilling
    -----------
    A single overpass often leaves large nodata gaps over an AOI (swath
    edges, and the L2A fill value). For each tile, older acquisitions
    are pulled in -- newest first -- writing ONLY into pixels still
    empty, until *fill_target* of that tile's AOI footprint carries
    data or the tile's zips are exhausted. Because gaps are only ever
    filled, never overwritten, the most recent observation always wins
    wherever it exists; older scenes just patch the holes.

    The threshold is evaluated per tile against that tile's own AOI
    footprint, not against the whole AOI: a tile covering a sliver of
    the AOI could never reach a whole-AOI threshold, and would then
    read every zip it has for nothing.
    """
    def _p(detail, frac):
        if progress_cb:
            try:
                progress_cb(detail, frac)
            except Exception:
                pass

    def _log(msg):
        sys.stderr.write(f'[l2_recent] {msg}\n')
        sys.stderr.flush()
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    ref = gdal.Open(ref_raster, gdal.GA_ReadOnly)
    if ref is None:
        raise L2RecentError(f'cannot open reference raster {ref_raster}')
    try:
        gt = ref.GetGeoTransform()
        proj = ref.GetProjection()
        rW, rH = ref.RasterXSize, ref.RasterYSize
    finally:
        ref = None

    from .aoi_stack import _window_for_bbox
    xmin, ymin, xmax, ymax = (float(v) for v in bbox_native)
    xoff, yoff, xsize, ysize, win_gt = _window_for_bbox(
        gt, rW, rH, xmin, ymin, xmax, ymax)
    total_px = xsize * ysize

    _p('finding intersecting Sentinel-2 tiles', 0.02)
    tiles = tiles_intersecting_bbox(bbox_native, proj, tiles_shp=tiles_shp)
    if not tiles:
        raise L2RecentError('no Sentinel-2 tiles intersect this AOI')
    _log(f'AOI is {xsize}x{ysize} px ({total_px:,} px at 20 m); '
         f'{len(tiles)} intersecting tile(s): {", ".join(tiles)}')

    # start_date caps how recent an acquisition may be.
    #
    # The backfill already walks newest-first and stops when the tile's
    # footprint is full, so dropping everything newer than start_date
    # makes it start there and go back exactly as it always does. That
    # is the whole change: same ordering, same threshold, same
    # gap-filling -- only the starting point moves.
    sd = (start_date or '').strip()
    if sd and (len(sd) != 8 or not sd.isdigit()):
        raise L2RecentError(f'start_date must be YYYYMMDD, got {sd!r}')
    if sd:
        _log(f'start date {sd}: ignoring acquisitions after this date')

    per_tile = []
    for t in tiles:
        z = zips_for_tile(t, mrap_dir=mrap_dir)
        if sd:
            z = [r for r in z if r[1] <= sd]
        if z:
            per_tile.append((t, z))
            _log(f'  {z[0][2]}: {len(z)} zip(s) available, '
                 f'newest {os.path.basename(z[0][3])}')
        else:
            _log(f'  {t}: no zips found in '
                 f'{os.path.join(mrap_dir, tile_dir_name(t))}')
    if not per_tile:
        raise L2RecentError(
            f'no L2A zips found for tile(s) {", ".join(tiles)}')

    mem = gdal.GetDriverByName('MEM')
    acc = mem.Create('', xsize, ysize, len(BANDS), gdal.GDT_Float32)
    acc.SetGeoTransform(win_gt)
    acc.SetProjection(proj)
    for i in range(1, len(BANDS) + 1):
        acc.GetRasterBand(i).WriteArray(
            np.full((ysize, xsize), np.nan, dtype=np.float32))
    # Band 1 drives the filled-mask; all four bands share nodata.
    filled = np.zeros((ysize, xsize), dtype=bool)
    # Per-pixel acquisition attribution for the whole AOI, and the
    # registry the indices refer to.
    date_map = np.full((ysize, xsize), -1, dtype=np.int16)
    date_list = []
    # Which platform(s) contributed each acquisition date. A single
    # date can be filled from more than one satellite where swaths
    # overlap, so this is a set per date, not a single value.
    date_sats = {}
    acq_datetimes = set()
    date_lock = threading.Lock()

    def _date_index(acq: str, sat: str = '',
                    dt: str = '') -> int:
        """Stable index for an acquisition date, assigned on first use.

        Workers run concurrently, so the registry is guarded; the index
        is what gets written into the per-pixel maps.
        """
        with date_lock:
            if sat:
                date_sats.setdefault(acq, set()).add(sat)
            # Full acquisition datetime (UTC) of everything composited.
            # The date alone is not enough to name a delivered product,
            # which carries the hour and minute.
            if dt:
                acq_datetimes.add(dt)
            try:
                return date_list.index(acq)
            except ValueError:
                date_list.append(acq)
                return len(date_list) - 1

    used = []
    n_tiles = len(per_tile)
    tile_secs = []          # per-tile wall time, for the ETA

    def _process_tile(ti, tile, zlist):
        """Fill one tile's footprint into a PRIVATE buffer.

        Each worker owns its arrays and its GDAL datasets, so tiles can
        run concurrently without locking. Results are merged into the
        shared accumulator afterwards, on one thread -- merging is
        cheap next to the ~1 GB reads this does.

        Returning a private buffer (rather than writing into `acc`) is
        what makes the parallelism safe: two tiles that overlap in the
        AOI would otherwise race on the same pixels.
        """
        t_start = time.time()
        token = zlist[0][2]
        tile_mask = _tile_footprint_mask(
            tile, win_gt, xsize, ysize, proj, tiles_shp)
        tile_px = int(tile_mask.sum())
        if tile_px == 0:
            return (tile, token, None, None, [], 0.0,
                    [f'  {token}: footprint does not overlap the AOI '
                     f'window; skipping'])

        lines = [f'  {token}: AOI footprint {tile_px:,} px; '
                 f'target {fill_target:.0%} filled']
        local = np.full((len(BANDS), ysize, xsize), np.nan,
                        dtype=np.float32)
        local_filled = np.zeros((ysize, xsize), dtype=bool)
        local_used = []
        # Which acquisition each pixel came from, as an index into
        # `date_list` below. Tracked per pixel rather than per zip
        # because a later tile can pre-empt an earlier one during the
        # merge; attributing dates from the zip loop alone would claim
        # pixels that never made it into the final mosaic.
        local_date = np.full((ysize, xsize), -1, dtype=np.int16)

        for zi, (_key, acq, _tok, zpath) in enumerate(zlist, 1):
            got = int((local_filled & tile_mask).sum())
            frac = got / tile_px
            if frac >= fill_target:
                break
            lines.append(
                f'    [{zi}/{len(zlist)}] {os.path.basename(zpath)} '
                f'(acq {acq}) -- footprint {frac:.1%} filled, '
                f'extracting ...')
            try:
                src = extract_bands_to_mem(zpath)
            except L2RecentError as exc:
                lines.append(f'      skipped: {exc}')
                continue
            try:
                drv = gdal.GetDriverByName('MEM')
                warped = drv.Create('', xsize, ysize, len(BANDS),
                                    gdal.GDT_Float32)
                warped.SetGeoTransform(win_gt)
                warped.SetProjection(proj)
                for i in range(1, len(BANDS) + 1):
                    warped.GetRasterBand(i).WriteArray(
                        np.full((ysize, xsize), np.nan, dtype=np.float32))
                gdal.Warp(warped, src, resampleAlg='near',
                          srcNodata=np.nan, dstNodata=np.nan)
                new0 = warped.GetRasterBand(1).ReadAsArray()
                gap = (~local_filled) & ~np.isnan(new0)
                n_new = int(gap.sum())
                if n_new:
                    for i in range(len(BANDS)):
                        nb = warped.GetRasterBand(i + 1).ReadAsArray()
                        local[i][gap] = nb[gap]
                    local_filled |= gap
                    # Platform prefix comes straight off the SAFE
                    # name (S2A_/S2B_/S2C_), so it needs no extra I/O.
                    _sat = os.path.basename(zpath)[:3].upper()
                    # Third underscore field of the SAFE name is the
                    # acquisition datetime (YYYYMMDDTHHMMSS, UTC);
                    # `acq` is only its date half.
                    _parts = os.path.basename(zpath).split('_')
                    _dt = _parts[2] if len(_parts) > 2 else ''
                    local_date[gap] = _date_index(
                        acq, _sat if _sat.startswith('S2') else '', _dt)
                    local_used.append((token, acq))
                after = int((local_filled & tile_mask).sum())
                lines.append(f'      +{n_new:,} px; footprint now '
                             f'{after / tile_px:.1%} filled')
            finally:
                src = None
                warped = None

        secs = time.time() - t_start
        final = int((local_filled & tile_mask).sum()) / tile_px
        lines.append(f'  {token}: done -- {final:.1%} of its AOI '
                     f'footprint filled in {secs:.0f}s')
        return (tile, token, local, local_filled, local_used, secs,
                lines, local_date)

    # Tiles are independent until the merge, and each is dominated by
    # decompressing ~1 GB zips, so running them concurrently is close
    # to a linear speedup. Capped so a many-tile AOI cannot exhaust
    # memory: each worker holds a full AOI-sized 4-band float32 buffer.
    max_workers = max(1, min(len(per_tile), 4))
    _log(f'extracting {n_tiles} tile(s) with {max_workers} worker(s) '
         f'in parallel ...')
    done_n = 0
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_process_tile, ti, tile, zl): (ti, tile)
                for ti, (tile, zl) in enumerate(per_tile, 1)}
        for fut in as_completed(futs):
            ti, tile = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:
                _log(f'  {tile}: FAILED ({exc})')
                continue
            done_n += 1
            for ln in res[6]:
                _log(ln)
            if res[5]:
                tile_secs.append(res[5])
            results.append(res)
            eta = ''
            if tile_secs and done_n < n_tiles:
                mean = sum(tile_secs) / len(tile_secs)
                pending = n_tiles - done_n
                # With N workers the remaining tiles overlap, so the
                # wall-clock estimate divides by the worker count.
                remaining = mean * pending / max_workers
                eta = f' (ETA {remaining:.0f}s remaining this date)'
            _p(f'extracted {res[1]} {done_n}/{n_tiles} tiles{eta}',
               0.05 + 0.85 * done_n / n_tiles)

    # Merge private buffers. Deterministic order (sorted by tile) so the
    # same inputs always give the same mosaic regardless of which
    # worker finished first.
    for (tile, token, local, local_filled, local_used, _secs, _lines,
         local_date) in sorted(results, key=lambda r: r[0]):
        if local is None:
            continue
        gap = (~filled) & local_filled
        if not gap.any():
            continue
        for i in range(len(BANDS)):
            dst = acc.GetRasterBand(i + 1).ReadAsArray()
            dst[gap] = local[i][gap]
            acc.GetRasterBand(i + 1).WriteArray(dst)
        # Attribute only the pixels this tile actually contributed.
        date_map[gap] = local_date[gap]
        filled |= gap
        used.extend(local_used)

    if not used:
        raise L2RecentError('every intersecting tile failed to extract')

    aoi_frac = float(filled.sum()) / max(1, total_px)
    _log(f'AOI coverage: {int(filled.sum()):,}/{total_px:,} px '
         f'({aoi_frac:.1%}) filled with non-nodata.')

    # NaN is the right sentinel WHILE building -- the backfill logic
    # depends on "is this pixel still empty?" -- but it must not reach
    # the stack. The MRAP mosaics use 0 for nodata, so a stack built
    # from them is entirely finite; an L2 stack carrying NaN in its
    # post bands (and therefore in the anomaly bands derived from them)
    # is a different kind of raster than the mapping CLI has ever been
    # given. scikit-learn's t-SNE/HDBSCAN/RandomForest all reject
    # non-finite input, which fails every run identically regardless of
    # parameters.
    #
    # Converting to 0 here makes the L2 product match the MRAP
    # convention exactly, so downstream code cannot tell the two apart.
    n_nan_total = 0
    for i in range(1, len(BANDS) + 1):
        arr = acc.GetRasterBand(i).ReadAsArray()
        n_nan = int(np.isnan(arr).sum())
        n_nan_total += n_nan
        if n_nan:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            acc.GetRasterBand(i).WriteArray(arr)
    if n_nan_total:
        _log(f'nodata: converted {n_nan_total:,} NaN cell(s) across '
             f'{len(BANDS)} band(s) to 0 (matching the MRAP nodata '
             f'convention; sklearn rejects non-finite input)')

    _p('writing L2-recent composite', 0.9)
    os.makedirs(os.path.dirname(out_bin) or '.', exist_ok=True)
    tmp = f'{out_bin}.tmp{os.getpid()}'
    drv = gdal.GetDriverByName('ENVI')
    out = drv.Create(tmp, xsize, ysize, len(BANDS), gdal.GDT_Float32,
                     options=['INTERLEAVE=BSQ'])
    if out is None:
        raise L2RecentError(f'could not create {tmp}')
    out.SetGeoTransform(win_gt)
    out.SetProjection(proj)
    for i in range(1, len(BANDS) + 1):
        out.GetRasterBand(i).WriteArray(
            acc.GetRasterBand(i).ReadAsArray())
    out.FlushCache()
    out = None
    acc = None

    hdr_tmp = os.path.splitext(tmp)[0] + '.hdr'
    hdr_out = os.path.splitext(out_bin)[0] + '.hdr'
    os.replace(tmp, out_bin)
    if os.path.isfile(hdr_tmp):
        os.replace(hdr_tmp, hdr_out)

    newest = max(d for _, d in used)

    _p('outlining per-date coverage', 0.95)
    dates_json = date_polygons_path(out_bin)
    poly_payload = write_date_polygons(
        dates_json, date_map, date_list, xsize, ysize,
        date_sats={k: sorted(v) for k, v in date_sats.items()},
        acq_times=sorted(acq_datetimes))
    # Deliberately does NOT print the path: this sidecar is written
    # beside the temporary post buffer and relocated next to the stack
    # by the caller, so printing it here shows a filename that no
    # longer exists a moment later.
    _log('date coverage: '
         + ', '.join(f"{e['date']} ({e['pixels']:,} px)"
                     for e in poly_payload['dates']))

    _p('L2-recent composite ready', 1.0)
    return {
        'path': out_bin,
        'width': xsize,
        'height': ysize,
        'bands': len(BANDS),
        'tiles': sorted({t for t, _ in used}),
        'tile_dates': dict(used),
        'post_date': newest,
        'filled_fraction': aoi_frac,
        'dates_json': dates_json,
        'date_coverage': poly_payload['dates'],
        'sources': poly_payload.get('sources', []),
        'acq_newest_utc': poly_payload.get('acq_newest_utc', ''),
        'filled_px': int(filled.sum()),
        'total_px': total_px,
        'geotransform': win_gt,
        'projection': proj,
    }
