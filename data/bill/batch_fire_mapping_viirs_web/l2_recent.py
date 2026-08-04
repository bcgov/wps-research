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
import os
import re
import sys
import zipfile

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
#                ^sensing datetime            ^tile   ^processing
_ZIP_RE = re.compile(
    r'^S2[A-Z]_MSIL2A_(\d{8})T(\d{6})_.*?_T([0-9]{2}[A-Z]{3})_'
    r'(\d{8})T(\d{6})\.zip$', re.IGNORECASE)


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

def most_recent_zip_for_tile(tile: str, mrap_dir: str = MRAP_DIR):
    """Newest L2A zip for *tile*, as ``(sensing_yyyymmdd, path)``.

    Ranked by the SENSING datetime in the filename, not mtime and not
    the processing timestamp: a granule reprocessed later still images
    the same day, and mtime reflects when it was downloaded.
    """
    d = os.path.join(mrap_dir, tile_dir_name(tile))
    if not os.path.isdir(d):
        return None
    best = None
    for path in glob.glob(os.path.join(d, '*.zip')):
        m = _ZIP_RE.match(os.path.basename(path))
        if not m:
            continue
        sens_date, sens_time, ztile, proc_date, proc_time = m.groups()
        if normalize_tile_id(ztile) != normalize_tile_id(tile):
            continue
        key = (sens_date, sens_time, proc_date, proc_time)
        if best is None or key > best[0]:
            best = (key, sens_date, path)
    if best is None:
        return None
    return best[1], best[2]


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

def build_l2_recent_post(bbox_native, ref_raster: str, out_bin: str,
                         progress_cb=None,
                         mrap_dir: str = MRAP_DIR,
                         tiles_shp: str = TILES_SHP) -> dict:
    """Build the 4-band most-recent-L2 composite over the AOI.

    The output is written on the SAME grid the AOI stack uses (derived
    from *ref_raster*'s geotransform and CRS), so the L2 post bands are
    pixel-aligned with the pre bands and the anomaly can be computed
    without any further resampling.
    """
    def _p(detail, frac):
        if progress_cb:
            try:
                progress_cb(detail, frac)
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

    _p('finding intersecting Sentinel-2 tiles', 0.05)
    tiles = tiles_intersecting_bbox(bbox_native, proj, tiles_shp=tiles_shp)
    if not tiles:
        raise L2RecentError('no Sentinel-2 tiles intersect this AOI')

    chosen = []
    for t in tiles:
        got = most_recent_zip_for_tile(t, mrap_dir=mrap_dir)
        if got:
            chosen.append((t, got[0], got[1]))
    if not chosen:
        raise L2RecentError(
            f'no L2A zips found for tile(s) {", ".join(tiles)}')

    _p(f'{len(chosen)} tile(s): '
       + ', '.join(f'{t}@{d}' for t, d, _ in chosen), 0.1)

    mem = gdal.GetDriverByName('MEM')
    acc = mem.Create('', xsize, ysize, len(BANDS), gdal.GDT_Float32)
    acc.SetGeoTransform(win_gt)
    acc.SetProjection(proj)
    for i in range(1, len(BANDS) + 1):
        acc.GetRasterBand(i).WriteArray(
            np.full((ysize, xsize), np.nan, dtype=np.float32))

    used = []
    for n, (tile, sens_date, zpath) in enumerate(chosen, 1):
        _p(f'extracting {tile} ({sens_date})',
           0.1 + 0.75 * (n - 1) / len(chosen))
        try:
            src = extract_bands_to_mem(zpath)
        except L2RecentError as exc:
            sys.stderr.write(f'[l2_recent] skipping {tile}: {exc}\n')
            continue
        try:
            warped = mem.Create('', xsize, ysize, len(BANDS),
                                gdal.GDT_Float32)
            warped.SetGeoTransform(win_gt)
            warped.SetProjection(proj)
            for i in range(1, len(BANDS) + 1):
                warped.GetRasterBand(i).WriteArray(
                    np.full((ysize, xsize), np.nan, dtype=np.float32))
            # Each tile is in its own UTM zone; reprojecting into the
            # AOI's CRS here is what lets tiles from different zones
            # mosaic together correctly.
            gdal.Warp(warped, src, resampleAlg='near',
                      srcNodata=np.nan, dstNodata=np.nan)
            for i in range(1, len(BANDS) + 1):
                dst = acc.GetRasterBand(i).ReadAsArray()
                new = warped.GetRasterBand(i).ReadAsArray()
                # First tile to cover a pixel wins. Tiles are processed
                # in sorted order so the result is deterministic rather
                # than depending on filesystem listing order.
                fill = np.isnan(dst) & ~np.isnan(new)
                dst[fill] = new[fill]
                acc.GetRasterBand(i).WriteArray(dst)
            used.append((tile, sens_date))
        finally:
            src = None
            warped = None

    if not used:
        raise L2RecentError('every intersecting tile failed to extract')

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
    _p('L2-recent composite ready', 1.0)
    return {
        'path': out_bin,
        'width': xsize,
        'height': ysize,
        'bands': len(BANDS),
        'tiles': [t for t, _ in used],
        'tile_dates': dict(used),
        'post_date': newest,
        'geotransform': win_gt,
        'projection': proj,
    }
