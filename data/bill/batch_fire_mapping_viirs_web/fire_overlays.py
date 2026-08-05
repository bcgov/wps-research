"""fire_overlays.py — vector overlays for the individual-fire view.

The fire view renders plain PNG previews, so unlike the province-wide
overview it has no vector layer. This module supplies one: for a given
fire it produces the Sentinel-2 tile grid and the BCWS fire
polygons/points/numbers, already clipped to the AOI and already
converted into the crop raster's PIXEL coordinates.

Doing the projection server-side matters. The client only ever knows
the displayed image size, and the preview PNGs are the crop at full
resolution, so pixel coordinates map to screen by a single uniform
scale -- the same CSS transform that zooms the image carries the
overlay with it, and registration survives zoom and pan for free. If
the client had to hold geotransforms and CRS definitions it would have
to redo that maths on every redraw and would drift.

The result is cached as JSON next to the AOI stack on the ramdisk. It
is derived data over a fixed AOI, so it is cheap to rebuild and
expendable -- exactly what the ramdisk is for.
"""

import json
import os
import sys

from osgeo import gdal, ogr, osr

gdal.UseExceptions()

from .l2_recent import TILES_SHP, normalize_tile_id


def _inv_geotransform(gt):
    """World -> pixel affine. Returns a callable (x, y) -> (col, row)."""
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if det == 0:
        raise ValueError('degenerate geotransform')

    def to_px(x, y):
        dx = x - gt[0]
        dy = y - gt[3]
        col = (dx * gt[5] - dy * gt[2]) / det
        row = (-dx * gt[4] + dy * gt[1]) / det
        return [col, row]
    return to_px


def _crop_info(crop_bin: str):
    ds = gdal.Open(crop_bin, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f'cannot open {crop_bin}')
    try:
        return (ds.GetGeoTransform(), ds.GetProjection(),
                ds.RasterXSize, ds.RasterYSize)
    finally:
        ds = None


def _transform_to(target_wkt: str, layer_srs):
    """CoordinateTransformation into the crop CRS, or None if same.

    Axis order is forced to lon/lat on both sides: GDAL 3 honours the
    authority order for EPSG:4326 (lat,lon), which silently transposes
    every coordinate and would put the whole grid in the wrong place.
    """
    dst = osr.SpatialReference()
    dst.ImportFromWkt(target_wkt)
    try:
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if layer_srs is not None:
            layer_srs.SetAxisMappingStrategy(
                osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass
    if layer_srs is None or layer_srs.IsSame(dst):
        return None
    return osr.CoordinateTransformation(layer_srs, dst)


def _aoi_polygon(gt, w, h):
    """The crop's own footprint, as a polygon in its CRS."""
    xs = [gt[0], gt[0] + w * gt[1]]
    ys = [gt[3], gt[3] + h * gt[5]]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((min(xs), min(ys)), (max(xs), min(ys)),
                 (max(xs), max(ys)), (min(xs), max(ys)),
                 (min(xs), min(ys))):
        ring.AddPoint(x, y)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def _tile_grid_px(gt, proj, w, h, tiles_shp: str = TILES_SHP) -> list:
    """Sentinel-2 tiles intersecting the AOI, as full pixel rectangles.

    Whole tile outlines are kept rather than clipped to the AOI: the
    grid is a reference frame, and a clipped rectangle would draw a
    misleading border along the AOI edge that looks like a tile
    boundary but isn't. The client simply lets them run past the edge.
    """
    if not os.path.isfile(tiles_shp):
        return []
    to_px = _inv_geotransform(gt)
    aoi = _aoi_polygon(gt, w, h)
    out = []
    ds = ogr.Open(tiles_shp)
    if ds is None:
        return []
    try:
        layer = ds.GetLayer()
        ct = _transform_to(proj, layer.GetSpatialRef())
        # Filter in the layer's own CRS, so transform the AOI outward.
        probe = aoi.Clone()
        if ct is not None:
            inv = osr.CoordinateTransformation(
                osr.SpatialReference(wkt=proj), layer.GetSpatialRef())
            try:
                probe.Transform(inv)
            except Exception:
                probe = None
        if probe is not None:
            layer.SetSpatialFilter(probe)
        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            g = geom.Clone()
            if ct is not None:
                try:
                    g.Transform(ct)
                except Exception:
                    continue
            if not g.Intersects(aoi):
                continue
            name = (feat.GetField('Name')
                    if feat.GetFieldIndex('Name') >= 0 else None)
            if not name and feat.GetFieldIndex('Row_Labels') >= 0:
                name = feat.GetField('Row_Labels')
            ring = g.GetGeometryRef(0) if g.GetGeometryCount() else None
            if ring is None:
                continue
            pts = [to_px(*ring.GetPoint_2D(i))
                   for i in range(ring.GetPointCount())]
            out.append({'name': normalize_tile_id(name or ''),
                        'ring': pts})
        layer.SetSpatialFilter(None)
    finally:
        ds = None
    return out


def _bcws_px(state, gt, proj, w, h) -> dict:
    """BCWS polygons, points and fire numbers, in crop pixel coords.

    Reuses the province-wide overlay JSON that bcws.py already
    maintains -- it is stored in the raster's native CRS, which is the
    crop's CRS too, so only the affine to pixels is needed. Rebuilding
    from the shapefiles here would duplicate that work and risk the two
    views disagreeing.
    """
    from .bcws import _overlay_json_path
    path = _overlay_json_path(state)
    if not path or not os.path.isfile(path):
        return {'polygons': [], 'points': [],
                'polygon_fire_nums': [], 'point_fire_nums': []}
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {'polygons': [], 'points': [],
                'polygon_fire_nums': [], 'point_fire_nums': []}

    to_px = _inv_geotransform(gt)

    def _inside(cx, cy, pad=0.0):
        return (-pad <= cx <= w + pad) and (-pad <= cy <= h + pad)

    polys, poly_nums = [], []
    src_polys = data.get('polygons') or []
    src_pnums = data.get('polygon_fire_nums') or []
    for i, ring in enumerate(src_polys):
        if not ring:
            continue
        px = [to_px(x, y) for x, y in ring]
        # Keep a polygon if any vertex is within a generous margin of
        # the crop, so shapes that merely overlap the AOI still draw
        # their visible portion instead of vanishing.
        if not any(_inside(cx, cy, pad=max(w, h)) for cx, cy in px):
            continue
        polys.append(px)
        poly_nums.append(src_pnums[i] if i < len(src_pnums) else '')

    pts, pt_nums = [], []
    src_pts = data.get('points') or []
    src_ptnums = data.get('point_fire_nums') or []
    for i, p in enumerate(src_pts):
        cx, cy = to_px(p[0], p[1])
        if not _inside(cx, cy, pad=0.05 * max(w, h)):
            continue
        pts.append([cx, cy])
        pt_nums.append(src_ptnums[i] if i < len(src_ptnums) else '')

    return {'polygons': polys, 'points': pts,
            'polygon_fire_nums': poly_nums,
            'point_fire_nums': pt_nums}


def overlay_cache_path(crop_bin: str) -> str:
    """Cache beside the AOI stack -- same lifetime, same expendability."""
    return os.path.splitext(crop_bin)[0] + '_overlays.json'


def build_fire_overlays(state, fire, force: bool = False) -> dict:
    """Tile grid + BCWS features for *fire*, in crop pixel coordinates.

    Cached on the ramdisk next to the stack; rebuilt automatically when
    the cache is missing (a reboot clears /ram) or when the crop is
    newer than the cache (a padding change resized it, which invalidates
    every pixel coordinate in here).
    """
    crop = getattr(fire, 'crop_bin', '')
    if not crop or not os.path.isfile(crop):
        return {'tiles': [], 'bcws': {}, 'width': 0, 'height': 0}

    cache = overlay_cache_path(crop)
    if not force and os.path.isfile(cache):
        try:
            if os.path.getmtime(cache) >= os.path.getmtime(crop):
                with open(cache, encoding='utf-8') as f:
                    return json.load(f)
        except (OSError, ValueError):
            pass

    gt, proj, w, h = _crop_info(crop)
    try:
        tiles = _tile_grid_px(gt, proj, w, h)
    except Exception as exc:
        sys.stderr.write(f'[fire_overlays] tile grid failed: {exc}\n')
        tiles = []
    try:
        bcws = _bcws_px(state, gt, proj, w, h)
    except Exception as exc:
        sys.stderr.write(f'[fire_overlays] bcws failed: {exc}\n')
        bcws = {'polygons': [], 'points': [],
                'polygon_fire_nums': [], 'point_fire_nums': []}

    out = {'tiles': tiles, 'bcws': bcws, 'width': w, 'height': h}
    try:
        tmp = f'{cache}.tmp{os.getpid()}'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f)
        os.replace(tmp, cache)
    except OSError as exc:
        sys.stderr.write(f'[fire_overlays] cache write failed: {exc}\n')
    return out
