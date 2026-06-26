"""BCWS current-fire points + polygons overlay.

Downloads the two live BCWS datasets from data.gov.bc.ca (current fire
points and current fire polygons), extracts them, and reprojects every
feature's coordinates into a target raster's native CRS so they can be
drawn directly on the /new_fire overview using the same
nativeToCanvas() transform already used for the bbox drawer -- no
separate coordinate logic needed client-side.

Source datasets (BC Wildfire Service, data.gov.bc.ca):
    points:   https://pub.data.gov.bc.ca/datasets/2790e3f7-6395-4230-8545-04efb5a18800/prot_current_fire_points.zip
    polygons: https://pub.data.gov.bc.ca/datasets/cdfc2d7b-c046-4bf0-90ac-4897232619e1/prot_current_fire_polys.zip

These are always-overwrite "latest snapshot" downloads -- the caller
asked not to keep timestamped backups here (unlike the original
standalone refresh script this was adapted from), since the cached
JSON below is itself the only artifact the app reads.
"""

import json
import os
import re
import ssl
import shutil
import sys
import urllib.request
import zipfile

from osgeo import ogr, osr

ogr.UseExceptions()
osr.UseExceptions()

# Pattern for BCWS fire numbers: 1 letter + 5 digits (e.g. G80280, C12345)
_FIRE_NUM_RE = re.compile(r'^[A-Za-z]\d{5}$')

BCWS_DATASETS = [
    {
        'key': 'points',
        'filename': 'prot_current_fire_points.zip',
        'url': ('https://pub.data.gov.bc.ca/datasets/'
                '2790e3f7-6395-4230-8545-04efb5a18800/'
                'prot_current_fire_points.zip'),
    },
    {
        'key': 'polys',
        'filename': 'prot_current_fire_polys.zip',
        'url': ('https://pub.data.gov.bc.ca/datasets/'
                'cdfc2d7b-c046-4bf0-90ac-4897232619e1/'
                'prot_current_fire_polys.zip'),
    },
]


def _bcws_dir(state) -> str:
    d = os.path.join(state.shared_root, '.web_cache', '_bcws')
    os.makedirs(d, exist_ok=True)
    return d


def _overlay_json_path(state) -> str:
    return os.path.join(_bcws_dir(state), 'overlay.json')


def _download_and_extract(dataset: dict, dest_dir: str) -> None:
    """Download dataset['url'] into dest_dir/dataset['filename'],
    overwriting any previous copy, then extract it in place. No
    timestamped backup is kept -- dest_dir is itself a cache, and the
    overlay.json built from it is the only thing the app reads."""
    ssl_context = ssl.create_default_context(cafile=_certifi_where())
    zip_path = os.path.join(dest_dir, dataset['filename'])
    tmp_path = zip_path + '.tmp'
    with urllib.request.urlopen(dataset['url'], context=ssl_context,
                                timeout=120) as response, \
            open(tmp_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    os.replace(tmp_path, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)


def _certifi_where() -> str | None:
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return None  # falls back to the system default CA bundle


def _find_shapefile(dest_dir: str, expected_stem_hint: str) -> str | None:
    """The zip's internal shapefile name doesn't necessarily match the
    zip's own filename, so just take the most-recently-extracted .shp
    in dest_dir whose name loosely matches the hint (point/poly), and
    fall back to the newest .shp in the dir if nothing matches."""
    candidates = [
        os.path.join(dest_dir, f) for f in os.listdir(dest_dir)
        if f.lower().endswith('.shp')
    ]
    if not candidates:
        return None
    hinted = [c for c in candidates
              if expected_stem_hint in os.path.basename(c).lower()]
    pool = hinted or candidates
    return max(pool, key=os.path.getmtime)


def _reproject_ring(ring, ct) -> list:
    pts = []
    for i in range(ring.GetPointCount()):
        x, y, *_ = ring.GetPoint(i)
        tx, ty, _ = ct.TransformPoint(x, y)
        pts.append([round(tx, 2), round(ty, 2)])
    return pts


def _detect_fire_num_field(shp_path: str) -> str | None:
    """Auto-detect which field in the shapefile contains BCWS fire
    numbers (pattern: 1 letter + 5 digits, e.g. G80280). Scans all
    text fields and returns the first field name where at least one
    stripped value matches. Returns None if no field matches."""
    from osgeo import gdal as _gdal
    _gdal.PushErrorHandler('CPLQuietErrorHandler')
    try:
        ds = ogr.Open(shp_path)
    finally:
        _gdal.PopErrorHandler()
    if ds is None:
        return None
    layer = ds.GetLayer(0)
    defn = layer.GetLayerDefn()
    # Collect candidate text field names
    text_fields = []
    for i in range(defn.GetFieldCount()):
        fd = defn.GetFieldDefn(i)
        if fd.GetType() == ogr.OFTString:
            text_fields.append(fd.GetName())
    if not text_fields:
        ds = None
        return None
    # Check up to 200 features for a match
    layer.ResetReading()
    for _ in range(200):
        feat = layer.GetNextFeature()
        if feat is None:
            break
        for fname in text_fields:
            val = (feat.GetField(fname) or '').strip()
            if _FIRE_NUM_RE.match(val):
                ds = None
                return fname
    ds = None
    return None


def _read_features(shp_path: str, target_crs_wkt: str,
                   fire_num_field: str = None) -> tuple:
    """Returns (points, polygons, point_fire_nums, polygon_fire_nums):
    points is a list of [x, y] in the target CRS; polygons is a list
    of rings (each a list of [x, y]), one ring per polygon outer
    boundary (holes are ignored). point_fire_nums and
    polygon_fire_nums are parallel arrays of fire number strings (or
    None for features without one)."""
    # Suppress GDAL's "Value '...' of field FEATURE_CD parsed
    # incompletely to real 0" warnings -- BCWS's FEATURE_CD is a
    # text code (e.g. "JA70003000") that OGR's shapefile driver
    # tries to opportunistically read as numeric and warns about;
    # harmless here since this field isn't used at all, just noisy.
    from osgeo import gdal as _gdal
    _gdal.PushErrorHandler('CPLQuietErrorHandler')
    try:
        ds = ogr.Open(shp_path)
    finally:
        _gdal.PopErrorHandler()
    if ds is None:
        return [], []
    layer = ds.GetLayer(0)
    src_srs = layer.GetSpatialRef()
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(target_crs_wkt)
    if src_srs is not None:
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = (osr.CoordinateTransformation(src_srs, dst_srs)
          if src_srs is not None else None)

    points: list = []
    polygons: list = []
    point_fire_nums: list = []
    polygon_fire_nums: list = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        gtype = geom.GetGeometryType()
        gtype_flat = ogr.GT_Flatten(gtype)

        # Extract fire number if field is known
        fire_num = None
        if fire_num_field:
            val = (feature.GetField(fire_num_field) or '').strip()
            if _FIRE_NUM_RE.match(val):
                fire_num = val.upper()

        def _xform_point(x, y):
            if ct is None:
                return [round(x, 2), round(y, 2)]
            tx, ty, _ = ct.TransformPoint(x, y)
            return [round(tx, 2), round(ty, 2)]

        if gtype_flat == ogr.wkbPoint:
            x, y = geom.GetX(), geom.GetY()
            points.append(_xform_point(x, y))
            point_fire_nums.append(fire_num)
        elif gtype_flat == ogr.wkbMultiPoint:
            for i in range(geom.GetGeometryCount()):
                pt = geom.GetGeometryRef(i)
                points.append(_xform_point(pt.GetX(), pt.GetY()))
                point_fire_nums.append(fire_num)
        elif gtype_flat == ogr.wkbPolygon:
            if geom.GetGeometryCount() > 0:
                ring = geom.GetGeometryRef(0)  # outer ring only
                if ct is not None:
                    polygons.append(_reproject_ring(ring, ct))
                else:
                    polygons.append([
                        [round(ring.GetX(i), 2), round(ring.GetY(i), 2)]
                        for i in range(ring.GetPointCount())
                    ])
                polygon_fire_nums.append(fire_num)
        elif gtype_flat == ogr.wkbMultiPolygon:
            for i in range(geom.GetGeometryCount()):
                poly = geom.GetGeometryRef(i)
                if poly.GetGeometryCount() > 0:
                    ring = poly.GetGeometryRef(0)
                    if ct is not None:
                        polygons.append(_reproject_ring(ring, ct))
                    else:
                        polygons.append([
                            [round(ring.GetX(i), 2), round(ring.GetY(i), 2)]
                            for i in range(ring.GetPointCount())
                        ])
                    polygon_fire_nums.append(fire_num)
    ds = None
    return points, polygons, point_fire_nums, polygon_fire_nums


def refresh_bcws_overlay(state) -> dict:
    """Download both BCWS datasets fresh, extract, reproject into the
    active raster's CRS, and write the combined overlay.json cache.
    Returns the overlay dict ({'points': [...], 'polygons': [...],
    'updated_at': iso8601, 'crs_wkt': ...}). Raises on any failure --
    callers decide how to surface that (e.g. as a 500 to the UI)
    rather than silently leaving a stale/empty cache in place; the
    previous good overlay.json is left untouched until this returns
    successfully.
    """
    import datetime
    from osgeo import gdal
    gdal.UseExceptions()

    if not state.raster_path or not os.path.isfile(state.raster_path):
        raise RuntimeError('no active raster to reproject BCWS data into')

    ds = gdal.Open(state.raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'cannot open active raster: {state.raster_path}')
    target_crs_wkt = ds.GetProjection() or ''
    ds = None
    if not target_crs_wkt:
        raise RuntimeError('active raster has no projection defined')

    dest_dir = _bcws_dir(state)
    all_points: list = []
    all_polygons: list = []
    all_point_fire_nums: list = []
    all_polygon_fire_nums: list = []
    for dataset in BCWS_DATASETS:
        _download_and_extract(dataset, dest_dir)
        shp_path = _find_shapefile(dest_dir, dataset['key'][:5])
        if shp_path is None:
            raise RuntimeError(
                f"no .shp found after extracting {dataset['filename']}")
        # Auto-detect which field holds the fire number
        fn_field = _detect_fire_num_field(shp_path)
        if fn_field:
            sys.stderr.write(
                f'[bcws] Detected fire number field in '
                f'{os.path.basename(shp_path)}: {fn_field}\n')
        pts, polys, pt_fnums, poly_fnums = _read_features(
            shp_path, target_crs_wkt, fire_num_field=fn_field)
        all_points.extend(pts)
        all_polygons.extend(polys)
        all_point_fire_nums.extend(pt_fnums)
        all_polygon_fire_nums.extend(poly_fnums)

    overlay = {
        'points': all_points,
        'polygons': all_polygons,
        'point_fire_nums': all_point_fire_nums,
        'polygon_fire_nums': all_polygon_fire_nums,
        'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'crs_wkt': target_crs_wkt,
        'n_points': len(all_points),
        'n_polygons': len(all_polygons),
    }
    tmp = _overlay_json_path(state) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(overlay, f)
    os.replace(tmp, _overlay_json_path(state))
    return overlay


def load_bcws_overlay(state) -> dict | None:
    """Return the cached overlay dict, or None if it's never been
    downloaded yet."""
    path = _overlay_json_path(state)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
