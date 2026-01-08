'''search imagery within shapefile bounds'''
import os
import fnmatch
from osgeo import ogr, gdal, osr


def raster_to_polygon(dataset):
    """
    Create an OGR polygon representing the bounding box of a raster.
    """
    gt = dataset.GetGeoTransform()
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    # Compute coordinates of the four corners
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + gt[1] * x_size
    y_min = y_max + gt[5] * y_size

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_min, y_min)
    ring.AddPoint(x_min, y_max)
    ring.AddPoint(x_max, y_max)
    ring.AddPoint(x_max, y_min)
    ring.AddPoint(x_min, y_min)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly


def load_shapefile_polygon(shp_path):
    """
    Load first polygon geometry from a shapefile using OGR.
    """
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ds = drv.Open(shp_path, 0)  # readonly
    layer = ds.GetLayer()
    feature = layer.GetNextFeature()
    geom = feature.GetGeometryRef().Clone()
    return geom, layer.GetSpatialRef()


def find_intersecting_rasters(root_folder, shapefile_path):
    shp_geom, shp_srs = load_shapefile_polygon(shapefile_path)

    matching_rasters = []

    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            if fnmatch.fnmatch(filename, "*_fcir_*.tif"):
                raster_path = os.path.join(root, filename)

                # Open raster with GDAL
                ds = gdal.Open(raster_path)
                if ds is None:
                    print(f"Cannot open: {raster_path}")
                    continue

                # Build raster footprint polygon
                raster_poly = raster_to_polygon(ds)

                # Check CRS match; if different, reproject raster geometry
                raster_wkt = ds.GetProjection()
                if raster_wkt:
                    raster_srs = osr.SpatialReference()
                    raster_srs.ImportFromWkt(raster_wkt)

                    if not raster_srs.IsSame(shp_srs):
                        # Reproject raster footprint to shapefile CRS
                        transform = osr.CoordinateTransformation(raster_srs, shp_srs)
                        raster_poly.Transform(transform)

                # Intersection test
                if raster_poly.Intersects(shp_geom):
                    print(f"INTERSECTS: {raster_path}")
                    matching_rasters.append(raster_path)
                else:
                    print(f"NO INTERSECTION: {raster_path}")

    return matching_rasters


# -------- Example usage -------- #

root_dir = r"/data/urgent_status_imagery/"
shapefile = r"circle_100km.shp"

intersecting = find_intersecting_rasters(root_dir, shapefile)

print("\n=== Intersecting Rasters ===")
for r in intersecting:
    print(r)
    cmd = 'cp -v ' + r + ' /ram/parker/ '
    print(cmd)
    a = os.system(cmd)

print(len(intersecting))

