'''search imagery within shapefile bounds - smart copy version'''
root_dir = r"/data/urgent_status_imagery/"  # directory to search for TIF files
shapefile = r"circle_100km.shp"  # shapefile they need to intersect
dest_dir = r"/ram/parker/"  # destination directory

import os
import sys
import fnmatch
from osgeo import ogr, gdal, osr

if not os.path.exists(shapefile):
    a = os.system("cp /data/fort_nelson/circle* /ram/parker/")
    if a != 0:
        print("Error: ")
        sys.exit(1)


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


def smart_copy(intersecting_files, destination):
    """
    Copy files intelligently:
    - Skip if destination exists and is >= source size
    - Overwrite if destination exists but is smaller (broken/partial)
    - Copy if destination doesn't exist (new)
    
    Returns counts of: new, broken (overwritten), skipped, and orphaned files
    """
    new_count = 0
    broken_count = 0
    skipped_count = 0
    
    # Track which files we're selecting this run
    selected_filenames = set()
    
    for src_path in intersecting_files:
        filename = os.path.basename(src_path)
        selected_filenames.add(filename)
        dest_path = os.path.join(destination, filename)
        
        src_size = os.path.getsize(src_path)
        
        if os.path.exists(dest_path):
            dest_size = os.path.getsize(dest_path)
            
            if dest_size >= src_size:
                # File exists and is complete - skip
                print(f"SKIPPED (already exists): {filename}")
                skipped_count += 1
            else:
                # File exists but is smaller - broken/partial copy
                print(f"OVERWRITING (broken - {dest_size} < {src_size} bytes): {filename}")
                cmd = f'cp -v "{src_path}" "{destination}"'
                a = os.system(cmd)
                if a == 0:
                    broken_count += 1
                else:
                    print(f"ERROR copying: {src_path}")
        else:
            # File doesn't exist - new
            print(f"COPYING (new): {filename}")
            cmd = f'cp -v "{src_path}" "{destination}"'
            a = os.system(cmd)
            if a == 0:
                new_count += 1
            else:
                print(f"ERROR copying: {src_path}")
    
    # Count orphaned files (exist in destination but not selected this time)
    orphan_count = 0
    orphan_files = []
    if os.path.exists(destination):
        for filename in os.listdir(destination):
            if fnmatch.fnmatch(filename, "*_fcir_*.tif"):
                if filename not in selected_filenames:
                    orphan_count += 1
                    orphan_files.append(filename)
    
    return new_count, broken_count, skipped_count, orphan_count, orphan_files


# go
intersecting = find_intersecting_rasters(root_dir, shapefile)

print("\n=== Intersecting Rasters ===")
for r in intersecting:
    print(r)

print(f"\nTotal intersecting: {len(intersecting)}")

print("\n=== Smart Copy ===")
new, broken, skipped, orphans, orphan_files = smart_copy(intersecting, dest_dir)

print("\n" + "=" * 50)
print("=== COPY SUMMARY ===")
print("=" * 50)
print(f"NEW files copied:                  {new}")
print(f"BROKEN files overwritten:          {broken}")
print(f"SKIPPED (already complete):        {skipped}")
print(f"ORPHANED (not selected this run):  {orphans}")
print("=" * 50)
print(f"Total selected this run:           {len(intersecting)}")

if orphan_files:
    print("\n=== Orphaned Files ===")
    for f in orphan_files:
        print(f"  {f}")
