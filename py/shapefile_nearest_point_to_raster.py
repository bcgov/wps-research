'''20230712 match point-shapefile locations to raster
* Assume one shapefile-point is in the raster and/or:
* find the nearest shapefile-point to raster footprint 
      centroid


* Make sure raster and shapefile are in same CRS.
E.g. I transformed both to EPSG 4326 first.

Does this script actually work?

I didn't get the right answer yet.'''
from osgeo import gdal
from osgeo import ogr
import sys
args = sys.argv

def err(m):
    print("Error", m); sys.exit(1)

if len(args) < 3:
    err("shapefile_nearest_point_to_raster.py [input point shapefile] [ input raster]")

shapefile_path, raster_path = args[1], args[2]
shapefile_ds = ogr.Open(shapefile_path)
raster_ds = gdal.Open(raster_path)

raster_extent = raster_ds.GetGeoTransform()
raster_centroid_x = (raster_extent[0] + raster_extent[1]) / 2
raster_centroid_y = (raster_extent[3] + raster_extent[5]) / 2
raster_centroid = ogr.Geometry(ogr.wkbPoint)
raster_centroid.AddPoint(raster_centroid_x, raster_centroid_y)

layer = shapefile_ds.GetLayer()
nearest_point, nearest_feature = None, None
nearest_distance = float('inf')

for feature in layer:
    point_geometry = feature.GetGeometryRef()
    distance = point_geometry.Distance(raster_centroid)
    print(feature.GetFieldAsString(0), point_geometry, raster_centroid)

    if distance < nearest_distance:
        nearest_distance = distance
        nearest_point = point_geometry
        nearest_feature = feature

# Print the nearest point coordinates
if nearest_point is not None:
    x = nearest_point.GetX()
    y = nearest_point.GetY()
    print(f"Nearest point: ({x}, {y})")
    print(nearest_feature)

    feature = nearest_feature
    num_fields = feature.GetFieldCount()
    
    # Iterate through each field
    for i in range(num_fields):
        # Get the field name and value
        field_name = feature.GetFieldDefnRef(i).GetName()
        field_value = feature.GetFieldAsString(i)
        
        # Print the field name and value
        print(f"{field_name}:{field_value}")
else:
    print("No points found in the shapefile.")
