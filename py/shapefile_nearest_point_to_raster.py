'''20230712 match point shapefile locations to raster. Assume one point is in the raster.'''
from osgeo import gdal
from osgeo import ogr
from misc import args, err

if len(args) < 3:
    err("shapefile_nearest_point_to_raster.py [input point shapefile] [ input raster]")

shapefile_path = args[1]
raster_path = args[2]

shapefile_ds = ogr.Open(shapefile_path)
raster_ds = gdal.Open(raster_path)

raster_extent = raster_ds.GetGeoTransform()
raster_centroid_x = (raster_extent[0] + raster_extent[1])# / 2
raster_centroid_y = (raster_extent[3] + raster_extent[5])# / 2
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


'''
If the centroid of the raster footprint appears to be off and needs to be multiplied by 2, it suggests that there might be an issue with the transformation to the desired coordinate system (EPSG 4326) or the interpretation of the resulting coordinates.

Here are a few potential reasons why this could occur:

    Coordinate System Mismatch: It's possible that the shapefile or the raster is not actually being transformed to EPSG 4326 (WGS 84) during the coordinate transformation process. Double-check that the correct EPSG code (4326) is being used for the transformation and that both the shapefile and raster are transformed using the same code.

    Unit Conversion: Ensure that the units of the shapefile and raster are correctly accounted for during the transformation. For example, if the shapefile is in meters and the raster is in degrees, the resulting coordinates will not align properly. Make sure that the units of both datasets are compatible and consistent.

    Incorrect Extent or Resolution: Verify that the extent and resolution of the raster are correctly interpreted and applied during the transformation. If the extent or resolution is not properly considered, it can lead to inconsistencies in the resulting coordinates.

    Data Interpretation: Check how the transformed coordinates are being interpreted or used in subsequent calculations. It's possible that there might be an error in the calculation or interpretation of the centroid coordinates, which is unrelated to the transformation itself.

To diagnose the issue, you can print and inspect the transformed coordinates of some specific points in both the shapefile and the raster, both before and after the transformation. This can help identify any discrepancies or inconsistencies in the transformation process.

Additionally, verifying the correctness of the transformed coordinates by comparing them with known reference points or using visualization techniques can aid in identifying the problem.
'''
