'''20230712 match point-shapefile locations to raster

* find the nearest shapefile-point to raster footprint 
      centroid

* Condition: make sure raster and shp are in same CRS.
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


'''
# Shared by Conor: #1
import geopandas as gpd
import rasterio
from shapely.geometry import Point
# Load the shapefile containing point geometries
points_shapefile = 'path/to/points.shp'
points = gpd.read_file(points_shapefile)
# Load the raster file
raster_file = 'path/to/raster.tif'
raster = rasterio.open(raster_file)
# Get the raster footprint as a polygon
raster_footprint = raster.bounds
raster_polygon = gpd.GeoSeries(raster_footprint).envelope
# Create an empty list to store intersecting points
intersecting_points = []
# Iterate over each point geometry in the shapefile
for point in points.geometry:
    # Check if the point intersects with the raster footprint
    if point.intersects(raster_polygon.iloc[0]):
        intersecting_points.append(point)
# Print the intersecting points
for point in intersecting_points:
    print(point)
# Optional: Convert intersecting points to a new GeoDataFrame
intersecting_points_gdf = gpd.GeoDataFrame(geometry=intersecting_points)
intersecting_points_gdf.crs = points.crs
# Save the intersecting points as a new shapefile
intersecting_points_shapefile = 'path/to/intersecting_points.shp'
intersecting_points_gdf.to_file(intersecting_points_shapefile)
'''

'''
# Shared by Conor: #2
from osgeo import gdal, ogr
# Open the shapefile
shapefile = 'path/to/points.shp'
shapefile_ds = ogr.Open(shapefile)
layer = shapefile_ds.GetLayer()
# Open the raster
raster_file = 'path/to/raster.tif'
raster_ds = gdal.Open(raster_file)
# Get raster properties
raster_band = raster_ds.GetRasterBand(1)
raster_transform = raster_ds.GetGeoTransform()
# Create an output shapefile to store intersecting points
output_shapefile = 'path/to/intersecting_points.shp'
driver = ogr.GetDriverByName('ESRI Shapefile')
output_ds = driver.CreateDataSource(output_shapefile)
output_layer = output_ds.CreateLayer('intersecting_points', geom_type=ogr.wkbPoint)
# Add fields to the output layer
field_def = ogr.FieldDefn('id', ogr.OFTInteger)
output_layer.CreateField(field_def)
# Iterate over the points in the shapefile
for feature in layer:
    point = feature.GetGeometryRef()
    x, y = point.GetX(), point.GetY()
    # Convert point coordinates to pixel coordinates
    pixel_x = int((x - raster_transform[0]) / raster_transform[1])
    pixel_y = int((y - raster_transform[3]) / raster_transform[5])
    # Read the pixel value from the raster
    pixel_value = raster_band.ReadAsArray(pixel_x, pixel_y, 1, 1)[0][0]
    # Check if the pixel value is valid (e.g., not nodata)
    if pixel_value != raster_band.GetNoDataValue():
        # Create a new point feature in the output shapefile
        output_feature = ogr.Feature(output_layer.GetLayerDefn())
        output_feature.SetGeometry(point)
        output_feature.SetField('id', feature.GetField('id'))
        output_layer.CreateFeature(output_feature)
        output_feature = None
# Cleanup
shapefile_ds = None
raster_ds = None
output_ds = None
'''
