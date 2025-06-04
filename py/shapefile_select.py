''' 20240529 shapefile_select.py

Create a new file limited to features where:
    field \in {"value1", "value2", ...}

usage: e.g. 
    python3 shapefile_select.py [input shapefile] [attribute] [value1] [value2] .. [valueN]
e.g.
    python3 shapefile_select.py prot_current_fire_polys.shp FIRE_NUM G90228 G90324 G90287 G90319 G90399 G90289 G90285
'''
import os
import sys
args = sys.argv
from osgeo import ogr as ogr
from osgeo import osr as osr
from osgeo.ogr import GetDriverByName
from misc import err, exist, assert_exists

if len(args) < 4:
    args = [None,
            'prot_current_fire_polys.shp',
            'FIRE_NUM',
            'G90207', 'G90228', 'G90319', 'G90287', 'G90399', 'G90289', 'G90324']   #'G90228', 'G90324', 'G90287', 'G90319', 'G90399', 'G90289', 'G90285', 'G80628']

input_shapefile_path = args[1]
output_shapefile_path = args[1] + '_selected.shp' #  ('.'.join(args[1].split('.')[:-1])) + '_select.shp'

assert_exists(input_shapefile_path)
print('+r', input_shapefile_path)

attribute_name = args[2]   # attribute / column name to match
attribute_values = args[3:] # list of values to match
print('attribute_name', attribute_name)
print('attribute_values', attribute_values)
datasource = GetDriverByName('ESRI Shapefile').Open(input_shapefile_path, 0)  # 0 means read-only, 1 means write

if datasource is None:
    err(f"Could not open {input_shapefile_path}")

input_layer = datasource.GetLayer()  # input layer and attribute filter
formatted_values = ", ".join(f"'{value}'" for value in attribute_values)
filter_expression = f"{attribute_name} IN ({formatted_values})"
input_layer.SetAttributeFilter(filter_expression) # set attr filter

if exist(output_shapefile_path):  # output shapefile
    print('rm ' + output_shapefile_path)
    GetDriverByName('ESRI Shapefile').DeleteDataSource(output_shapefile_path)

output_datasource = GetDriverByName('ESRI Shapefile').CreateDataSource(output_shapefile_path) # driver.Open(output_shapefile_path, 1) 
if output_datasource is None:
    err(f'could not create output shapefile: {output_shapefile_path}')

# Create the output layer with the same spatial reference as the input
spatial_ref = input_layer.GetSpatialRef()
output_layer = output_datasource.CreateLayer(input_layer.GetName(),
                                             spatial_ref,
                                             geom_type=input_layer.GetGeomType())

# Copy the fields from the input layer to the output layer
layer_defn = input_layer.GetLayerDefn()
for i in range(layer_defn.GetFieldCount()):
    field_defn = layer_defn.GetFieldDefn(i)
    output_layer.CreateField(field_defn)

# Get the output layer definition
output_layer_defn = output_layer.GetLayerDefn()

# Iterate through the filtered features and add them to the output layer
for feature in input_layer:
    output_feature = ogr.Feature(output_layer_defn)
    output_feature.SetGeometry(feature.GetGeometryRef().Clone())
    for i in range(output_layer_defn.GetFieldCount()):
        output_feature.SetField(output_layer_defn.GetFieldDefn(i).GetNameRef(),
                                feature.GetField(i))
    output_layer.CreateFeature(output_feature)
    output_feature = None  # Destroy the feature to free resources

# Cleanup
output_datasource.Destroy()
datasource.Destroy()
print(f"Selected features have been written to {output_shapefile_path}")

