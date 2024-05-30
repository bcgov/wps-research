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
from osgeo import ogr, osr

args = sys.argv

if len(args) < 4:
    args = [None,
            'prot_current_fire_polys.shp',
            'FIRE_NUM',
            'G90228', 'G90324', 'G90287', 'G90319', 'G90399', 'G90289', 'G90285']

# Path to the input shapefile
input_shapefile_path = args[1] # 'path/to/your/input_shapefile.shp'

# Path to the output shapefile
output_shapefile_path = args[1] + '_select.shp' # spath/to/your/output_shapefile.shp'

# List of values to match
attribute_name = args[2]   #   'your_attribute_name'
attribute_values = args[3:] #  ['value1', 'value2', 'value3']  # Example list of values

# Open the input shapefile
driver = ogr.GetDriverByName('ESRI Shapefile')
datasource = driver.Open(input_shapefile_path, 0)  # 0 means read-only, 1 means writeable

if datasource is None:
    print(f"Could not open {input_shapefile_path}")
else:
    # Get the input layer
    input_layer = datasource.GetLayer()

    # Define the attribute filter
    formatted_values = ", ".join(f"'{value}'" for value in attribute_values)
    filter_expression = f"{attribute_name} IN ({formatted_values})"

    # Set the attribute filter
    input_layer.SetAttributeFilter(filter_expression)

    # Create the output shapefile
    if os.path.exists(output_shapefile_path):
        driver.DeleteDataSource(output_shapefile_path)

    output_datasource = driver.CreateDataSource(output_shapefile_path)
    if output_datasource is None:
        print(f"Could not create {output_shapefile_path}")
    else:
        # Create the output layer with the same spatial reference as the input
        spatial_ref = input_layer.GetSpatialRef()
        output_layer = output_datasource.CreateLayer(
            input_layer.GetName(), spatial_ref, geom_type=input_layer.GetGeomType()
        )

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
                output_feature.SetField(output_layer_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
            output_layer.CreateFeature(output_feature)
            output_feature = None  # Destroy the feature to free resources

        # Cleanup
        output_datasource.Destroy()

    datasource.Destroy()

    print(f"Selected features have been written to {output_shapefile_path}")

