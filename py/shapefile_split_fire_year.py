'''20260326 split a shapefile with FIRE_YEAR attribute, into separate shapefiles by year
'''
from osgeo import ogr, osr
import os
import sys

# Default input
input_shp = "IN_HISTORICAL_FIRE_POLYGONS_SVW.shp"

# Override from command line if provided
if len(sys.argv) > 1:
    input_shp = sys.argv[1]

output_dir = "output_fire_years"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open input shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
ds = driver.Open(input_shp, 0)  # 0 = read-only
if ds is None:
    raise RuntimeError(f"Could not open input shapefile: {input_shp}")

layer = ds.GetLayer()

# Get spatial reference
spatial_ref = layer.GetSpatialRef()

# Get layer definition (schema)
layer_defn = layer.GetLayerDefn()

# Dictionary to hold output datasources by FIRE_YEAR
outputs = {}

for feature in layer:
    fire_year = feature.GetField("FIRE_YEAR")

    if fire_year is None:
        continue

    fire_year = str(fire_year)

    # Create output shapefile if not already created
    if fire_year not in outputs:
        out_path = os.path.join(
            output_dir,
            f"IN_HISTORICAL_FIRE_POLYGONS_SVW_{fire_year}.shp"
        )

        # Remove if exists (shapefile = multiple files)
        if os.path.exists(out_path):
            driver.DeleteDataSource(out_path)

        out_ds = driver.CreateDataSource(out_path)
        out_layer = out_ds.CreateLayer(
            "fires",
            srs=spatial_ref,
            geom_type=layer.GetGeomType()
        )

        # Copy fields
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            out_layer.CreateField(field_defn)

        outputs[fire_year] = (out_ds, out_layer)

    out_ds, out_layer = outputs[fire_year]

    # Create new feature
    out_feature = ogr.Feature(out_layer.GetLayerDefn())

    # Copy attributes
    for i in range(layer_defn.GetFieldCount()):
        out_feature.SetField(
            layer_defn.GetFieldDefn(i).GetNameRef(),
            feature.GetField(i)
        )

    # Copy geometry
    geom = feature.GetGeometryRef()
    if geom is not None:
        out_feature.SetGeometry(geom.Clone())

    # Add feature to output layer
    out_layer.CreateFeature(out_feature)

    out_feature = None  # free memory

# Cleanup
for out_ds, _ in outputs.values():
    out_ds = None

ds = None

print(f"Done splitting shapefile by FIRE_YEAR from: {input_shp}")
