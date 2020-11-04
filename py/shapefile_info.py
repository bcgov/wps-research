import os
import sys
import json
from osgeo import gdal # need gdal / python installed!
from osgeo import ogr
from osgeo import gdalconst

def err(m):
    print("Error: " + m); sys.exit(1)

# parse arguments 
args = sys.argv
if len(args) < 2:
    err("Error: shapefile_info.py [input shapefile .shp]")

InputVector = args[1] # shapefile to rasterize

# Open Shapefile
Shapefile = ogr.Open(InputVector)
layer = Shapefile.GetLayer()
layerDefinition = layer.GetLayerDefn()
feature_count = layer.GetFeatureCount()

def records(layer):
    # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())

print("feature count: " + str(feature_count))
features = records(layer)
feature_names, feature_ids = [], []
for f in features:  # print(f.keys())
    feature_id = f['id']
    feature_ids.append(feature_id) # print(f['properties'].keys())
    feature_name = ''
    try:
        feature_name = f['properties']['Name']
    except Exception:
        pass # feature name not available
    feature_names.append(feature_name)



