# this script extracts spectra (from a raster) at point locations in a shapefile

# next: extract on grid pattern

import os
import sys
import json
import struct
from osgeo import gdal
from osgeo import ogr
args = sys.argv

def err(m):
    printf("Error: " + str(m)); sys.exit(1)

if len(args) < 3:
    err("python3 extract_spectra.py [input shapefile name] [input image name]")

shp = args[1] # input shapefile
img = args[2]
if not os.path.exists(shp): err('file not found: ' + shp)
if not os.path.exists(img): err('file not found: ' + img)

# Open image
Image = gdal.Open(img, gdal.GA_ReadOnly)
nc, nr, nb = Image.RasterXSize, Image.RasterYSize, Image.RasterCount # rows, cols, bands

# Open Shapefile
Shapefile = ogr.Open(shp)
layer = Shapefile.GetLayer()
layerDefinition, feature_count = layer.GetLayerDefn(), layer.GetFeatureCount()

def records(layer):
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())
print("feature count: " + str(feature_count))
features = records(layer)

feature_names, feature_ids, coordinates = [], [], []
for f in features: # print(f.keys())
    feature_id = f['id']
    feature_ids.append(feature_id) # print("feature properties.keys()", f['properties'].keys())
    feature_name = ''
    try:
        feature_name = f['properties']['Name']
    except Exception:
        pass # feature name not available
    feature_names.append(feature_name)
    
    # print("feature id=", feature_id, "name", feature_name)
    # print("feature geometry=", f['geometry'])
    fgt = f['geometry']['type']
    if fgt != 'Point':
        err('Point geometry expected. Found geometry type: ' + str(fgt))
    coordinates.append(f['geometry']['coordinates'])

count = 0
# extract spectra
for i in range(feature_count):
    # print(feature_ids[i], coordinates[i])

    cmd = ["gdallocationinfo",
           img,
           '-wgs84',
           str(coordinates[i][0]),
           str(coordinates[i][1])]
    cmd = ' '.join(cmd)
    # print(cmd) 

    lines = [x.strip() for x in os.popen(cmd).readlines()]
    if len(lines) >= 2 * (1 + nb):
        print(lines)
        count += 1
print("number of spectra extracted:", count)
