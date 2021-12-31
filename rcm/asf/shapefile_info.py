# min size in hectares
MIN_SIZE = 3000
MIN_YEAR = 2006
MAX_YEAR = 2011
PROV = 'BC'

import os
import sys
import json
from osgeo import gdal # need gdal / python installed!
from osgeo import ogr
from osgeo import gdalconst

def err(m):
    print("Error: " + m); sys.exit(1)
args = sys.argv  # arguments
if len(args) < 2: err("Error: shapefile_info.py [input shapefile .shp]")
InputVector = args[1] # shapefile to rasterize

Shapefile = ogr.Open(InputVector) # open shp
layer = Shapefile.GetLayer()
layerDefinition = layer.GetLayerDefn()
feature_count = layer.GetFeatureCount()

def records(layer):
    # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())

print("feature_count," + str(feature_count))
print("feature","type", "x","y")
features = records(layer)
feature_names, feature_ids = [], []
for f in features:  
    geom = ""
    #print("keys", f.keys())
    #for k in f.keys():
    #    print('  ', k, ':', f[k])
    try:
        geom = f['geometry']
    except Exception:
        pass

    year = int(f['properties']['YEAR'])
    prov = f['properties']['SRC_AGENCY'].strip()
    
    if year <= MAX_YEAR and year >= MIN_YEAR:
        if prov == PROV:
            ix = f['properties']['FIRE_ID']
            sz = f['properties']['SIZE_HA']
            sz = int(sz)
            if sz > MIN_SIZE:
                print(ix, year, sz) #             print(f['properties'])

    feature_id = f['id']
    feature_ids.append(feature_id) # print(f['properties'].keys())
    feature_name = ''
    try:
        feature_name = f['properties']['Name']
    except Exception:
        pass # feature name not available
    feature_names.append(feature_name)

    if True:
        continue
    if geom['type'] == 'Point':
        stuff = [feature_id,
                 geom['type'],
                 geom['coordinates'][0],
                 geom['coordinates'][1]]
        print(','.join([str(x) for x in stuff]))
    else:
        print(geom['type'])
        for c in geom['coordinates']:
            for d in c:
                print("  " + str(d))
