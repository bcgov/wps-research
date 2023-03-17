'''20230316 extract Sentinel-2 rowID for each footprint over BC'''
import os
import sys
import json
from osgeo import ogr
from osgeo import gdal
from osgeo import gdalconst
args = sys.argv
sep = os.path.sep
abspath = os.path.abspath
def get_pd():
    return abspath(sep.join(abspath(__file__).split(sep)[:-1])) + sep  # python directory i.e. path to here

InputVector = get_pd() + "Sentinel_BC_Tiles.shp"  # open Shapefile
Shapefile = ogr.Open(InputVector)
layer = Shapefile.GetLayer()
layerDefinition = layer.GetLayerDefn()
feature_count = layer.GetFeatureCount()
spatialRef = layer.GetSpatialRef()

def records(layer): # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())
features, feature_names, feature_ids, row_id = records(layer), [], [], []
for f in features:  
    geom = "" 
    # print(f.keys())
    for key in f.keys():
        if key=='properties':
            pass # print(f[key])
    try: geom = f['geometry']
    except Exception: pass

    feature_id = f['id']
    feature_ids.append(feature_id) # print(f['properties'].keys())
    feature_name = ''
    try: feature_name = f['properties']['Name']
    except Exception: pass # feature name not available
    feature_names.append(feature_name)

    if geom['type'] == 'Point':
        stuff = [feature_id,
                 geom['type'],
                 geom['coordinates'][0],
                 geom['coordinates'][1]]
        print(','.join([str(x) for x in stuff]))
    else:
        #print(geom['type'])
        for c in geom['coordinates']:
            coords = list(c) #print(coords)

            s = 'Intersects(POLYGON(('
            for i in range(len(coords)):
                if i > 0: 
                    s += ','
                s += (str(coords[i][0]) + ' ' + str(coords[i][1]))
            s += ')))'
            row_id += [feature_name] # print(feature_name, s)
print(' '.join(row_id))
