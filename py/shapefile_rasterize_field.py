'''20221025 rasterize a shapefile onto a provided raster extent. Generate one band for each observed value, of the selected attribute'''
import os
import sys
import json
from osgeo import ogr
from osgeo import gdal
from osgeo import gdalconst

stop = False
shp = sys.argv[1] # 'V082M_reproject.shp'
select_key = 'Fuel_Type_'
InputVector = shp

count = {}
# gdal_rasterize -ts 1000 1000 -burn 1 -where "Fuel_Type_='C-5'" V082M_reproject.shp out.tif

# Open Shapefile
Shapefile = ogr.Open(InputVector); layer = Shapefile.GetLayer(); layerDefinition = layer.GetLayerDefn(); feature_count = layer.GetFeatureCount(); spatialRef = layer.GetSpatialRef()
print("CRS:", spatialRef)

def records(layer):
    # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())
print("feature_count," + str(feature_count))
print("feature","type", "x","y"); fi = -1
features = records(layer)
feature_names, feature_ids = [], []
for f in features:
    if fi % 100 == 0:
        print(fi, feature_count)
    fi += 1
    if stop:
        if fi > 10:
            break
    geom = ""
    #print("f.keys()", f.keys())
    for key in f.keys():
        if key=='properties':
            fuel_type = f[key][select_key]
            #print(select_key, '=', fuel_type)
            if fuel_type not in count:
                count[fuel_type] = 0
            count[fuel_type] += 1
    try:
        geom = f['geometry']
    except Exception:
        pass

    feature_id = f['id']
    feature_ids.append(feature_id) # print(f['properties'].keys())
    feature_name = ''
    try:
        feature_name = f['properties']['Name']
    except Exception:
        pass # feature name not available
    feature_names.append(feature_name)

    if geom['type'] == 'Point':
        pass
    else:
        #print(geom['type'])
        for c in geom['coordinates']:
            for d in c:
                pass
                #print("  " + str(d))
print(count)

