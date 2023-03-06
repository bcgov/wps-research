'''20221025 rasterize a shapefile onto a provided raster extent. Generate one band for each observed value, of the selected attribute'''
import os
import sys
import json
from osgeo import ogr
from osgeo import gdal
from osgeo import gdalconst
from misc import err, run, args, read_hdr, write_hdr
if len(args) < 3:
    err("shapefile_rasterize_field.py [input shapefile] [raster footprint to be matched by output]")

stop = False
shp = sys.argv[1] # 'V082M_reproject.shp'
raster = sys.argv[2] # raster footprint for output to match
select_key = 'FL_TYP_CD' # 'Fuel_Type_'
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

if not os.path.exists(".counts.txt"):
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
                fuel_type = f[key][select_key] #print(select_key, '=', fuel_type)
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
else:
    exec("count = " + open(".counts.txt").read().strip())
print(count)

i = 0
ixs = []
for c in count:
    ix = str(i).zfill(2); ixs.append(ix)
    out_f = shp + '_' + ix + '.bin'
    if not os.path.exists(out_f):
        cmd = 'gdal_rasterize -of ENVI -ot Float32 -ts 10000 10000 -burn 1 -where "' + select_key + "='" + c + "'" + '" ' + shp + ' ' + out_f
        run(cmd)

    result_file = ix + '.bin'
    if not os.path.exists(result_file):
        cmd = 'po ' + out_f + ' ' + raster + ' ' + result_file + ' 1'
        run(cmd)
    i += 1

samples, lines, bands = read_hdr(raster[:-3] + 'hdr')
band_names = [c for c in count]
file_names = [ix + '.bin' for ix in ixs]
if not os.path.exists('raster.bin'):
    cmd = 'cat ' + (' '.join(file_names)) + ' > raster.bin'
    run(cmd)
    write_hdr('raster.hdr', samples, lines, len(ixs), band_names)
