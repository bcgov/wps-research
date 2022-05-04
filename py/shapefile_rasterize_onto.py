''' rasterize a shapefile, separate raster for each "feature"! e.g.
        
        python3 rasterize_onto.py \
                boundary.shp \
                S2A_MSIL2A_20190908T195941_N0213_R128_T09VUE_20190908T233509_RGB.tif \
                boundary.bin

        python3 rasterize_onto.py \
                FTL_test1.shp \
                S2A_MSIL2A_20190908T195941_N0213_R128_T09VUE_20190908T233509_RGB.tif \
                out.bin

        python3 shapefile_rasterize_onto.py \
                large_fires_reproject_clip.shp \
                stack.bin_EPSG4326.bin \
                rasterize.bin
20220504: output feature names + values (special case)'''
import os
import sys
import json
import math
import datetime
import numpy as np
from osgeo import ogr
from osgeo import gdal # need gdal / python installed!
from osgeo import gdalconst
from misc import err, exist, args, hdr_fn, read_hdr, write_binary, write_hdr, read_binary

if len(args) < 4:
    err('python3 rasterize_onto.py [shapefile to rasterize] ' +
       '[image file: footprint to rasterize onto] [output filename]')

InputVector = args[1] # shapefile to rasterize
RefImage = args[2] # footprint to rasterize onto
OutputImage = args[3]

if os.path.exists(OutputImage):
    err("output file already exists")

if OutputImage[-4:] != '.bin':
    err("output file extension expected: .bin")

# data output formatting
gdalformat = 'ENVI'
datatype = gdal.GDT_Float32 # Byte
burnVal = 1. #value for the output image pixels

# Get projection info from reference image
Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

# Open Shapefile
Shapefile = ogr.Open(InputVector)
layer = Shapefile.GetLayer()
layerDefinition = layer.GetLayerDefn()
feature_count = layer.GetFeatureCount()

def records(layer): # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())

print("feature count: " + str(feature_count))
features = records(layer)
feature_names, feature_ids = [], []
is_fire = False

for f in features:
    # print("f.keys()", f.keys())
    feature_id = str(f['id']).zfill(4)
    feature_ids.append(feature_id)
    # print("f['properties'].keys()", f['properties'].keys())
    '''f['properties'].keys():
         dict_keys(['SRC_AGENCY', 'FIRE_ID', 'FIRENAME', 'YEAR', 'MONTH',
                    'DAY', 'REP_DATE', 'DATE_TYPE', 'OUT_DATE', 'DECADE',
                    'SIZE_HA', 'CALC_HA', 'CAUSE', 'MAP_SOURCE',
                    'SOURCE_KEY', 'MAP_METHOD', 'WATER_REM',
                    'UNBURN_REM', 'MORE_INFO', 'POLY_DATE', 'CFS_REF_ID',
                    'CFS_NOTE1', 'CFS_NOTE2', 'AG_SRCFILE', 'ACQ_DATE',
                    'SRC_AGY2']) 
    '''
    feature_name = ''
    try:
        feature_name = f['properties']['Name']
    except Exception:
        pass # feature name not available

    prop = f['properties']
    keys = prop.keys()
    
    if 'FIRE_ID' in keys:
        # this is Canadian NFDB
        FIRE_ID = prop['FIRE_ID']
        YEAR = str(prop['YEAR']).zfill(4)  # YYYY
        MONTH = str(prop['MONTH']).zfill(2)  # MM
        DAY = str(prop['DAY']).zfill(2)  # DD
        SIZE = str(int(round(prop['SIZE_HA'], 0))).zfill(8)
        feature_name = ''.join([YEAR, 
                                MONTH, 
                                DAY,
                                '_',
                                FIRE_ID.ljust(12, '_'),
                                '_',
                                SIZE])
        is_fire = True
    
    feature_names.append(feature_name)
    print("feature id=", feature_id, "name", feature_name)

# print("Name  -  Type  Width  Precision")
for i in range(layerDefinition.GetFieldCount()):
    fieldName =  layerDefinition.GetFieldDefn(i).GetName()
    fieldTypeCode = layerDefinition.GetFieldDefn(i).GetType()
    fieldType = layerDefinition.GetFieldDefn(i).GetFieldTypeName(fieldTypeCode)
    fieldWidth = layerDefinition.GetFieldDefn(i).GetWidth()
    GetPrecision = layerDefinition.GetFieldDefn(i).GetPrecision()
    if False:
        print(fieldName + " - " +
              fieldType+ " " +
              str(fieldWidth) + " " +
              str(GetPrecision))

'''
# Rasterise all features to same layer (coverage of all features)
print("+w", OutputImage)
'''
gd = gdal.GetDriverByName(gdalformat)
'''
Output = gd.Create(OutputImage,
                   Image.RasterXSize,
                   Image.RasterYSize,
                   1,
                   datatype)

Output.SetProjection(Image.GetProjectionRef())
Output.SetGeoTransform(Image.GetGeoTransform())
Band = Output.GetRasterBand(1) # write data to band 1
Band.SetNoDataValue(0)

gdal.RasterizeLayer(Output,
                    [1],
                    layer,
                    burn_values=[burnVal])

Output = None  # close ds
'''
out_files = []
for i in range(feature_count): # confirm feature intersects reference map first?
    fid_list = [feature_ids[i]]
    my_filter = "FID in {}".format(tuple(fid_list))
    my_filter = my_filter.replace(",", "")  # comma in tuple throws error for single element
    layer.SetAttributeFilter(my_filter)
    X = ('_' + str(feature_names[i]).strip())
    out_fn = (OutputImage[:-4] + '_' +
              str(feature_ids[i]) +
              ('' if feature_names[i] == ''  else ('_' + X)) +
              '.bin')

    if 'FIRE_ID' in keys:
        out_fn = '_'.join([feature_names[i],
                           OutputImage[:-4],
                           feature_ids[i] + '.bin'])
    
    out_files.append(out_fn)  # record output filename for post-processing
    if exist(out_fn):
        continue
    print("+w",
          out_fn)  # feature_names[i])
    
    # rasterise
    Output = gd.Create(out_fn,
                       Image.RasterXSize,
                       Image.RasterYSize,
                       1,
                       datatype)
    
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform())

    # write data to band1
    Band = Output.GetRasterBand(1) 
    Band.SetNoDataValue(0)
    
    gdal.RasterizeLayer(Output,
                        [1],
                        layer,
                        burn_values=[burnVal])
    Output = None

# close datasets
Band = None
Image = None
Shapefile = None

today = datetime.date.today()
today = datetime.datetime(today.year, today.month, today.day)

if is_fire:  # post-processing of fire data
    out_files.sort()  # sort in time!
    nc, nr, nb = [int(i)
                  for i in read_hdr(hdr_fn(out_files[0]))]

    npx = nr * nc  # number of pixels
    dat = np.array([np.nan for i in range(npx)])
    
    for f in out_files:
        x = f.split('_')[0]
        YYYY, MM, DD = [int(i) for i in [x[0:4], x[4:6], x[6:8]]]
        if MM == 0 and DD == 0:
            MM, DD = 6, 15  # assume middle of year
        # print([x, YYYY,MM,DD])
        t = datetime.datetime(YYYY, MM, DD)
        d = float((t - today).days)  # larger value: more recent

        samples, lines, bands, data = read_binary(f)
        [samples, lines, bands] = [int(i) for i in [samples, lines, bands]]
        if samples*lines != npx:
            err("unexpected image dimension")

        for i in range(npx):
            if data[i] == 1.:
                if np.isnan(dat[i]):
                    dat[i] = d
                else:
                    dat[i] = math.max(dat[i], d)

    ofn = OutputImage[:-4] + '_days_since_burn.bin'
    print('+w', ofn)
    write_binary(dat, ofn)
