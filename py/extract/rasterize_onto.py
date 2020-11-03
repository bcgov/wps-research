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
if len(args) < 4:
    err("Error: rasterize_onto.py: usage:" +
        "  python3 rasterize_onto.py [shapefile to rasterize] [image file: footprint to rasterize onto] [output filename]")
    sys.exit(1)

InputVector = args[1] # shapefile to rasterize
RefImage = args[2] # footprint to rasterize onto
OutputImage = args[3]
if os.path.exists(OutputImage): err("output file already exists")
if OutputImage[-4:] != '.bin': err("output file extension expected: .bin")

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

def records(layer):
    # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())

print("feature count: " + str(feature_count))
features = records(layer)
feature_names, feature_ids = [], []
for f in features:
    # print(f.keys())
    feature_id = f['id']
    feature_ids.append(feature_id)
    # print(f['properties'].keys())
    feature_name = f['properties']['Name']
    feature_names.append(feature_name)
    # print("feature id=", feature_id, "name", feature_name)

# print("Name  -  Type  Width  Precision")
for i in range(layerDefinition.GetFieldCount()):
    fieldName =  layerDefinition.GetFieldDefn(i).GetName()
    fieldTypeCode = layerDefinition.GetFieldDefn(i).GetType()
    fieldType = layerDefinition.GetFieldDefn(i).GetFieldTypeName(fieldTypeCode)
    fieldWidth = layerDefinition.GetFieldDefn(i).GetWidth()
    GetPrecision = layerDefinition.GetFieldDefn(i).GetPrecision()
    # print(fieldName + " - " + fieldType+ " " + str(fieldWidth) + " " + str(GetPrecision))

# Rasterise
print("+w", OutputImage)
Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype)
Output.SetProjection(Image.GetProjectionRef())
Output.SetGeoTransform(Image.GetGeoTransform())

# Write data to band 1
Band = Output.GetRasterBand(1)
Band.SetNoDataValue(0)
gdal.RasterizeLayer(Output, [1], layer, burn_values=[burnVal])

# Close datasets
# Band = None
Output = None
#  Image = None
# Shapefile = None

for i in range(feature_count):
    fid_list = [feature_ids[i]]
    my_filter = "FID in {}".format(tuple(fid_list))
    my_filter = my_filter.replace(",", "")  # the comma in a tuple, if only one element, throws an error
    layer.SetAttributeFilter(my_filter)

    out_fn = OutputImage[:-4] + '_' + str(feature_ids[i]) + '_' + str(feature_names[i].strip() + '.bin'
    print("+w", out_fn)
    
    # Rasterise
    Output = gdal.GetDriverByName(gdalformat).Create(out_fn, Image.RasterXSize, Image.RasterYSize, 1, datatype)
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform())

    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)
    gdal.RasterizeLayer(Output, [1], layer, burn_values=[burnVal])

    Output = None

    # Close datasets
Band = None
Image = None
Shapefile = None

