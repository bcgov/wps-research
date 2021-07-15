'''this script is not tested. Not sure what units'''
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

InputVector = args[1] # shapefile

# Open Shapefile
Shapefile = ogr.Open(InputVector)
layer = Shapefile.GetLayer()


#new_field = ogr.FieldDefn("Area", ogr.OFTReal)
#new_field.SetWidth(32)
#new_field.SetPrecision(2) #added line to set precision
#layer.CreateField(new_field)

for feature in layer:
    geom = feature.GetGeometryRef()
    area = geom.GetArea()
    print(area)
    #feature.SetField("Area", area)
    #layer.SetFeature(feature)
#dataSource = None
