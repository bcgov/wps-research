# this script not finished..
#   .. think the code is from GDAL OGR cookbook

# reproject a shapefile based on EPSG
import os
import sys
from osgeo import ogr
from osgeo import osr
args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

if len(args) < 4:
    err("reproject_shp.py [input shapefile] [new CRS: EPSG code] [output shapefile]")

# in_epsg = 5514
out_epsg = int(args[2]) #  4326
# in_shp = 'path/to/input.shp'
# out_shp = '/path/to/reprojected.shp'
in_shp = args[1]
out_shp = args[3]

driver = ogr.GetDriverByName('ESRI Shapefile')

# get the input layer
inDataSet = driver.Open(in_path)
inLayer = inDataSet.GetLayer()


# input SpatialReference
inSpatialRef = osr.SpatialReference()
inSpatialRef.ImportFromEPSG(in_epsg)

# output SpatialReference
outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(out_epsg)

# create the CoordinateTransformation
coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

# create the output layer
if os.path.exists(out_shp):
    driver.DeleteDataSource(out_shp)
outDataSet = driver.CreateDataSource(out_shp)
outLayer = outDataSet.CreateLayer("reproject", geom_type=ogr.wkbMultiPolygon)

# add fields
inLayerDefn = inLayer.GetLayerDefn()
for i in range(0, inLayerDefn.GetFieldCount()):
    fieldDefn = inLayerDefn.GetFieldDefn(i)
    outLayer.CreateField(fieldDefn)

# get the output layer's feature definition
outLayerDefn = outLayer.GetLayerDefn()

# loop through the input features
inFeature = inLayer.GetNextFeature()
while inFeature:
    # get the input geometry
    geom = inFeature.GetGeometryRef()
    # reproject the geometry
    geom.Transform(coordTrans)
    # create a new feature
    outFeature = ogr.Feature(outLayerDefn)
    # set the geometry and attribute
    outFeature.SetGeometry(geom)
    for i in range(0, outLayerDefn.GetFieldCount()):
        outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
    # add the feature to the shapefile
    outLayer.CreateFeature(outFeature)
    # dereference the features and get the next input feature
    outFeature = None
    inFeature = inLayer.GetNextFeature()

# Save and close the shapefiles
inDataSet = None
outDataSet = None


