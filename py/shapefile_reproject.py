import sys
from osgeo import ogr
from osgeo import gdal
from osgeo import gdalconst
from misc import err, args, exist

if len(args) < 4:
    err('python3 shapefile_reproject.py [input shapefile] [output CRS (EPSG)] [output shapefile]')
# https://gis.stackexchange.com/questions/265589/change-shapefile-coordinate-system-using-python

in_shp = args[1]
if not exist(in_shp):
    err('please check input file')

err("implementation not finished")
sys.exit(1)

in_epsg = 5514
out_epsg = 4326
in_shp = 'path/to/input.shp'
out_shp = '/path/to/reprojected.shp'

driver = ogr.GetDriverByName('ESRI Shapefile')

# input SpatialReference
inSpatialRef = osr.SpatialReference()
inSpatialRef.ImportFromEPSG(in_epsg)

# output SpatialReference
outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(out_epsg)

# create the CoordinateTransformation
coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

# get the input layer
inDataSet = driver.Open(in_path)
inLayer = inDataSet.GetLayer()

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

