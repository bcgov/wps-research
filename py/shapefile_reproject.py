'''
References:
[1] https://pcjericks.github.io/py-gdalogr-cookbook/projection.html
[2] https://gdal.org/programs/gdalsrsinfo.html
'''
import sys
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
from osgeo import gdalconst
from misc import err, args, exist

if len(args) < 4:
    err('python3 shapefile_reproject.py [input shapefile] [shapefile or raster to get CRS from] [output shapefile]')
# https://gis.stackexchange.com/questions/265589/change-shapefile-coordinate-system-using-python

in_path = args[1]
out_shp = args[3]
if not exist(in_path) or not exist(args[2]):
    err('please check input files')

if exist(out_shp):
    err('output file already exists')

driver = ogr.GetDriverByName('ESRI Shapefile')
dataset = driver.Open(in_path)  # input layer
layer = dataset.GetLayer()
sr = layer.GetSpatialRef()

sr2 = None  # spatial reference to be matched in reprojected dataset
try:
	print("Try to interpret as RASTER")
	d2 = gdal.Open(args[2])
	sr2 = osr.SpatialReference(wkt=d2.GetProjection())
except:
	driver2 = ogr.GetDriverByName('ESRI Shapefile')
	dataset2 = driver.Open(args[2])
	layer2 = dataset.GetLayer()  # from layer
	sr2 = layer.GetSpatialRef()
	print("interpreted as SHAPEFILE")
	'''
	# from Geometry
	feature = layer.GetNextFeature()
	geom = feature.GetGeometryRef()
	spatialRef = geom.GetSpatialReference()'''

#print("CRS to mATCH:", sr2)
# proj = osr.SpatialReference(wkt=d.GetProjection())
# print(proj.GetAttrValue('AUTHORITY',1))

# create the CoordinateTransformation
coordTrans = osr.CoordinateTransformation(sr, sr2)

# create the output layer
outDataSet = driver.CreateDataSource(out_shp)
geom_type=layer.GetLayerDefn().GetGeomType()
print("GEOM_TYPE", geom_type, "ogr.wkbMultiPolygon", ogr.wkbMultiPolygon)
outLayer = outDataSet.CreateLayer("reproject", sr2, geom_type=layer.GetLayerDefn().GetGeomType()) # ogr.wkbMultiPolygon)

# add fields
layerDefn = layer.GetLayerDefn()
for i in range(0, layerDefn.GetFieldCount()):
    fieldDefn = layerDefn.GetFieldDefn(i)
    outLayer.CreateField(fieldDefn)
    # outLayer.SetSpatialRef(sr2)

# get the output layer's feature definition
outLayerDefn = outLayer.GetLayerDefn()

# loop through the input features
inFeature = layer.GetNextFeature()
while inFeature:   # get the input geometry
    geom = inFeature.GetGeometryRef()     # reproject the geometry
    geom.Transform(coordTrans)     # create a new feature
    outFeature = ogr.Feature(outLayerDefn)      # set the geometry and attribute
    outFeature.SetGeometry(geom)
    for i in range(0, outLayerDefn.GetFieldCount()):
        outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))   # add the feature to the shapefile
    outLayer.CreateFeature(outFeature) # dereference the features and get the next input feature
    outFeature = None
    inFeature = layer.GetNextFeature()

dataset = None  # save and close shapefiles
outDataSet = None
