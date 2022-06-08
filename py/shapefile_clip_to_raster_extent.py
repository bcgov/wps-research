'''20220504 clip shapefile to raster extent, adapted from:
https://gis.stackexchange.com/questions/275086/clipping-polygon-precisely-to-raster-extent-using-gdal

Warning: may need to project to EPSG:4326 (or other!) before using this..

Use shapefile_reproject.py to do this!'''
from misc import err, run, args, exists
from osgeo import gdal
import sys

if len(args) < 4:
    err('usage: python3 clip_shapefile_to_raster_extent.py ' +
        '[input raster] [input vector] [output vector]')
inRasterPath, inVectorPath, outVectorPath = args[1: 4]  # file paths

if exists(outVectorPath):
    err('output file already exists: ' + outVectorPath)

if not exists(inRasterPath) or not exists(inVectorPath):
    err('please check input files')

# get the extent of the raster
src = gdal.Open(inRasterPath)
ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
sizeX = src.RasterXSize * xres
sizeY = src.RasterYSize * yres
lrx = ulx + sizeX
lry = uly + sizeY

extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly)
run(['ogr2ogr',  # clip command: default to shapefile
     outVectorPath,
     inVectorPath,
     '-clipsrc',
     extent])
