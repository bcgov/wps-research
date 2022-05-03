'''20220502 determine pixel location (lat lon) for each point in a raster
note this doesn't generalize to southern hemisphere'''
import os
import utm
import pyproj
from osgeo import gdal
from misc import args, err, run, read_hdr, hdr_fn

if len(args) < 2:
    err("python3 raster_pixels_location.py [input image name]")

img = args[1]  # open image and get geotransform
if not img:
    err('pls check input file')
src = gdal.Open(img)
GT = src.GetGeoTransform()
ncol, nrow, nband = [int(x) for x in read_hdr(hdr_fn(img))]
d = os.popen('gdalsrsinfo -o proj4 ' + img).read().strip()
w = [x.strip('+').split('=') for x in d.split()]
srs_info = {x[0]: x[1] if len(x) > 1 else '' for x in w}

print('X_pixel,Y_line,X_geo,Y_geo,Lat,Lon')
for Y_line in range(nrow): # row 0-idx
    for X_pixel in range(ncol): # col 0-idx
        X_geo = GT[0] + X_pixel * GT[1] + Y_line * GT[2]
        Y_geo = GT[3] + X_pixel * GT[4] + Y_line * GT[5]
        print(X_geo, Y_geo) # , utm_zone, 'N')
