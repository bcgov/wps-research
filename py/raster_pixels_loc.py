'''20220502 determine pixel location (lat lon) for each point in a raster
note this doesn't generalize to southern hemisphere'''
import os
import utm
import pyproj
import shutil
import numpy as np
from osgeo import gdal
from misc import args, err, run, read_hdr, hdr_fn, write_binary, pd, sep

if len(args) < 2:
    err("python3 raster_pixels_location.py [input image name]")

img = args[1]  # open image and get geotransform
if not img:
    err('pls check input file')
src = gdal.Open(img)
gx = src.GetGeoTransform()
ncol, nrow, nband = [int(x) for x in read_hdr(hdr_fn(img))]
d = os.popen('gdalsrsinfo -o proj4 ' + img).read().strip()
w = [x.strip('+').split('=') for x in d.split()]
srs_info = {x[0]: x[1] if len(x) > 1 else '' for x in w}
npx = nrow * ncol

dat = np.zeros(npx * 2)
for Y_line in range(nrow): # row 0-idx
    y_i = Y_line * ncol
    y_i2 = y_i + npx
    for X_pixel in range(ncol): # col 0-idx
        X_geo = gx[0] + X_pixel * gx[1] + Y_line * gx[2]
        Y_geo = gx[3] + X_pixel * gx[4] + Y_line * gx[5]
        dat[y_i + X_pixel] = X_geo
        dat[y_i2 + X_pixel] = Y_geo # print(X_geo, Y_geo)

fn = args[1] + '_lonlat.bin'
# print('+w', fn)
write_binary(dat, fn)

hn = args[1] + '_lonlat.hdr'
hi = hdr_fn(args[1])
print('+w', hn)
shutil.copyfile(hi, hn)

# update image shape & band names
run(['python3', # python interp
     pd + 'envi_header_modify.py', # python script path
     hn, # header file name
     str(nrow), # number of rows
     str(ncol), # number of cols
     '2',  # two bands
     'lon',  # band name 1
     'lat']) # band name 2

# copy mapinfo
run(['python3', pd + 'envi_header_copy_mapinfo.py', hi, hn])
