'''20220502 determine pixel loc'n (lat lon) for each point in raster
NB doesn't generalize to southern hemisphere. Thanks:
https://stackoverflow.com/questions/50191648/
gis-geotiff-gdal-python-how-to-get-coordinates-from-pixel'''
import os
import utm
import pyproj
import shutil
import numpy as np
from osgeo import osr
from osgeo import gdal
from misc import args, err, run, read_hdr, hdr_fn, write_binary, pd, sep
if len(args) < 2:
    err("python3 raster_pixels_location.py [input image name]")

img = args[1]  # open image and get geotransform
if not img:
    err('pls check input file')
src = gdal.Open(img)
gx = src.GetGeoTransform()
xoffset, px_w, rot1, yoffset, px_h, rot2 = src.GetGeoTransform()
ncol, nrow, nband = [int(x) for x in read_hdr(hdr_fn(img))]
d = os.popen('gdalsrsinfo -o proj4 ' + img).read().strip()
w = [x.strip('+').split('=') for x in d.split()]
srs_info = {x[0]: x[1] if len(x) > 1 else '' for x in w}
npx = nrow * ncol

# get CRS from dataset
crs = osr.SpatialReference()
print(src.GetProjectionRef())
crs.ImportFromWkt(src.GetProjectionRef())

# create lat/long crs: WGS84 datum
crsGeo = osr.SpatialReference()
crsGeo.ImportFromEPSG(4326)  # 4326 is EPSG-id of lat/lon crs
t = osr.CoordinateTransformation(crs, crsGeo)

dat = np.zeros(npx * 2)  # output raster for lon, lat
for Y_line in range(nrow):  # row 0-idx
    y_i = Y_line * ncol
    y_i2 = y_i + npx
    yl = Y_line  # + .5
    for X_pixel in range(ncol):  # col 0-idx
        xp = X_pixel  # + .5
        X_geo = gx[0] + xp * gx[1] + yl * gx[2]
        Y_geo = gx[3] + xp * gx[4] + yl * gx[5]
        X_geo += px_w / 2.  # shift to pixel centre
        Y_geo += px_h / 2.
        (lat, lon, z) = t.TransformPoint(X_geo, Y_geo)
        dat[y_i + X_pixel] = lon
        dat[y_i2 + X_pixel] = lat
fn = img + '_lonlat.bin'
hn = img + '_lonlat.hdr'
write_binary(dat, fn)  # write out raster
hi = hdr_fn(img)

print('+w', hn)  # copy existing hdr..
shutil.copyfile(hi, hn)

# .. now update dimensions & band names
run(['python3',  # python interp
     pd + 'envi_header_modify.py',  # python script path
     hn,  # header file name
     str(nrow),  # number of rows
     str(ncol),  # number of cols
     '2',   # two bands
     'lon',   # band name 1
     'lat'])  # band name 2

# update map info for raster
run(['python3',  # python interp
     pd + 'envi_header_copy_mapinfo.py',  # python script path
     hi,  # input header file to copy from
     hn])  # copy mapinfo to this header file
