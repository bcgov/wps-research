'''determine pixel location (lat lon) for point within a raster.
Uses map info from the raster, but not the raster data itself'''
import os
import sys
import utm
import json
import struct
import pyproj
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
import matplotlib.pyplot as plt
from misc import args, err, run
if len(args) < 4:
    err("python3 raster pixel location [input image name] [row] [col].. " +
        "..plus additional [row] [col] pairs as needed")
# open image and get geotransform
img = args[1]
if not img: err('pls check input file')
src = gdal.Open(img)
GT = src.GetGeoTransform()
print('GeoTransform', GT)
n_coords = int((len(args) - 2) / 2)  # number of coordinate pairs provided
print('X_pixel,X_line,X_geo,Y_geo,lat,lon')

from pyproj import Proj
d = os.popen('gdalsrsinfo -o proj4 ' + img).read().strip()
print(d)
# myProj = Proj(os.popen('gdalsrsinfo -o proj4 ' + img).read().strip())

w = d.split()
w = [x.strip('+').split('=') for x in w]
srs_info = {x[0]: x[1] if len(x) > 1 else '' for x in w}
print(srs_info)

pts = []
for i in range(n_coords):
    i2 = 2 * i
    X_pixel = float(args[i2 + 3]) # col 0-idx
    Y_line = float(args[i2 + 2 ]) # row 0-idx
    X_geo = GT[0] + X_pixel * GT[1] + Y_line * GT[2]
    Y_geo = GT[3] + X_pixel * GT[4] + Y_line * GT[5]
    latlon = utm.to_latlon(X_geo, Y_geo, int(srs_info['zone']), 'N')
    print(X_pixel, Y_line, X_geo, Y_geo, latlon[0], latlon[1])
    pts.append(latlon)
    # latlon 
    # lon.append(myProj(X_geo, Y_geo, inverse=True))
    # lon, lat = myProj(df['Meters East'].values, df['Meters South'].values, inverse=True)

import simplekml
pts2 = [(p[1], p[0]) for p in pts]
kml = simplekml.Kml()
pol = kml.newpolygon(name="fire")
pol.outerboundaryis.coords = pts2

for p in pts:
    print("point", p)
    pp = kml.newpoint(name='Point: {0}{0}'.format(p[1], p[0])) # lon,lat))
    pp.coords = [(p[1],p[0])]
    pp.style=simplekml.Style()
kml.save('poly.kml')

'''Note that the pixel/line coordinates in the above are from (0.0,0.0) at the top left corner of the top left pixel to (width_in_pixels,height_in_pixels) at the bottom right corner of the bottom right pixel. The pixel/line location of the center of the top left pixel would therefore be (0.5,0.5).'''


'''
if True:
    if True:
        if True:
            cmd = ["gdallocationinfo",
                   img, # input image
                   str(pix_j), # default: pixl number (0-indexed) aka row
                   str(lin_j)] # default: line number (0-indexed) aka col
            cmd = ' '.join(cmd)
            print("  \t" + cmd)
            lines = [x.strip() for x in os.popen(cmd).readlines()]

            for line in lines:
                print('\t' + line)
            if len(lines) != 2 * (1 + nb):
                err("unexpected result line count")

            w = lines[1].split()
            if w[0] != "Location:":
                err("unexpected field")
            pix_k, lin_k = w[1].strip('(').strip(')').split(',')
            if pix_k[-1] != 'P' or lin_k[-1] != 'L':
                err('unexpected data')
'''
