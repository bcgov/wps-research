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
if len(args) < 3:
    err("python3 raster_pixel_location.py [input image name] [alpha points file]")
# need to handle multipolygon case

# open image and get geotransform
img = args[1]
if not img: err('pls check input file')
src = gdal.Open(img)
GT = src.GetGeoTransform()
n_coords = int((len(args) - 2) / 2)  # number of coordinate pairs provided
print('X_pixel,Y_line,X_geo,Y_geo,Lat,Lon')
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
pol.polystyle.color = '990000ff'  # Red


'''
# alternately, for adding points to the kmz
for p in pts:
    print("point", p)
    pp = kml.newpoint(name='Point: {0}{0}'.format(p[1], p[0])) # lon,lat))
    pp.coords = [(p[1],p[0])]
    pp.style=simplekml.Style()
'''
kml.save('poly_' +  ('.'.join(args[1].split('.')[:-1])) + '.kml')
