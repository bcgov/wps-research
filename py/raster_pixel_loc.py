'''determine pixel location (lat lon) for point within a raster.
Uses map info from the raster, but not the raster data itself
20220328 added support for multipolygon case'''
import os
import sys
import utm
import json
import struct
import pyproj
import simplekml
from osgeo import ogr
from osgeo import osr
from osgeo import gdal
import matplotlib.pyplot as plt
from misc import args, err, run
if len(args) < 3:
    err("python3 raster_pixel_location.py [input image name] [alpha points file]")

# open image and get geotransform
img = args[1]
if not img: err('pls check input file')
src = gdal.Open(img)
GT = src.GetGeoTransform()
print('X_pixel,Y_line,X_geo,Y_geo,Lat,Lon')


from pyproj import Proj
cmd = 'gdalsrsinfo -o proj4 ' + img
print(cmd)
d = os.popen(cmd).read().strip()
print(d)

w = d.split()
w = [x.strip('+').split('=') for x in w]
srs_info = {x[0]: x[1] if len(x) > 1 else '' for x in w}
print(srs_info)

if 'zone' not in srs_info:
    print("was Assumed to be in UTM coordinates")

pts_in = []
data = open(args[2]).read().strip()
if data.startswith('POLYGON'):
    data = data.split('((')[1].strip(')').strip(')').split(',')
    X = [x.strip().split() for x in data]
    pts_in.append([(int(x[0]), int(x[1])) for x in X])
elif data.startswith('MULTIPOLYGON'):
    data = data.replace(')), ((', ')) ((')
    data = data.split('(((')[1].strip(')').strip(')').strip(')').split(')) ((')
    for x in data:
        X = x.strip().split()
        pts_in.append([(int(X[i].strip(',')),
                        int(X[i+1].strip(',')))
                        for i in range(0, len(X), 2)])
else:
    err('unrecognized')

# let's output the largest of multipolygons first
Y = [[len(i), i] for i in pts_in]
Y.sort(reverse=True)
pts_in = [i[1] for i in Y]

# prepare to write KML
print("srs_info", srs_info)
kml = simplekml.Kml()

# def^ne source & target coordinate systems
# source_crs = pyproj.CRS(srs_info['proj'])  # Use the CRS information from your data
# Open the raster dataset to get its CRS
raster_dataset = gdal.Open(args[1]) # 'your_raster_file.tif')  # Replace with the path to your raster file
if raster_dataset is None:
    raise Exception("Failed to open the raster dataset.")
print(raster_dataset)
print(gdal.Info(raster_dataset, format='json'))
wkt = raster_dataset.GetProjection()
src = osr.SpatialReference()
src.ImportFromWkt(wkt)

x = gdal.Info(raster_dataset, format='json')
print("X")
for y in x:
    print(y, x[y])
print("X")

# source image
src = gdal.Open(args[1]) # , gdalconst.GA_ReadOnly)
src_proj, src_geotrans = src.GetProjection(), src.GetGeoTransform()
src_sr = osr.SpatialReference(wkt=src.GetProjection())  # spatial reference
print("src_proj", src_proj)
print("src_geot", src_geotrans)
print("src_sr ", src_sr)
src = None




source_crs = pyproj.CRS(gdal.Info(raster_dataset, format='json')['coordinateSystem']['wkt'])
target_crs = pyproj.CRS('EPSG:4326')  # WGS 84, which is commonly used for lat/lon
transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

for k in range(len(pts_in)):
    P = pts_in[k]
    n_coords, pts = len(P), []
    for i in range(n_coords):
        X_pixel = P[i][1] # float(args[i2 + 3]) # col 0-idx
        Y_line = P[i][0] # float(args[i2 + 2 ]) # row 0-idx
        X_geo = GT[0] + X_pixel * GT[1] + Y_line * GT[2]
        Y_geo = GT[3] + X_pixel * GT[4] + Y_line * GT[5]
        

        # latlon = utm.to_latlon(X_geo, Y_geo, int(srs_info['zone']), 'N')
        lon, lat = transformer.transform(X_geo, Y_geo)
        latlon = [lat, lon] # [lon,lat]
         
        '''
latlon (58.54544607900768, -120.24008482430301)
lon -120.24008482768689 lat 58.54544604697952
3031 365 660620.0 6492740.0 58.54544607900768 -120.24008482430301'''

        # print("latlon", latlon)
        # print("lon", lon, "lat", lat)   
        # print(X_pixel, Y_line, X_geo, Y_geo, latlon[0], latlon[1])
        pts.append(latlon)
    pts2 = [(p[1], p[0]) for p in pts]
    pol = kml.newpolygon(name=("fire" + ("" if k == 0 else ('_' + str(k)))))
    pol.outerboundaryis.coords = pts2
    pol.polystyle.color = '990000ff'  # Red

pfn = 'poly_' + args[1][:-4] + '.kml'
print('+w', pfn)
kml.save(pfn)

'''# alternately, for adding points to the kmz
for p in pts:
    print("point", p)
    pp = kml.newpoint(name='Point: {0}{0}'.format(p[1], p[0])) # lon,lat))
    pp.coords = [(p[1],p[0])]
    pp.style = simplekml.Style()
'''
