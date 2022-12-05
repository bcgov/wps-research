'''20220831 running https://github.com/bopen/elevation

Deps:
    pip3 install elevation
    pip3 install rasterio    # for raster reference/footprint file (here we assume raster)
    pip3 install fiona       # for shapefile reference/footprint file

Steps:
    1. raster_pixels_loc.py # determine lat/lon for each pixel in image
    2. raster_quickstats    # determine min,max for both lat and lon
    3. # {increase, decrease} the {lat/lon}  {max, min} respectively, to make sure we get results for all the data (broaden the area slightly so we don't miss any at the edge)
'''
import os
import elevation
from misc import run, err, args, exist

fn = args[1]  # input raster footprint name
out_fn = fn + '_dem.tif'  # output dem file name!

lon_lat = fn + '_lonlat.bin'

if not exist(lon_lat):
    run('raster_pixels_loc.py ' + fn)

lines = [x.strip() for x in os.popen('raster_quickstats ' + lon_lat).readlines()][1:]
lines = [line.split(',') for line in lines]

lat_min = float(lines[2][1]) - .01
lat_max = float(lines[2][2]) + .01
lon_min = float(lines[1][1]) - .01
lon_max = float(lines[1][2]) + .01
print("lat", lat_min, lat_max)
print("lon", lon_min, lon_max)

s = ' '.join(['eio clip -o ',
              out_fn,
              '--bounds',
              str(lon_min),
              str(lat_min),
              str(lon_max),
              str(lat_max),
              '--reference',
              fn])
run(s)

# dem = elevation.clip(bounds=(lon_min, lat_min, lon_max, lat_max), output=out_fn)
# print(dem)
# a = os.system('find')
# elevation.clean()

run('fh ' + fn + '_dem.hdr')
run('fh dem.hdr')
run('gdal_translate -of ENVI -ot Float32 ' + fn + '_dem.tif ' + fn + '_dem.bin')
run('po ' + fn + '_dem.bin ' + fn + ' dem.bin')

run('raster_threshold dem.bin GEQ 0.')
run('raster_mult dem.bin dem.bin_thres.bin DEM.bin')
run('raster_zero_to_nan DEM.bin')
run('raster_stack.py DEM.bin ' + fn + ' stack_dem.bin')

# eio clip -o dem.tif --bounds -1.267663421630859E+02 5.595667114257812E+01 -1.246634750366211E+02 5.620212707519531E+01
if False:
    run('eio --product SRTM3 clip -o ' + out_fn + ' --reference ' + fn)
