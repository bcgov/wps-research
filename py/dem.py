'''20220831 running https://github.com/bopen/elevation

python3 -m pip install elevation

20221205 update'''
import os
import elevation
from misc import run, err, args, exist

fn = args[1]  # input raster footprint name
out_fn = fn + '_dem.tif'  # output dem file name!

lon_lat = fn + '_lonlat.bin'

if not exist(lon_lat):
    run('raster_pixels_loc.py ' + fn)

lines = [x.strip() for x in
         os.popen('raster_quickstats ' + lon_lat).readlines()][1:]
lines = [line.split(',') for line in lines]

lat_min = float(lines[2][1]) - .01  # pad a bit to ensure no empty border pix
lat_max = float(lines[2][2]) + .01
lon_min = float(lines[1][1]) - .01
lon_max = float(lines[1][2]) + .01
print("lat", lat_min, lat_max)
print("lon", lon_min, lon_max)

run(' '.join(['eio clip -o ',
              out_fn,
              '--bounds',
              str(lon_min),
              str(lat_min),
              str(lon_max),
              str(lat_max),
              '--reference',
              fn]))

run('fh ' + fn + '_dem.hdr')
run('fh dem.hdr')
run(' '.join(['gdal_translate -of ENVI -ot Float32',
              fn + '_dem.tif',
              fn + '_dem.bin']))
run('po ' + fn + '_dem.bin ' + fn + ' dem.bin')  # project onto

run('raster_threshold dem.bin GEQ 0.')
run('raster_mult dem.bin dem.bin_thres.bin DEM.bin')
run('raster_zero_to_nan DEM.bin')
run('raster_stack.py DEM.bin ' + fn + ' stack_dem.bin')
print("result in DEM.bin")