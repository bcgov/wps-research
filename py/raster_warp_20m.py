'''use GDAL to resample a raster to 20m (ENVI format)
using parallelism (all cores) 20220411'''
import os
import sys
from misc import err, run, args, exists

message = "python3 raster_warp_20m.py [input raster file name]"

if len(args) < 2:
    err(message)
infile = args[1]
ofn = infile + '_20m.bin'

if not exists(infile):
    err("please check input file")

print('+w ' + ofn)

if exists(ofn):
    err("output file already exists: " + ofn)

run(' '.join(['gdalwarp -of ENVI -tr 20 20',
              '-multi',
              '-wo NUM_THREADS=val/ALL_CPUS',
              infile,
              ofn]))
