'''use GDAL to resample a raster to 20m (ENVI format)
using parallelism (all cores) 20220411'''
import os
import sys
from misc import err, run, args, exists, pd

message = "python3 raster_warp_20m.py [input raster file name]"

if len(args) < 2:
    err(message)
infile = args[1]
ofn = infile + '_20m.bin'

if not exists(infile):
    err("please check input file")

print('+w ' + ofn)

if exists(ofn):
    print("output file already exists: " + ofn)
else:
    run(' '.join(['gdalwarp -of ENVI -tr 20 20',
                  '-multi',
                  '-wo NUM_THREADS=val/ALL_CPUS',
                  infile,
                  ofn]))

run(['python3',
     pd + 'envi_header_cleanup.py',
     infile[:-4] + '.hdr'])

run(['python3',
     pd + 'envi_header_copy_mapinfo.py',
     infile[:-4] + '.hdr',
     ofn[:-4] + '.hdr'])
