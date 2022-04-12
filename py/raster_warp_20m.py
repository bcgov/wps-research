import os
import sys
from misc import err, run, args

message = "python3 raster_warp_20m.py [input raster file name]"

if len(args) < 2:
    err(message)
infile = args[1]
ofn = infile + '_20m.bin'

run(' '.join(['gdalwarp -of ENVI -tr 20 20',
              infile,
              ofn]))
