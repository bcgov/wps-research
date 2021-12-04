import os
import sys
from misc import pd
from misc import err
from misc import run
from misc import args
from misc import hdr_fn
from misc import read_hdr

if len(args) < 2:
    err('python3 raster_video.py [input ENVI format file]')

samples, lines, bands = [int(x) for x in read_hdr(hdr_fn(args[1]))]

if bands % 12 != 0:
    err('expected bands to be multiple of 12')

frames, bi = int(bands / 12), [4, 3, 2]
for i in range(frames):
    cmd = ' '.join([pd + 'raster_plot.py', args[1]] + [str(j) for j in bi] + ['1'])
    bi = [j + 12 for j in bi]
    run(cmd)
