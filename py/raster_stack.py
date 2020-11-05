# stack raster files
import os
import sys
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

if len(args) < 4:
    err("python3 raster_stack.py [input raster 1] [input raster 2] .." +
        " [input raster n] [output raster name]")

rasters = args[1:-1]
outf = args[-1]

cmd = ['cat'] + rasters + ['>', outf]
cmd = ' '.join(cmd)
print(cmd)
if not exists(outf):
    a = os.system(cmd)







