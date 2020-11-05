# stack raster files
import os
import sys
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0: err("failed to run: " + str(c))

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


cmd = ['python3', # envi_header_cat.py is almost like a reverse-polish notation. Have to put the "first thing" on the back..
       pd + 'envi_header_cat.py',
       rasters[1][:-4] + '.hdr',
       rasters[0][:-4] + '.hdr',
       'raster.hdr']

cmd = ' '.join(cmd)
run(cmd)


for i in range(2, len(rasters)):
    cmd = ['python3', # envi_header_cat.py is almost like a reverse-polish notation. Have to put the "first thing" on the back..
           pd + 'envi_header_cat.py',
           rasters[i][:-4] + '.hdr',
           'raster.hdr',
           'raster.hdr']

cmd = ' '.join(cmd)
print('*', cmd)











