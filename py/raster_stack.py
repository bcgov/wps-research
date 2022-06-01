# stack raster files
import os
import sys
from misc import args, sep, exists, pd, run, err

if len(args) < 4:
    err("python3 raster_stack.py [input raster 1] [input raster 2] .." +
        " [input raster n] [output raster name]")

rasters = args[1:-1]  # raster files to cat together
outf = args[-1]  # output filename
ofhn = args[-1][:-4] + '.hdr'  # output header file name
cmd = ['cat'] + rasters + ['>', outf]  # cat command
cmd = ' '.join(cmd)
print(cmd)
print('')
if not exists(outf):
    a = os.system(cmd)

for r in rasters:
    print(" ", r)

cmd = ['python3', # envi_header_cat.py is almost like a reverse-polish notation. Have to put the "first thing" on the back..
       pd + 'envi_header_cat.py',
       rasters[1][:-4] + '.hdr',
       rasters[0][:-4] + '.hdr',
       ofhn]  # 'raster.hdr']
cmd = ' '.join(cmd)
run(cmd)

print('')

# envi_header_cat.py almost like a RPN. Put "first thing" on the back
if len(rasters) > 2:
    for i in range(2, len(rasters)):
        cmd = ['python3',
               pd + 'envi_header_cat.py',
               rasters[i][:-4] + '.hdr',
               ofhn,  # 'raster.hdr',
               ofhn]  # 'raster.hdr']
        cmd = ' '.join(cmd)
        run(cmd)

    cmd = ['python3', pd + 'envi_header_cleanup.py',
            ofhn]
    run(cmd)
