'''20240527 plot sentinel2 products (filenames beginning with frame ID) 
'''
from misc import err, run
import os

lines = [x.strip() for x in os.popen('ls -1 S*.bin').readlines()]

# process in order
lines = [[line.split('_')[2], line] for line in lines]
lines.sort()
lines = [line[1] for line in lines]

for line in lines:
    cmd = "raster_plot.py " + line + " 1 2 3 1 "
    run(cmd)


