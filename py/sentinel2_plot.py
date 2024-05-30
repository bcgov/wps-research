'''20240527 plot sentinel2 products (filenames beginning with frame ID) 
'''
from misc import err, run, parfor
import os

lines = [x.strip() for x in os.popen('ls -1 S*.bin').readlines()]

# process in order
lines = [[line.split('_')[2], line] for line in lines]
lines.sort()
lines = [line[1] for line in lines]


cmds = ["raster_plot.py " + line + " 1 2 3 1 " for line in lines]

def r(x):
    return os.system(x)

parfor(r, cmds, 8) 

'''Now: prefix the S2.png files by date:
'''
lines = os.popen("ls -1 S2*.png").readlines()
lines = [x.strip() for x in lines]

for line in lines:
    T = line.split('_')[2].split('T')[0]
    run('mv -v ' + line + ' ' + T + '_' + line)
