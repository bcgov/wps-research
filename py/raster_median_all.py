import os
import sys

lines = os.popen('ls -1 *.bin').readlines()
lines = [x.strip() for x in lines]

cmd = 'raster_median ' + ' '.join(lines) + ' median.bin'
print(cmd)

