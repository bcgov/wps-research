'''20231108: run raster_nan_to_zero on all .bin files in folder
'''
import os
import sys

lines = os.popen("ls -1 *.bin").readlines()

for line in lines:
    line = line.strip()
    cmd = "raster_nan_to_zero " + line
    a = os.system(cmd)
