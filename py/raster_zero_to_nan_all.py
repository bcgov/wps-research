'''20231108: run raster_zero_to_nan on all .bin files in folder
'''
import os
import sys

lines = os.popen("ls -1 *.bin").readlines()

for line in lines:
    line = line.strip()
    cmd = "raster_zero_to_nan " + line
    a = os.system(cmd)
