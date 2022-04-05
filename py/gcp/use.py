'''deprecated: a script to assemble rasters into a stack'''
import os
import sys
from ../misc import pd, run # python directory
dates = [x.strip() for x in open("use.txt_sort.csv").read().strip().split('\n')]
lines = os.popen('find  -name "*L2A_EPSG_*_10m.bin"').readlines()

binfile = {}
for line in lines:
    ds = line[13:13+8]
    binfile[ds] = line

by_year = {}
for d in dates:
    year = d[:4]
    if year not in by_year:
        by_year[year] = []
    by_year[year].append(d)


for year in by_year:
    cmd = 'python3 ' + pd + 'raster_stack.py'
    for d in by_year[year]:
        b = binfile[d].strip() # print("\t", d, b)
        cmd += ' ' + b
    cmd += ' ' + str(year) + ".bin"
    run(cmd)
