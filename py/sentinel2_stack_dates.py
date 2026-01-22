'''20240723 stack sentinel2 .bin files in temporal order

one tile only supported'''
from misc import run, hdr_fn, band_names
import os

lines = [x.strip() for x in os.popen('ls -1 S*.bin').readlines()]

# process in order
lines = [[line.split('_')[2], line] for line in lines]
lines.sort()
lines = [line[1] for line in lines]

for line in lines:
    ds = line.split('_')[2].split('T')[0]
    print(ds, line)

run(' '.join(['raster_stack.py'] + lines + ['stack.bin']))
