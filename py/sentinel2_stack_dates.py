'''20240723 stack sentinel2 .bin files in temporal order

one tile only supported'''
from misc import run, hdr_fn, band_names, read_hdr
import os

lines = [x.strip() for x in os.popen('ls -1 S*.bin').readlines()]

# process in order
lines = [[line.split('_')[2], line] for line in lines]
lines.sort()
lines = [line[1] for line in lines]

ncol, nrow, nb = 0, 0, 0
bn_new = []
for line in lines:
    ds = line.split('_')[2].split('T')[0]
    print(ds, line)

    ncol, nrow, nb = read_hdr(hdr_fn(line))
    bn = band_names(hdr_fn(line))

    bn_new += [ds + ': ' + i for i in bn]


print(bn_new)
run(' '.join(['raster_stack.py'] + lines + ['stack.bin']))
#  envi_header_modify.py [.hdr file to modify] [nrow] [ncol] [nband] [band 1 name]... [band n name]')
ncol, nrow, nb = read_hdr('stack.hdr')
run('envi_header_modify.py ' + (' '.join([str(x) for x in [ncol, nrow, nb] + bn_new])))
