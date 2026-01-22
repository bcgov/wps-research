# stack raster files # should really not do a cumulative / sequential header modification
import os
import sys
from misc import args, sep, exists, pd, run, err, hdr_fn, read_hdr, band_names, write_hdr

if len(args) < 4:
    err("python3 raster_stack.py [input raster 1] [input raster 2] .." +
        " [input raster n] [output raster name]")

rasters = args[1:-1]  # raster files to cat together

if len(rasters) < 2:
    error("expected 2 or more rasters")

outf = args[-1]  # output filename
ofhn = args[-1][:-4] + '.hdr'  # output header file name
cmd = ['cat'] + rasters + ['>', outf]  # cat command
cmd = ' '.join(cmd)
print(cmd)
print('')

if not exists(outf):
    a = os.system(cmd)
else:
    err("output file already exists")

ncol, nrow, nb = 0,0,0
ncols, nrows, nbands = [], [], 0
bn = []
for r in rasters:
    ncol, nrow, nb = read_hdr(hdr_fn(r))
    ncols += [ncol]
    nrols += [nrow]
    nbands += int(nb)
    bn += band_names(hdr_fn(r))

# def write_hdr(hfn, samples, lines, bands, band_names = None):
write_hdr(ohfn, ncols[0], nrows[0], nbands, bn)
'''
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
'''
