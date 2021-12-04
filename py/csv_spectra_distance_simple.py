'''calculate distance from spectra in file resulting from raster_extract_spectra_simple.py,
to a raster

python3 csv_spectra_distance_simple.py [input csv file] [raster file]'''
import os
import sys
import csv
import numpy
import math
from misc import read_csv
from misc import exist
from misc import err
from misc import hdr_fn
from misc import read_binary
from misc import write_hdr
from misc import write_binary
args = sys.argv

if len(args) < 3:
    err('python3 csv_spectra_distance_simple.py [csv spectra file (one spectrum)] [raster file]')

csv_fn, dfn = args[1], args[2]

'''read the csv and locate the spectra'''
fields, data = read_csv(csv_fn)
nf = len(fields)  # number of fields
f_i = {fields[i]:i for i in range(nf)}

spec_fi = []
for i in range(nf):
    if fields[i][-2:] == 'nm':
        spec_fi += [i]
print('spectra col-ix', spec_fi)
print('number of cols', len(spec_fi))

'''before we plot, code the categorical possibilities from 0 to however many there are'''
N = len(data[0]) # number of data points
if N != 1:
    err("expected one spectral data point only")

if True:
    i = 0
    spec = [float(data[j][i]) for j in spec_fi]
    print("spectrum", spec)

    ncol, nrow, nband, data = read_binary(dfn)
    np = nrow * ncol
    if len(spec) != nband:
        err("unexpected number of spectral bands")

    out = numpy.zeros(nrow * ncol)

    for i in range(nrow):
        print(i)
        for j in range(ncol):
            d = 0.
            ix = i * ncol + j
            for k in range(nband):
                x = spec[k] - data[ix + (k * np)]
                d += math.sqrt(x * x)
            out[ix] = d  
    numbers = []
    w = csv_fn.split(os.path.sep)[-1][:-4].split('_')
    for i in w:
        try:
            numbers.append(int(i))
        except Exception:
            pass
    ofn = '_'.join([dfn] + [str(x) for x in numbers]) + '_distance.bin'
    ohn = ofn[:-4] + '.hdr'
    write_binary(out, dfn + '_distance.bin')
    write_hdr(dfn + '_distance.hdr', ncol, nrow, 1)
