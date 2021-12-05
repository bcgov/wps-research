'''test this on smaller data

python3 ~/GitHub/bcws-psu-research/py/csv_spectra_distance_simple.py ./sub.bin_first_spectrum.csv sub.binalculate distance from spectra in file resulting from raster_extract_spectra_simple.py,
to a raster

python3 csv_spectra_distance_simple.py [input csv file] [raster file]'''
import os
import sys
import csv
import numpy
import math
from misc import parfor
from misc import read_csv
from misc import exist
from misc import err
from misc import hdr_fn
from misc import read_binary
from misc import write_hdr
from misc import write_binary
args = sys.argv
from multiprocessing import Lock
lock = Lock()
n_processed = 0

if len(args) < 3:
    err('python3 csv_spectra_distance_simple.py [csv spectra file (one spectrum)] ' +
        ' [field to select from] [field value to select]' +     
        ' [raster file]')

csv_fn, dfn = args[1], args[4]
select_field = args[2]
select_value = args[3]

'''read the csv and locate the spectra'''
fields, csv_data = read_csv(csv_fn)
nf = len(fields)  # number of fields
f_i = {fields[i]:i for i in range(nf)}

spec_fi = []
for i in range(nf):
    if fields[i][-2:] == 'nm':
        spec_fi += [i]
print('spectra col-ix', spec_fi)
print('number of cols', len(spec_fi))

select_i = f_i[select_field] # index of col indicated to match on..

''' average the spectra where field select_field matches the value select_value'''
N = len(csv_data[0]) # number of data points
n_select, spec_avg = 0., [0. for i in range(len(spec_fi))] # averaged spectra goes here

for i in range(N):
    if csv_data[select_i][i] == select_value:
        n_select += 1
        spec =  [float(csv_data[j][i]) for j in spec_fi]  # extract a spectrum
        print(spec)
        spec_avg = [spec_avg[j] + spec[j] for j in range(len(spec_fi))] # add it onto the to-be average

# divide by N
spec_avg = [spec_avg[i] / n_select for i in range(len(spec_avg))]
print("spec_avg", spec_avg)

# check we selected something
if n_select < 1:
    err("no spectra selected")

# image stuff
ncol, nrow, nband, data = read_binary(dfn)
if len(spec_fi) != nband:
    err("unexpected number of spectral bands")
np = nrow * ncol
out = numpy.zeros(np)

def dist_row(i):
    global lock, out, spec, nrow, ncol, nband, data, n_processed   #for i in range(nrow):
    
    with lock:
        n_processed += 1
    print('%', 100. * n_processed / nrow, i)

    result = []
    ix = i * ncol
    knp = [k * np for k in range(nband)]
    for j in range(ncol):
        d = 0.
        ij = ix + j
        for k in range(nband):
            x = spec[k] - data[ij + knp[k]]
            d += x * x # math.sqrt(x * x)
        result.append(d)
    return result

x = parfor(dist_row, range(nrow))
print("assembling")
for i in range(nrow):
    out[i*ncol: (i+1)*ncol] = x[i]

numbers = []
w = csv_fn.split(os.path.sep)[-1][:-4].split('_')
for i in w:
    try:
        numbers.append(int(i))
    except Exception:
        pass
ofn = ('_'.join([dfn] +
       [str(x) for x in numbers]) + '_' +
       select_field.replace(' ','_') + '_eq_' +
       select_value.replace(' ', '_') + '_distance.bin')
ohn = ofn[:-4] + '.hdr'
write_binary(out, ofn)
write_hdr(ohn, ncol, nrow, 1)
