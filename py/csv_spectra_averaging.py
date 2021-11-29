'''20211128 averaging over a window, where the windowed data are from:
    raster_extract_spectra.py'''   
import os
import sys
import csv
from misc import read_csv
from misc import exist
from misc import err
args = sys.argv

in_f = args[1]
if not exist(in_f):
    err('could not find input file: ' + in_f)

'''read the csv and locate the spectra'''
fields, data = read_csv(in_f)
fields = [x.strip().replace(',', '_') for x in fields]  # forbid comma in header
nf = len(fields)  # number of fields
f_i = {fields[i]:i for i in range(nf)}

spec_fi = []  # col idx for spectral data
for i in range(nf):
    if fields[i][-2:] == 'nm':
        spec_fi += [i]
print('spectra col-ix', spec_fi)
print('number of cols', len(spec_fi))

N = len(data[0]) # number of data points
print("number of data points", N)

sys.exit(1)

x = range(len(spec_fi))
for i in range(N):
    value = data[fi][i] # categorical value
    spectrum = [float(data[j][i]) for j in spec_fi]
    

