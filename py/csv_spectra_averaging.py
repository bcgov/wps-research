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
nf = len(fields)  # number of fields
f_i = {fields[i]:i for i in range(nf)}
if args[2] not in fields:
    print('fields available:', fields)
    err('field: ' + fi + 'not found in:' + in_f)
fi = f_i[args[2]]  # col index of selected field for legending
field_label = args[2].strip().replace(' ', '-')  # spaces always bad?

spec_fi = []
for i in range(nf):
    if fields[i][-2:] == 'nm':
        spec_fi += [i]
print('spectra col-ix', spec_fi)
print('number of cols', len(spec_fi))

N = len(data[0]) # number of data points
print("number of data points", N)

x = range(len(spec_fi))
for i in range(N):
    value = data[fi][i] # categorical value
    spectrum = [float(data[j][i]) for j in spec_fi]
    

