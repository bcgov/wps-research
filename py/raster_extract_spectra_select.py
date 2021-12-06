'''
THIS SCRIPT NOT OPERATIONAL YET, MIGHT NEED TO DELETE THIS ONE..

20211205 just average the spectra on a CSV.. for instances where the selected field [field to select from]
matches the value provided [value to select]

this script not tested yet'''
DEBUG = False
import os
import sys
import csv
import math
import numpy
from misc import * 

if len(args) < 3:
    err('python3 csv_spectra_aggregate.py [csv spectra file (one spectrum)] ' +
        ' [field to select from] [value to select]' +     
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

'''now we have new spectrums, output them'''
lines = []
lines.append(','.join([fields[i] for i in spec_fi]))
lines.append(','.join([str(x) for x in spec_avg]))
ofn = in_f + '_' + select_field + '_eq_' + select_value + '_aggregate.csv'
print('+w', ofn)
open(ofn, 'wb').write(('\n'.join(lines)).encode())
