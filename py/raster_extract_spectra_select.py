'''20211205 
Extract and average spectra on a CSV.. for instances where the selected field [field to select from]
matches the value provided [value to select]
* the entry point for spectra extraction..'''
DEBUG = False
import os
import sys
import csv
import math
import numpy
from misc import * 

if len(args) < 5:
    err('python3 raster_extract_spectra_select.py ' +
        '[csv spectra file (no spectra but x, y (wgs-84) and select-field)] ' +
        '[field to select from (select-field)] ' +
        '[value to select (select-value)]' +     
        '[raster file to select spectra from]')

csv_fn, select_field, select_value, img_fn = args[1:5]

'''read the csv'''
fields, csv_data = read_csv(csv_fn)
nf = len(fields)  # number of fields
f_i = {fields[i]:i for i in range(nf)}

for i in ['x', 'y', select_field]:  # assert it contains lat/lon..
    if i not in fields:
        err('expected field in ' + csv_fn + ': ' + i)
[x_i, y_i, select_i] = [f_i[i]
                        for i in ['x', 'y', select_field]]  # which col-idx to use?


''' average the spectra where field select_field matches the value select_value'''
N = len(csv_data[0]) # number of data points in whole csv

hfn = hdr_fn(img_fn)  # raster info: samples, lines, bands, and band names
[nc, nr, nb], bn = [int(i) for i in read_hdr(hfn)], band_names(hfn)

for i in range(N):  # look at all rows in the csv
    if csv_data[select_i][i] == select_value:
        x = csv_data[x_i][i] 
        y = csv_data[y_i][i]
        row, col, spec = xy_to_pix_lin(img_fn, x, y, nb)  # extract pix lin loc, and spectra from file
        print("row", row, "col", col)
# output by catting onto the original csv! include the parameters in the addition!
# also output the averaged spectra (with stdv??) on the segment (segment of the csv)

'''
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

lines = []
lines.append(','.join([fields[i] for i in spec_fi]))
lines.append(','.join([str(x) for x in spec_avg]))
ofn = in_f + '_' + select_field + '_eq_' + select_value + '_aggregate.csv'
print('+w', ofn)
open(ofn, 'wb').write(('\n'.join(lines)).encode())'''
