'''20211205  wow 
Extract and average spectra on a CSV.. for instances where the selected field [field to select from]
matches the value provided [value to select]
* the entry point for spectra extraction..
e.g.
    python3 ~/GitHub/wps-research/py/raster_extract_spectra_select.py survey_label.csv cluster_label 6 sub.bin '''
import os
import sys
import csv
import math
import numpy
from misc import * 

if len(args) < 5:
    err('python3 raster_extract_spectra_select.py ' +
        '[csv file (no spectra but x, y (wgs-84) and select-field)] ' +
        '[field to select from (select-field)] ' +
        '[value to select (select-value)]' +     
        '[raster file to select spectra from]')

csv_fn, select_field, select_value, img_fn = args[1:5]

'''read the csv'''
fields, csv_data = read_csv(csv_fn)
nf = len(fields)  # number of fields

# sanitize for evil commas !
fields = [fields[i].strip().replace(',', ';') for i in range(len(fields))]
for i in range(len(csv_data)):
    for j in range(len(csv_data[i])):
        csv_data[i][j] = csv_data[i][j].strip().replace(',', ';')

f_i = {fields[i]:i for i in range(nf)} # create a lookup... use it to:
for i in ['x', 'y', select_field]:  # assert the header contains lat/lon..
    if i not in fields:
        err('expected field in ' + csv_fn + ': ' + i)
[x_i, y_i, select_i] = [f_i[i]
                        for i in ['x', 'y', select_field]]  # which col-idx to use?


''' average the spectra where field select_field matches the value select_value'''
N = len(csv_data[0]) # number of data points in whole csv

hfn = hdr_fn(img_fn)  # raster info: samples, lines, bands, and band names
[nc, nr, nb], bn = [int(i) for i in read_hdr(hfn)], band_names(hfn)

new = []
mean = None
count = 0.
for i in range(N):  # look at all rows in the csv
    if csv_data[select_i][i] == select_value:
        x = csv_data[x_i][i] 
        y = csv_data[y_i][i]
        row, col, spec = xy_to_pix_lin(img_fn, x, y, nb)  # extract pix lin loc, and spectra from file
        new.append([i, row, col, spec])
        print("row", row, "col", col)
        mean = spec if mean is None else [mean[j] + spec[j] for j in range(len(spec))]
        count += 1.
mean = [mean[j] / count for j in range(len(mean))]  # divide by N
print("mean", mean)
stdv = [numpy.std([new[j][3][k] for j in range(len(new))]) for k in range(len(mean))] # k in rnge(len(new))]) for j in range(len(mean))]
print("stdv", stdv)

csv_ofn = '_'.join([csv_fn, 'spectra', select_field, select_value + '.csv'])
print('+w', csv_ofn)

lines = [fields + ['row', 'col', 'csv_file', 'select_field', 'select_value', 'image'] + bn] # first line: header
# print(lines)
for i, row, col, spec in new:
    orig = [csv_data[j][i] for j in range(nf)]
    rowcol =  [str(row), str(col), csv_fn, select_field, select_value, img_fn] 
    spec = [str(x) for x in spec]
    lines += [orig + rowcol + spec]
lines = [','.join(line) for line in lines]
open(csv_ofn, 'wb').write(('\n'.join(lines)).encode())

csv_afn = '_'.join([csv_fn, 'spectra_mean', select_field, select_value + '.csv'])
lines = [['csv_file', 'select_field', 'select_value', 'image'] + bn]
lines += [[csv_fn, select_field, select_value, img_fn] + [str(j) for j in mean]]
print(lines)
lines = [','.join(line) for line in lines]
open(csv_afn, 'wb').write(('\n'.join(lines)).encode())
print('+w', csv_afn)


csv_sfn = '_'.join([csv_fn, 'spectra_stdv', select_field, select_value + '.csv'])
lines = [['csv_file', 'select_field', 'select_value', 'image'] + bn]
lines += [[csv_fn, select_field, select_value, img_fn] + [str(j) for j in stdv]]
lines = [','.join(line) for line in lines]
open(csv_sfn, 'wb').write(('\n'.join(lines)).encode())
print('+w', csv_sfn)
