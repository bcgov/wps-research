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

'''insist on fields xoff and yoff'''
if (not 'xoff' in fields) or (not 'yoff' in fields):
    err("missing req'd fields: xoff, yoff")

spec_fi, nonspec_fi = [], []  # list col-idx for all spectral data columns
for i in range(nf):
    if fields[i][-2:] == 'nm':
        spec_fi += [i]
    else:  # list non-spec fields except: offset-index coding analysis-window pos'n
        if fields[i] not in ['xoff', 'yoff', 'row', 'lin']:
            nonspec_fi += [i]
N = len(data[0])  # number of data points
print('spectra col-ix', spec_fi)
print('nonspec col-ix', nonspec_fi)
print('number of cols', len(spec_fi))
print("number of data points", N)

spectra = {}
for i in range(N):
    key = ','.join([data[j][i] for j in nonspec_fi])
    print(key)
    if key not in spectra:
        spectra[key] = []
    spectrum = [float(data[j][i]) for j in spec_fi]
    spectra[key].append(spectrum)  # list all the spectra for this key

total = 0
for key in spectra:
    total += len(spectra[key])
total /= len(list(spectra.keys()))
print('average number of spectra per point', total)
print('number of keys:', len(list(spectra.keys())))
new_spectra = {}
M = range(len(spec_fi))
for key in spectra:
    spectrums = spectra[key]
    new_spectrum = [0. for i in M]
    for s in spectrums:
        for i in M:
            new_spectrum[i] += s[i]
    new_spectra[key] = new_spectrum

'''now we have new spectrums, output them'''
lines = []
lines.append(','.join([fields[i] for i in nonspec_fi] + [fields[i] for i in spec_fi]))
print('new fields:', lines[0])
for key in new_spectra:
    spectrum = new_spectra[key]
    lines.append(key + ',' +
                 ','.join([str(x) for x in spectrum]))
ofn = in_f + '_averaged.csv'
print('+w', ofn)
open(ofn, 'wb').write(('\n'.join(lines)).encode())
