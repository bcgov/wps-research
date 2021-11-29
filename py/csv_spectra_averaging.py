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

spec_fi, non_spec_fi = [], []  # list col-idx for all spectral data columns
for i in range(nf):
    if fields[i][-2:] == 'nm':
        spec_fi += [i]
    else: 
        non_spec_fi += [1]

print('spectra col-ix', spec_fi)
print('number of cols', len(spec_fi))

N = len(data[0]) # number of data points
print("number of data points", N)

x = range(len(spec_fi))
for i in range(N):

    spectrum = [float(data[j][i]) for j in spec_fi]

'''
feature_id,ctr_lat,ctr_lon,image,row,lin,xoff,yoff,20190908 60m: B1 443nm,20190908 10m: B2 490nm,20190908 10m: B3 560nm,20190908 10m: B4 665nm,20190908 20m: B5 705nm,20190908 20m: B6 740nm,20190908 20m: B7 783nm,20190908 10m: B8 842nm,20190908 20m: B8A 865nm,20190908 60m: B9 945nm,20190908 20m: B11 1610nm,20190908 20m: B12 2190nm,20210729 60m: B1 443nm,20210729 10m: B2 490nm,20210729 10m: B3 560nm,20210729 10m: B4 665nm,20210729 20m: B5 705nm,20210729 20m: B6 740nm,20210729 20m: B7 783nm,20210729 10m: B8 842nm,20210729 20m: B8A 865nm,20210729 60m: B9 945nm,20210729 20m: B11 1610nm,20210729 20m: B12 2190nm
1,-131.1119300232046,58.094511533295574,raster.bin,7550,5885,0,0,321.923614501953,408.0,664.0,677.0,1079.1875,1881.3125,2052.625,2255.0,2207.6875,2235.91674804688,2484.5,1830.75,320.277770996094,393.0,656.0,545.0,1106.8125,2442.375,2691.0625,3004.0,2858.8125,2725.73608398438,2257.8125,1400.25
1,-131.1119300232046,58.094511533295574,raster.bin,7549,5885,-1,0,328.4375,435.0,679.0,728.0,1070.5625,1811.4375,1979.875,2100.0,2125.0625,2213.58325195312,2459.5,1818.25,327.166656494141,381.0,673.0,580.0,1090.9375,2342.125,2647.6875,2820.0,2814.9375,2689.375,2240.4375,1396.25
'''


