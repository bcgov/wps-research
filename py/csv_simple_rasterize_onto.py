'''DEPRECATED
20211216: burn point locations from a csv into a raster mask..
..use this to convert csv groundref locations into format used by TRU students! 

* input csv file must include columns named: row, col
* raster file provided is only used to read expected image dimensions from the header.'''
from misc import *
import matplotlib
import numpy
args = sys.argv

# instructions to run
if len(args) < 2:
    err('usage:\n\tpython3 csv_simple_rasterize_onto [csv file name] [raster filename]')

csv_fn, fn = args[1], args[2]  # csv and raster input file names
for i in [fn, csv_fn]:
    assert_exists(i)  # check file exists
hdr = hdr_fn(args[2]) # check raster header file exists
samples, lines, bands = [int(i) for i in read_hdr(hdr)]  # read raster image dims
npx = samples * lines # number of image pixels

flag = csv_fn[:-4].split('_')[-1]
ofn = fn + '_mask_' + flag + '.bin'
print('+w', ofn)  # output mask file location to write
ohn = ofn[:-4] + '.hdr'  # output header file name to write

fields, data = read_csv(args[1]) # open the csv data
for i in ['row', 'col']:  # check for required fields
    if i not in fields:
        err('required field: ' + i)
N = len(data[0])  # number of csv records
f_i = {fields[i]: i for i in range(len(fields))} # col index lookup

# rasterize the locations
dat = numpy.zeros(npx)
for i in range(N):
    row, col = [int(x) for x in
                [data[f_i['row']][i], data[f_i['col']][i]]]
    print('  row,col=', row, col)
    dat[row * samples + col] = 1.  # burn in the point location
write_binary(dat, ofn)  # write output raster mask

write_hdr(ohn, samples, lines, 1)  # write output header
