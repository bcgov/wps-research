'''20240221 given:
- a raster with geo-location info
- a csv file with field-names that match lat and lon,

Extract the data values under those locations.

Also create a .tgt file for the image (don't overwrite)'''
from misc import xy_to_pix_lin, err, exists, hdr_fn, read_hdr, band_names
import sys
args = sys.argv

fn = args[1]
csv_f = args[2]
tgt_f = args[1] + "_targets.csv" 

if exists(tgt_f):
    err("target file already exists")
if not exists(fn):
    err("please check input file:" + fn)
if not exists(csv_f):
    err("please check input file:" + csv_f)
if fn.split('.')[-1] != 'bin':
    err(".bin extension expected for " + fn)
if csv_f.split('.')[-1] != 'csv':
    err(".csv extension expected for " + csv_f)

ncol, nrow, nband = read_hdr(hdr_fn(fn))
print([ncol, nrow, nband])
bn = band_names(hdr_fn(fn))

lines = open(csv_f).readlines()
lines = [x.strip().split(',') for x in lines]

n_field = None
for line in lines:
    if n_field is None: 
        n_field = len(line)
    if n_field:
        if n_field != len(line):
            err("found line with " + str(len(line)) + " fields, expected: " + str(n_field))
print("All lines had " + str(n_field) + " records")

fields = [f.lower() for f in lines[0]]
data = lines[1:]

lat_i, lon_i, name_i = -1, -1, -1
for i in range(len(fields)):
    f = fields[i]
    if len(f.split('lat')) > 1:
        if lat_i != -1: err("more than one field matched lat")
        else: lat_i = i
    if len(f.split('lon')) > 1:
        if lon_i != -1: err("more than one field matched lon")
        else: lon_i = i    
    if len(f.split('name')) > 1:
        if name_i != -1: err("more than one field matched name")
        else: name_i = i

f = open(tgt_f, "wb")    
f.write("feature_id,row,lin,xoff,yoff".encode())
for line in data:
    lat, lon = line[lat_i], line[lon_i]
    
    row, col, dat = xy_to_pix_lin(fn, lon, lat, int(nband))

    print(lat, lon, row, col)
    print(dat)

    f.write(("\n" + ','.join([line[name_i], str(col), str(row), str(0), str(0) ])).encode())
f.close()

print("band names", bn)
