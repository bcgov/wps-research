''' filter vector data by:
1. translation
2. remove out of range targets!

For example, if on TC site we did: 
gdal_translate -srcwin 7172 5393 2500 2000 2021.bin -of ENVI -ot Float32 sub.bin

we could translate our "imv" targets file by:
python3 ~/GitHub/wps-research/py/vector_target_translation.py 2021.bin_targets.csv 7172 5393 sub.hdr sub.bin_targets.csv
'''

from misc import *

if len(args) < 6:
    print(args)
    err("Usage: vector_target_translation [targets.csv] [xoff] [yoff] [image header file (for new cut image file to translate vectors onto!)] [translated targets file] # optional args: [xmax] [ymax]")

targets = open(args[1]).read().strip().split("\n")
xoff, yoff, outf = None, None, None
xmax, ymax = None, None

try:
    xoff = int(args[2])
except Exception as e:
    xoff = int(open(args[2]).read().strip())

try:
    yoff = int(args[3])
except Exception as e:
    yoff = int(open(args[3]).read().strip())

print("xoff", xoff, "yoff", yoff)

try:
    xmax = int(args[6])
    ymax = int(args[7])
except Exception as e:
    pass

hdr = args[4] # header for what?
outf = open(args[5], "wb")

dat = [t.strip().split(',') for t in targets] # tabluar representation for csv
dat = [[x.strip() for x in d] for d in dat] # remove extra whitespace

cc = {} # check csv well formed
for d in dat:
    c = len(d)
    if c != cc: cc[c] = 0
    cc[c] += 1

print(cc)

if len(cc) != 1:
    err("csv not accepted")

fields = dat[0]
print(fields)
xoi = fields.index("xoff")
yoi = fields.index("yoff")
# print("xoi", xoi)
# print("yoi", yoi)

rowi = fields.index("row")
lini = fields.index("lin")

xo, yo = [], []
row, lin = [], []
for i in range(0, len(dat)):
    d = dat[i]
    xo.append(d[xoi])
    yo.append(d[yoi])
    row.append(d[rowi])
    lin.append(d[lini])
# print(xo)
# print(yo)
# print("row", row)
# print("lin", lin)

ns, nr, nb = read_hdr(hdr)
ns, nr, nb = int(ns), int(nr), int(nb)
i_use = []
for i in range(1, len(dat)):
    row[i] = 0 if (row[i] == '') else int(row[i])
    lin[i] = 0 if (lin[i] == '') else int(lin[i])
    row[i] -= xoff
    lin[i] -= yoff
    dat[i][rowi] = row[i]  # correct the data to output
    dat[i][lini] = lin[i]
    if row[i] >= 0 and lin[i] >= 0 and row[i] <= nr and lin[i] <= ns:
        i_use.append(i)

print("selected features:")
for i in i_use:
    print(i, dat[i][fields.index("feature_id")])

outf.write((','.join(dat[0])).encode())
for i in i_use:
    use = True
    dat[i] = [str(dat[i][j]) for j in range(0, len(dat[i]))]
    
    if xmax is not None and ymax is not None:
        if int(dat[i][rowi]) > xmax or int(dat[i][lini]) > ymax:
            use = False
    if use:
        outf.write(('\n' + (','.join(dat[i]))).encode())
outf.close()
