'''rasterize a target file (one hot encoding)..
..tgt file of the format used in the MVP software

what was this script for? '''
from misc import * 

if len(args) < 3:
    err("python3 vector_target_rasterization.py [input target file] [input raster file)]")
    # python3 vector_target_rasterization.py stack.bin_targets.csv stack.bin

tfn, rfn = args[1], args[2]
hfn = hdr_fn(rfn) # raster header file name

nc, nr, nb = read_hdr(hfn)
nc = int(nc)
nr = int(nr)

lines = open(tfn).readlines()
lines = [x.strip() for x in lines]
w = lines[0].split(',')

lookup = {w[i]: i for i in range(len(w))}
row_i = lookup['row']
lin_i = lookup['lin']
fid_i = lookup['feature_id']

d_s, c = {}, {} # mask for each label, count for each label

for line in lines[1:]:
    x = line.split(',')
    if len(x) != len(w):
        err('csv file not formed to spec. Might have extra commas')

    col = int(x[row_i])
    row = int(x[lin_i]) # line always means row. but row can mean col? haha
    label = x[fid_i] 
    if label not in d_s:
        d_s[label] = np.zeros(nr * nc)
        c[label] = 0
    c[label] += 1

    for di in range(-1, 2, 1):
        i = row + di
        for dj in range(-1, 2, 1):
            j = col + dj
            d_s[label][i * nc + j] = 1.

for label in d_s: # write a mask for each label
    label = label.strip().replace(' ', '_') # spaces in filenames would be bad!
    mfn = rfn[:-4] + '_' + label + '.bin'
    mhfn = mfn[:-4] + '.hdr' # header file name

    write_binary(d_s[label], mfn) # write out a mask file, aka one channel from one-hot encoding
    write_hdr(mhfn, nc, nr, 1) # write out header for the mask filea

for label in d_s:
    print(label, c[label])
