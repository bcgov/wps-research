'''produce a raster where the bands (and band names in the header) are
re-ordered according to our specifications. IEEE 32-bit bsq float as usual'''
import os
import shutil
from misc import *
args = sys.argv
sep = os.path.sep
pd = sep.join(__file__.split(sep)[:-1]) + sep

# instructions to run
if len(args) < 2:
    err('usage:\n\traster_shuffle_bands.py [input file] [output file] ' +
            '[new position for band 1]... [new position for last band] ' +
            '#note: indices are from 1, not 0')

# check file and header exist
fn, hdr = args[1], hdr_fn(args[1])
assert_exists(fn)
ofn = args[2]
print(' input:', fn)
print('output:', ofn)
print(len(args))

ncol, nrow, nband = [int(x) for x in read_hdr(hdr)]  # read image dims
if len(args) != nband + 3:  # check new pos'n listed for each band
    err('must supply a new position-index for each band')
pos = args[3:] # new positions
try:
    pos = [int(x) - 1 for x in pos]
except:
    err("failed parsing new band pos'n idx")
print(pos)
if len(list(set(pos))) != nband:  # check bands reprsented
    err("one new pos'n-index req'd for each band")
for i in pos:  # check new idx in range
    if i < 0 or i >= nband:
        err('invalid index')

'''write the header'''
bn = band_names(hdr)  # now re-order the header
if len(bn) != len(pos):  # sanity check
    err('unexpected number of band names read')
bn_new = [bn[pos[i]] for i in range(len(bn))]
for b in bn:
    print(b)

ohn = ofn[:-4] + '_tmp.hdr'
write_hdr(ohn, ncol, nrow, nband)

bn_ix = get_band_names_line_idx(open(hdr).read())  # get line idx of band names fields for original file
old_lines = open(hdr).read().strip().split('\n')
for i in range(len(bn)):
    bn_ix_i = int(bn_ix[i])
    old_lines[bn_ix_i] = old_lines[bn_ix_i].replace(bn[i], bn_new[i])

mod = open(ohn).read().strip().split('\n')
mod += [old_lines[bn_ix[i]] for i in range(len(bn))]
print('+w', ohn)
open(ohn, 'wb').write('\n'.join(mod).encode())

real_ohn = ofn[:-4] + '.hdr' # actual output header file
a = shutil.copy(hdr, real_ohn)  # copy the original header file

# replace the bands in the copy of orig header file, with the reordered ones!
a = os.system(' '.join(['python3',
                        pd + 'envi_update_band_names.py',
                        ohn,
                        real_ohn]))
os.remove(ohn) # cleanup fake file
print('+w', real_ohn)

'''write the binary'''
npx = nrow * ncol   # read IEEE 32-bit floats
d = read_float(fn).reshape(nband, npx)
of = open(ofn, 'wb')  # write the re-ordered binary
for i in range(nband):
    print('+w band(' + str(i) + ')')
    d[pos[i],:].astype(np.float32).tofile(of,
                                          '',
                                          '<f4')
of.close()  # close the output ENVI type-4 binary file
bn = band_names(hdr)  # now re-order the header
if len(bn) != len(pos):  # sanity check
    err('unexpected number of band names read')
bn_new = [bn[pos[i]] for i in range(len(bn))]
for b in bn:
    print(b)
print('done')
