'''in-place/overwrite reordering of raster bands based on wavelength..
..only reorder the fields that are in nm (but put them first)'''
import shutil
from misc import *
args = sys.argv
sep = os.path.sep
pd = sep.join(__file__.split(sep)[:-1]) + sep

# instructions to run
if len(args) < 2:
    err('usage:\n\traster_reorder_increasing_nm.py [input file]')

# check file and header exist
fn, hdr = args[1], hdr_fn(args[1])
assert_exists(fn)

ncol, nrow, nband = [int(x) for x in read_hdr(hdr)]  # read image dims
bn = band_names(hdr)  # read the band names

# find the band-name indices that have a space-separated field, ending with nm
bni = []
to_sort = []
for i in range(len(bn)):  # band name indices
    spectral_value = None
    w = bn[i].strip().split()
    for j in range(len(w)):
        if w[j][-2:] == 'nm':
            spectral_value = float(w[j][:-2])
            print(spectral_value)
            to_sort.append([spectral_value, i])
to_sort.sort(reverse=False)
for s in to_sort:
    print(s)

lookup = {}
for si in range(len(to_sort)):
    s = to_sort[si]
    lookup[s[1]] = si

for s in lookup:
    print(s, '->', lookup[s])

s = ''
for i in range(len(bn)):
    s += (str(lookup[i] + 1) + ' ')  # haha we used 1-idxing
s = s.strip()
print(s)

# now reorder the input file, to a temp file..
ofn_tmp = fn[:-4] + '_tmp.bin'
ofhn_tmp = fn[:-4] + '_tmp.hdr'

run(' '.join(['python3',
              pd + 'raster_shuffle_bands.py',
              fn,
              ofn_tmp,
              s]))

# now overwrite the input file, and the input header file, with the created files!
shutil.move(ofn_tmp, fn)
shutil.move(ofhn_tmp, hdr)
print('done')
