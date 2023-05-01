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
bni, to_sort, sv = [], [], []
for i in range(len(bn)):  # band name indices
    spectral_value = None
    w = bn[i].strip().strip(',').strip('}').split()
    # print(w)
    found = False
    for j in range(len(w)):
        if w[j][-2:] == 'nm':
            spectral_value = float(w[j][:-2])
            # print(spectral_value)
            to_sort.append([spectral_value, i])
            found = True
    if not found:
        spectral_value = sys.float_info.max
        to_sort.append([spectral_value, i])
    sv.append(spectral_value)

to_sort.sort(reverse=False)
print("to_sort", to_sort)
lookup = {i:to_sort[i][1] for i in range(len(to_sort))}
print(lookup)
'''
for si in range(len(to_sort)):
    s = to_sort[si]
    lookup[s[1]] = si '''
# print("new thing in a position", lookup)
# for si in range(nband):
#    print(si, '->', lookup[si])

s = ''
for i in range(len(bn)):
    s += (str(lookup[i] + 1) + ' ')  # haha we used 1-idxing
s = s.strip()
print('s', s)
default = ' '.join([str(x) for x in list(range(1, int(nband) + 1))])

if s != default:
    # now reorder the input file, to a temp file..
    ofn, ofh = fn + '_reorder.bin', fn + '_reorder.hdr'
    run(' '.join(['python3',
                  pd + 'raster_shuffle_bands.py',
                  fn,
                  ofn,
                  s]))
    # now overwrite the input file, and the input header file, with the created files!
    shutil.move(ofn, fn)
    shutil.move(ofh, hdr)
# print('done')
