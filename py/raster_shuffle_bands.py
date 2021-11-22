'''produce a raster where the bands (and band names in the header) are
re-ordered according to our specifications. IEEE 32-bit bsq float as usual'''
from misc import *
args = sys.argv

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
    pos = [int(x) -1 for x in pos]
except:
    err("failed parsing new band pos'n idx")
print(pos)

# check each band represented once
if len(list(set(pos))) != nband:
    err("one new pos'n-index req'd for each band")

# check new idx are all in range
for i in pos:
    if i < 0 or i >= nband:
        err('invalid index')

npx = nrow * ncol   # read IEEE 32-bit floats
d = read_float(fn).reshape(bands, npx)

sys.exit(1)

# read header and print parameters
samples_1, lines_1, bands_1 = read_hdr(hdr_1)
for f in ['samples_1', 'lines_1', 'bands_1']:
    exec(f + ' = int(' + f +')')

samples_2, lines_2, bands_2 = read_hdr(hdr_2)
for f in ['samples_2', 'lines_2', 'bands_2']:
    exec(f + ' = int(' + f +')')

if not(samples_1 == samples_2 and lines_1 == lines_2 and bands_1 == bands_2):
    err('input files dimension mismatch')

if os.path.exists(ofn):
    err("output file already exists: " + ofn)

# read binary IEEE 32-bit float data
npx = lines_1 * samples_1 # number of pixels
d_1 = read_float(fn_1).reshape((bands_1, npx))
d_2 = read_float(fn_2).reshape((bands_1, npx))
print("d_1", d_1.shape)

mu_1, mu_2 = np.zeros((bands_1)), np.zeros((bands_1))
d_3 = np.zeros((bands_1, npx), dtype = np.float32)

print("d_3", d_3.shape)

for i in range(0, npx):
    for j in range(0, bands_1):
        d_3[j, i] = abs(d_1[j, i] - d_2[j, i])

write_binary(d_3, ofn)
ohfn = (ofn[:-4] + '.hdr') if (ofn[-4:] == '.bin') else (ofn + '.hdr')

write_hdr(ohfn, samples_1, lines_1, bands_1)
