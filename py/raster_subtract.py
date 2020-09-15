# subtract two n-dimensional rasters
from misc import *
args = sys.argv

# instructions to run
if len(args) < 2:
    err('usage:\n\tsubtract.py [input file subtract from] [input file to subtract] [output file]')

# check file and header exist
fn_1, hdr_1 = args[1], hdr_fn(args[1]); assert_exists(fn_1)
fn_2, hdr_2 = args[2], hdr_fn(args[2]); assert_exists(fn_2)
ofn = args[3]

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
