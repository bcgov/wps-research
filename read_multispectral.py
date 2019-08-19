'''20190619 read_multispectral.py: read and visualize multispectral image!

usage:
    python read_multispectral.py sentinel2.bin

tested on Python 2.7.15+ (default, Nov 27 2018, 23:36:35) [GCC 7.3.0] on linux2
Ubuntu 18.04.2 LTS

with
    numpy.version.version
        '1.16.2'
and
    matplotlib.__version__
        '2.2.4'

installation of numpy and matplotlib (Ubuntu):
    sudo apt install python-matplotlib python-numpy
'''
import sys
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    for line in open(hdr).readlines():
        words = line.split('=')
        if len(words) == 2:
            for f in ['samples', 'lines', 'bands']:
                if words[0].strip() == f:
                    exec(f + ' = int(' + words[1].strip() + ')')
    return samples, lines, bands

# print message and exit
def err(c):
    print('Error: ' + c); sys.exit(1)

# instructions to run
if len(sys.argv) < 2:
    err('usage:\n\tread_multispectral.py [input file name]')

# check file and header exist
fn, hdr = sys.argv[1], sys.argv[1][:-4] + '.hdr'
if not path.exists(fn): err('could not find file: ' + fn)
if not path.exists(hdr): err('expected header file: ' + hdr)

# read header and print parameters
samples, lines, bands = read_hdr(hdr)
for f in ['samples', 'lines', 'bands']:
    exec('print("' + f + ' =" + str(' +  f + '));')

# read binary IEEE 32-bit float data
data = np.fromfile(sys.argv[1], '<f4').reshape((bands, lines * samples))
print("bytes read: " + str(data.size))

# select bands for visualization: default value [3, 2, 1]. Try changing to anything from 0 to 12-1==11!
band_select = [0, 0, 0] if bands == 1 else ([11, 8, 2] if bands > 3 else [0, 1, 2])
rgb = np.zeros((samples, lines, 3))
for i in range(0, 3):
    rgb[:,:,i] = data[band_select[i],:].reshape((samples, lines))

values = {}
for i in range(0, lines):
    for j in range(0, samples):
        d = data[0][i* samples + j]
        if d not in values:
            values[d] = 0
        values[d] += 1

print values
print "n_clust", len(values)
plt.title("n_clust " + str(len(values)))
# plot the image
plt.imshow(rgb)
plt.tight_layout()
plt.show()