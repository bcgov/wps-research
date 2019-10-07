'''20190619 read_multispectral.py: read and visualize multispectral image!

adapted from a script we originally worked on together at: https://github.com/franarama/satellite-clustering

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
from misc import *

# instructions to run
if len(sys.argv) < 2:
    err('usage:\n\tread_multispectral.py [input file name]')

# check file and header exist
fn, hdr = sys.argv[1], sys.argv[1][:-4] + '.hdr'
assert_exists([fn, hdr])

#assert_exists(hdr)

# read header and print parameters
samples, lines, bands = read_hdr(hdr)
for f in ['samples', 'lines', 'bands']:
    exec('print("' + f + ' =" + str(' +  f + '));')

# read binary IEEE 32-bit float data
data = np.fromfile(sys.argv[1], '<f4').reshape((bands, lines * samples))
print("bytes read: " + str(data.size))

# select bands for visualization: default value [3, 2, 1]. Try changing to anything from 0 to 12-1==11! 
band_select = [3, 2, 1]
rgb = np.zeros((lines, samples, 3))
for i in range(0, 3):
    rgb[:, :, i] = data[band_select[i],:].reshape((lines, samples))
    
    # scale band in range 0 to 1
    rgb_min, rgb_max = np.min(rgb[:, :, i]), np.max(rgb[:, :, i])
    rgb[:, :, i] -= rgb_min
    rgb[:, :, i] /= (rgb_max - rgb_min)

    # so called "2% linear stretch"
    values = copy.deepcopy(rgb[:,:,i]) # values.shape
    values = values.reshape(np.prod(values.shape))
    values = values.tolist() # len(values)
    values.sort()
    npx = len(values) # number of pixels
    if values[-1] < values[0]:
        err("failed to sort")

    rgb_min = values[int(math.floor(float(npx)*0.02))]
    rgb_max = values[int(math.floor(float(npx)*0.98))]
    rgb[:, :, i] -= rgb_min
    rgb[:, :, i] /= (rgb_max - rgb_min)



# plot the image
plt.imshow(rgb)
plt.tight_layout()
plt.savefig(fn + ".png")
plt.show()
