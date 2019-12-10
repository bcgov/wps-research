#!/usr/bin/env python
'''20190619 read_multispectral.py: read and visualize multispectral image!

adapted from a script worked on with francesca at: https://github.com/franarama/satellite-clustering

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
args = sys.argv

# instructions to run
if len(args) < 2:
    err('usage:\n\tread_multispectral.py [input file name]\n' +
        'optional args: [hide plot] [cancel scaling]')

show_plot = len(args) < 3
override_scaling = len(args) > 3

# check file and header exist
fn, hdr = sys.argv[1], hdr_fn(sys.argv[1])
assert_exists(fn)

# read header and print parameters
samples, lines, bands = read_hdr(hdr)
for f in ['samples', 'lines', 'bands']:
    exec('print("' + f + ' =" + str(' +  f + '));')

# read binary IEEE 32-bit float data
npx = lines * samples # number of pixels
data = read_float(sys.argv[1]).reshape((bands, npx))
print("bytes read: " + str(data.size))

# select bands for visualization: default value [3, 2, 1]. Try changing to anything from 0 to 12-1==11! 
band_select = [3, 2, 1] if bands > 3 else [0, 1, 2]
if bands == 1:
    band_select = [0, 0, 0,]
rgb = np.zeros((lines, samples, 3))
for i in range(0, 3):
    rgb[:, :, i] = data[band_select[i],:].reshape((lines, samples))
    
    if not override_scaling:
        # scale band in range 0 to 1
        rgb_min, rgb_max = np.min(rgb[:, :, i]), np.max(rgb[:, :, i])
        print("rgb_min: " + str(rgb_min) + " rgb_max: " + str(rgb_max))
        rgb[:, :, i] -= rgb_min
        rng = rgb_max - rgb_min
        if rng != 0.:
            rgb[:, :, i] /= rng #(rgb_max - rgb_min)

        # so called "1% linear stretch"
        values = rgb[:, :, i]
        values = values.reshape(np.prod(values.shape)).tolist()
        values.sort()

        # sanity check
        if values[-1] < values[0]:
            err("failed to sort")

        for j in range(0, npx - 1):
            if values[j] > values[j + 1]:
                err("failed to sort")

        rgb_min = values[int(math.floor(float(npx)*0.01))]
        rgb_max = values[int(math.floor(float(npx)*0.99))]
        rgb[:, :, i] -= rgb_min
        rng = rgb_max - rgb_min
        if rng > 0.:
            rgb[:, :, i] /= (rgb_max - rgb_min)

        # need to add this update in misc.py as well, and move this code out
        d = rgb[:, :, i]
        (rgb[:, :, i])[d < 0.] = 0.
        (rgb[:, :, i])[d > 1.] = 1.

# plot the image
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.imshow(rgb, vmin = 0., vmax = 1.)
plt.tight_layout()
title_s = fn.split("/")[-1]
plt.title(title_s, fontsize=11)
plt_fn = fn + ".png"
print("+w", plt_fn)
plt.savefig(plt_fn,
        dpi=300,
        pad_inches =0.)
if show_plot:
    plt.show()
