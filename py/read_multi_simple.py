#!/usr/bin/python3
'''20211202 simple version that was stripped down from 2020 ftl mvp version
usage:
    python read_multi.py sentinel2.bin
    
tested on Python 2.7.15+ (default, Nov 27 2018, 23:36:35) [GCC 7.3.0] on linux2 (Ubuntu 18.04.2 LTS)
with numpy.version.version '1.16.2' and matplotlib.__version__ '2.2.4'

installation of numpy and matplotlib (Ubuntu):
    sudo apt install python-matplotlib python-numpy '''

from misc import *
import matplotlib
args = sys.argv

# instructions to run
if len(args) < 2:
    err('usage:\n\tread_multispectral.py [input file name]')

# check file and header exist
fn, hdr = sys.argv[1], hdr_fn(sys.argv[1])
assert_exists(fn)

# read header and print parameters
samples, lines, bands = read_hdr(hdr)
for f in ['samples', 'lines', 'bands']:
    exec('print("' + f + ' =" + str(' +  f + '))')
    exec(f + ' = int(' + f + ')')
    
npx = lines * samples # number of pixels.. binary IEEE 32-bit float data 
data = read_float(sys.argv[1]).reshape((bands, npx))
print("bytes read: " + str(data.size))

# select bands for visualization: default value [3, 2, 1]. Try changing to anything from 0 to 12-1==11! 
# band_select = [3, 2, 1] if bands > 3 else [0, 1, 2]
band_select = [0, 1, 2]
n_points = 0
if bands == 1:
    band_select = [0, 0, 0,] # could be class map. or just one band map

rgb = np.zeros((lines, samples, 3))
for i in range(0, 3):
    rgb[:, :, i] = data[band_select[i], :].reshape((lines, samples))
    
    if not override_scaling:
        print("scaling")  # scale band in range 0 to 1
        rgb_min, rgb_max = np.nanmin(rgb[:, :, i]), np.nanmax(rgb[:, :, i])
        print("rgb_min: " + str(rgb_min) + " rgb_max: " + str(rgb_max))
        rgb[:, :, i] -= rgb_min
        rng = rgb_max - rgb_min
        if rng != 0.:
            rgb[:, :, i] /= rng #(rgb_max - rgb_min)

        # so called "x-% linear stretch"
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
        if rng > 0.: rgb[:, :, i] /= (rgb_max - rgb_min)

        # need to add this update in misc.py as well, and move this code out
        d = rgb[:, :, i]
    
        for j in range(rgb.shape[0]):
            for k in range(rgb.shape[1]):
                d = rgb[j,k,i]
                if d < 0.: rgb[j,k,i] = 0.
                if d > 1.: rgb[j,k,i] = 1.
print("done scaling..")

# plot the image: no class labels
if True:
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(1,1,1)
    ff = os.path.sep.join((fn.split(os.path.sep))[:-1]) + os.path.sep
    title_s = fn.split("/")[-1] if not exists(ff + 'title_string.txt') else open(ff + 'title_string.txt').read().strip() 
    plt.title(title_s, fontsize=11)
    plt.style.use('dark_background')

    d_min, d_max = np.nanmin(rgb), np.nanmax(rgb)
    print("d_min", d_min, "d_max", d_max)
    rgb = rgb / (d_max - d_min)

    plt.imshow(rgb) #, vmin = 0., vmax = 1.) #plt.tight_layout()
    plt.tight_layout()
    if exists(ff + 'copyright_string.txt'):
        plt.xlabel(open(ff+ 'copyright_string.txt').read().strip())
    plt.savefig(fn + "_rgb.png")
