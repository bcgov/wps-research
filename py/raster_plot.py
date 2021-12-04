#!/usr/bin/python3
'''20211202 simple version that was stripped down from 2020 ftl mvp version
usage: e.g:
    python rasterplot.py sentinel2.bin 4 3 2
    
tested on Python 3.8.10 (default, Sep 28 2021, 16:10:42) 
with numpy.version.version '1.20.2' and matplotlib.__version__ '3.4.1'

e.g. installation of numpy and matplotlib (Ubuntu):
    sudo apt install python-matplotlib python-numpy '''
from misc import *
import matplotlib
args = sys.argv

# instructions to run
if len(args) < 2:
    err('usage:\n\tread_multispectral.py [input file name]' +
        ' [optional: red band idx]' + 
        ' [optional: green band idx]' +
        ' [optional: blue band idx] #band idx from 1')
fn, hdr = sys.argv[1], hdr_fn(sys.argv[1])  # check header exists
assert_exists(fn)  # check file exists

samples, lines, bands = read_hdr(hdr)  # read header and print parameters
for f in ['samples', 'lines', 'bands']:
    exec('print("' + f + ' =" + str(' +  f + '))')
    exec(f + ' = int(' + f + ')')
    
npx = lines * samples # number of pixels.. binary IEEE 32-bit float data 
data = read_float(sys.argv[1]).reshape((bands, npx))
print("bytes read: " + str(data.size))

bn = None
try: bn = band_names(hdr)  # try to read band names from hdr
except: pass

# select bands for visualization # band_select = [3, 2, 1] if bands > 3 else [0, 1, 2]
band_select, ofn = [0, 1, 2], None
try:  # see if we can set the (r,g,b) encoding (band selection) from command args
    for i in range(0, 3):
        bs = int(args[i + 2]) - 1
        if bs < 0 or bs >= bands: 
            err('band index out of range')
        band_select[i] = bs
except: pass

middle = None  # reproducibility: put band idx used, in output fn
try: middle = args[2: 2 + 3]
except: middle = [str(b + 1) for b in band_select]
ofn = '_'.join([fn] + middle + ["rgb.png"])

n_points, rgb = 0, np.zeros((lines, samples, 3))
band_select = [0, 0, 0,] if bands == 1 else band_select # could be class map. or just one band map
bn = [bn[i] for i in band_select] if bn else bn  # cut out the band names used, if applicable
print("band_select", band_select)

def scale_rgb(i):  # for i in range(3)
    rfn = fn + '_rgb_scaling_' + str(i) + '.txt'
    rgb_min, rgb_max = None, None
    rgb_i = data[band_select[i], :].reshape((lines, samples))
    
    if True: # if not override_scaling
        if not exists(rfn):
            values = rgb_i  # now do the so called x-% linear stretch (separate bands version)
            values = values.reshape(np.prod(values.shape)).tolist()
            values.sort()

            if values[-1] < values[0]:   # sanity check
                err("failed to sort")

            for j in range(0, npx - 1):
                if values[j] > values[j + 1]:
                    err("failed to sort")

            n_pct = 2. # percent for stretch value
            frac = n_pct / 100.
            rgb_min, rgb_max = values[int(math.floor(float(npx)*frac))],\
                           values[int(math.floor(float(npx)*(1. - frac)))]
            print('+w', rfn)
            open(rfn, 'wb').write((','.join([str(x) for x in [rgb_min, rgb_max]])).encode())
            # DONT FORGET TO WRITE THE FILE HERE
        else:  # assume we can restore
            rgb_min, rgb_max = [float(x) \
                    for x in open(rfn).read().strip().split(',')]
        
        rng = rgb_max - rgb_min  # apply restored or derived scaling
        rgb_i = (rgb_i - rgb_min) / (rng if rng != 0. else 1.)

        rgb_i[rgb_i < 0.] = 0.  # clip
        rgb_i[rgb_i > 1.] = 1.
    return rgb_i

rgb_i = parfor(scale_rgb, range(3), 3)
for i in range(3):
    rgb[:, :, i] = rgb_i[i]

if True:  # plot image: no class labels 
    base_in = 12.
    fig = plt.figure(figsize=(base_in, base_in * float(lines) / float(samples)))
    ax = fig.add_subplot(1, 1, 1)
    ff = os.path.sep.join((os.path.abspath(fn).split(os.path.sep))[:-1]) + os.path.sep
    title_s = fn.split("/")[-1] if not exists(ff + 'title_string.txt') else open(ff + 'title_string.txt').read().strip() 
    x_label = ''
    if bn:
        x_label += '(R,G,B) = (' + (','.join(bn)) + ')'
    plt.title(title_s, fontsize=11)
    plt.style.use('dark_background')

    d_min, d_max = np.nanmin(rgb), np.nanmax(rgb)
    print("d_min", d_min, "d_max", d_max)
    rgb = rgb / (d_max - d_min)

    plt.imshow(rgb) #, vmin = 0., vmax = 1.) #plt.tight_layout()
    print(ff)
    if exists(ff + 'copyright_string.txt'):
        x_label += (' ©' + open(ff+ 'copyright_string.txt').read().strip())
    plt.xlabel(x_label)
    print("+w", ofn)
    plt.tight_layout()
    plt.savefig(ofn)
    plt.show()
