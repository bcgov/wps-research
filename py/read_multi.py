#!/usr/bin/python3
'''20190619 read_multispectral.py: read and visualize multispectral image!

adapted from a script worked on with francesca at: https://github.com/franarama/satellite-clustering

usage:
    python read_multi.py sentinel2.bin
    
tested on Python 2.7.15+ (default, Nov 27 2018, 23:36:35) [GCC 7.3.0] on linux2 (Ubuntu 18.04.2 LTS)
with numpy.version.version '1.16.2' and matplotlib.__version__ '2.2.4'

installation of numpy and matplotlib (Ubuntu):
    sudo apt install python-matplotlib python-numpy '''
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
    exec('print("' + f + ' =" + str(' +  f + '))')
    exec(f + ' = int(' + f + ')')
    
# read binary IEEE 32-bit float data
npx = lines * samples # number of pixels
data = read_float(sys.argv[1]).reshape((bands, npx))
print("bytes read: " + str(data.size))

# select bands for visualization: default value [3, 2, 1]. Try changing to anything from 0 to 12-1==11! 
band_select = [3, 2, 1] if bands > 3 else [0, 1, 2]

kmeans_labels = {}
if bands == 1:
    # could be a class map! Or just a one-band map..
    band_select = [0, 0, 0,]

    # detect class map! Should have label field in header!
    ls = [line.strip() for line in open(hdr).readlines()]
    labels = None
    for line in ls:
        w = line.split()
        try:
            if w[0] == "kmeans_label_by_class":
                labels = line
        except:
            pass
    labels = labels.replace("kmeans_label_by_class", "kmeans_label_by_class=")
    exec(labels)
    print(kmeans_label_by_class)
    for L in kmeans_label_by_class:
        for lab in kmeans_label_by_class[L]:
            kmeans_labels[lab] = kmeans_labels[lab] if lab in kmeans_labels else set()
        kmeans_labels[lab].add(L)

    if str(kmeans_labels) != str("{}"):
        data = data.tolist()[0]
        for i in range(npx):
            if data[i] == float("NaN"):
                data[i] = 0.
            else:
                data[i] = data[i] + 1.
        data = np.array(data).reshape((bands, npx))

print(kmeans_labels, "kmeans_labels")

rgb = np.zeros((lines, samples, 3))
for i in range(0, 3):
    rgb[:, :, i] = data[band_select[i], :].reshape((lines, samples))
    
    if not override_scaling:
        # scale band in range 0 to 1
        rgb_min, rgb_max = np.nanmin(rgb[:, :, i]), np.nanmax(rgb[:, :, i])
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
    
        for j in range(rgb.shape[0]):
            for k in range(rgb.shape[1]):
                d = rgb[j,k,i]
                if d < 0.:
                    rgb[j,k,i] = 0.
                if d > 1.:
                    rgb[j,k,i] = 1.

        # (rgb[:, :, i])[d < 0.] = 0.
        # (rgb[:, :, i])[d > 1.] = 1.

# plot the image
if kmeans_labels == {}:
    plt.style.use('dark_background')
    ff = os.path.sep.join((fn.split(os.path.sep))[:-1]) + os.path.sep
    title_s = fn.split("/")[-1] if not exists(ff + 'title_string.txt') else open(ff + 'title_string.txt').read().strip() 
    plt.title(title_s, fontsize=11)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(rgb, vmin = 0., vmax = 1.) #plt.tight_layout()

    if exists(ff + 'copyright_string.txt'):
        plt.xlabel(open(ff+ 'copyright_string.txt').read().strip())

if kmeans_labels != {}:
    data = read_float(sys.argv[1])
    data = data.reshape((lines, samples))
    fig = plt.figure()
    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap='Spectral')

    import collections
    kmeans_labels = collections.OrderedDict(sorted(kmeans_labels.items()))
    print("kmeans_labels", kmeans_labels)
    cbar = plt.colorbar(img) # p.array(data)) #gb)#  .legend([0, 1, 2, 3], ['0', '1', '2', '3'])\
    tick_labels = [] # "noise"]
    ci = 0 
    for label in kmeans_labels: # eans_label_by_class:
        x = kmeans_labels[label] #_by_class[label]
        tick_labels.append(x) # label)

        if set([ci]) != x:
            print(str(set([ci])), str(x))
            # err("color index problem")
        ci += 1
    cbar.set_ticks(np.arange(len(tick_labels)))
    print("tick_labels", tick_labels)
    cbar.ax.set_yticklabels(tick_labels) #"bad", "good", "other", "more", "what"])

plt.tight_layout()
plt_fn = fn + ".png"
print("+w", plt_fn)

plt.savefig(plt_fn,
        dpi=300,
        pad_inches =0.)

if show_plot: plt.show()

'''
compare with this:

print("X.shape", X.shape)
    a[:, :, 0] = X.S2A_4.values.reshape(lines, samples)
    a[:, :, 1] = X.S2A_3.values.reshape(lines, samples)
    a[:, :, 2] = X.S2A_2.values.reshape(lines, samples)
    a = (a - np.min(a)) / np.max(a)

    for i in range(0, 3):
        d = a[:, :, i]
        npx = samples * lines
        values = d.reshape(np.prod(d.shape)).tolist()
        values.sort()
        mn = values[int(math.floor(float(npx) * 0.01))]
        mx = values[int(math.floor(float(npx) * 0.99))]
        print("i", i, "mn", mn, "mx", mx)
        rng = mx - mn
        a[:, :, i] -= mn 
        if rng > 0.:
            a[:, :, i] /= rng
        (a[:, :, i])[a[:, :, i] < 0.] = 0.
        (a[:, :, i])[a[:, :, i] > 1.] = 1.
'''
