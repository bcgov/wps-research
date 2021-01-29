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


'''use c program to count data'''
sep = os.path.sep
path = sep.join(__file__.split(sep)[:-1]) + sep  # path to this file
print("path", path)
path = os.path.abspath(os.path.expanduser(os.path.expandvars(path))) + sep
p = path + "../cpp/"
run("rm -f " + p + "class_count.exe")
if not exist(p + "class_count.exe"):
    run("g++ -w -O3 " + p + "class_count.cpp " + p + "misc.cpp -o " + p + "class_count.exe -lpthread")

class_count = os.popen(p + "class_count.exe " + fn).read().strip().replace("\n", " ").replace("NAN", "float(\"NaN\")")
statement = "count_by_label=" + class_count
print(statement)
exec(statement)
print(count_by_label)

# select bands for visualization: default value [3, 2, 1]. Try changing to anything from 0 to 12-1==11! 
# band_select = [3, 2, 1] if bands > 3 else [0, 1, 2]
band_select = [0, 1, 2]
kmeans_labels = {}
kmeans_labels_by_class = None
percent_by_label, percent_confused = {}, 0
confused_labels, confused_kmeans_labels = set(), set()
n_points = 0
n_nan = class_count[float('NaN')]
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
    try:
        labels = labels.replace("kmeans_label_by_class", "kmeans_label_by_class=")
    except:
        pass
    statement = labels if labels is not None else "kmeans_label_by_class={}"
    print("statement", [statement])
    exec(statement)

    print("kmeans_labels_by_class", kmeans_label_by_class)
    # kmeans_labels_by_class {'fireweeddeciduous': {2.0}, 'pineburneddeciduous': {11.0}, 'blowdownfireweed': {7.0}, 'blowdownlichen': {10.0, 3.0}, 'windthrowgreenherbs': {9.0}, 'exposed': {0.0, 6.0}, 'fireweedgrass': {8.0}, 'pineburned': {3.0, 4.0}, 'pineburnedfireweed': {4.0}, 'herb': {6.0}, 'lake': {1.0}, 'conifer': {12.0}, 'deciduous': {5.0}}
    classes_by_kmeans_label = {}
    for L in kmeans_label_by_class: # L is the text label...
        # print("L", L)
        for lab in kmeans_label_by_class[L]:
            if lab not in classes_by_kmeans_label:
                classes_by_kmeans_label[lab] = set()
            classes_by_kmeans_label[lab].add(L)

    kmeans_labels = classes_by_kmeans_label
    #kmeans_labels {2.0: {'fireweeddeciduous'}, 11.0: {'pineburneddeciduous'}, 7.0: {'blowdownfireweed'}, 10.0: set(), 3.0: {'blowdownlichen'}, 9.0: {'windthrowgreenherbs'}, 0.0: set(), 6.0: {'exposed', 'herb'}, 8.0: {'fireweedgrass'}, 4.0: {'pineburnedfireweed', 'pineburned'}, 1.0: {'lake'}, 12.0: {'conifer'}, 5.0: {'deciduous'}}
    print('data op')
    data = data.tolist()[0] # not sure why the data packed wierdly in here
    '''
    n_points = len(data)
    for i in range(len(data)):
        if i % 10000 == 0: 
            print(i, len(data))
        d = data[i]
        if math.isnan(d):
            # rint("NAN")
            n_nan += 1
        if d not in count_by_label:
            count_by_label[d] = 0
        count_by_label[d] += 1
    '''
    n_points =0
    for label in count_by_label:
        n_points += count_by_label[label] 

    print("count_by_label", count_by_label)
    print("kmeans_labels", kmeans_labels)
    for label in count_by_label:
        if label not in kmeans_labels:
            kmeans_labels[label] = "None"
        percent_by_label[label] = 100. * count_by_label[label] / float(len(data))
        if len(kmeans_labels[label]) > 1:
            percent_confused += percent_by_label[label]
            for c in kmeans_labels[label]:
                confused_labels.add(c)
            confused_kmeans_labels.add(str(kmeans_labels[label]))

    data = np.array(data).reshape((bands, npx))
        
    if str(kmeans_labels) != str("{}"):
        print("WHAT ARE WE DOING")
        data = data.tolist()[0] # not sure why the data packed wierdly in here
        for i in range(npx):
            #if i % 10000 == 0: 
            #    print(i, len(data))
            if data[i] == float("NaN"):
                data[i] = 0.
            else:
                data[i] = data[i] + 1.
        data = np.array(data).reshape((bands, npx))

print("kmeans_labels", kmeans_labels)

rgb = np.zeros((lines, samples, 3))
for i in range(0, 3):
    rgb[:, :, i] = data[band_select[i], :].reshape((lines, samples))
    
    if not override_scaling:
        print("scaling")
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
print("done scaling..")

# plot the image: no class labels
if str(kmeans_labels) == "{}":
    fig = plt.figure()
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

# plot image with class labels
if str(kmeans_labels) != "{}":
    print("plotting..")
    data = read_float(sys.argv[1])

    # d_min, d_max = np.nanmin(data), np.nanmax(data)
    #data = (data + 1.) / (1. + d_max - d_min)
    #for i in range(0, len(data)):
    #    if math.isnan(data[i]):
    #        data[i] = 0.
    # data = data / (d_max - d_min)
    #values = set(data)
    # print("values", values)
    data = data.reshape((lines, samples))
    # fig = plt.figure()
    fig, ax = plt.subplots()
    ff = os.path.sep.join((fn.split(os.path.sep))[:-1]) + os.path.sep
    title_s = fn.split("/")[-1] if not exists(ff + 'title_string.txt') else open(ff + 'title_string.txt').read().strip()
    title_s += " percent confused: %" + str(round(percent_confused, 2))
    plt.title(title_s, fontsize=11)
    # plt.style.use('dark_background')A
    img = ax.imshow(data, cmap='Spectral')

    import collections
    print("kmeans_labels", kmeans_labels)
    kmeans_labels = collections.OrderedDict(sorted(kmeans_labels.items()))
    print("kmeans_labels", kmeans_labels)
    cbar = plt.colorbar(img) # p.array(data)) #gb)#  .legend([0, 1, 2, 3], ['0', '1', '2', '3'])\
    tick_labels = [] # "noise"]
    ticks = []
    ci = 0 
    print("kmeans_labels", kmeans_labels)
    for label in kmeans_labels: # eans_label_by_class:
        x = kmeans_labels[label] #_by_class[label]
        try:
            my_percent =  " %" + str(round(percent_by_label[label], 2))
        except:
            print("count_by_label", count_by_label)
            print("percent_by_label", percent_by_label, "label", label)
            err("fail")
        tick_labels.append(str(label) + " --> " + str(x) + my_percent ) # this is the "set of classes" label
        ticks.append(label) # this is the float label
        if set([ci]) != x:
            print(str(set([ci])), str(x))
            # err("color index problem")
        ci += 1
    cbar.set_ticks(ticks) #/ (d_max - d_min)) # p.arange(len(tick_labels)) / (d_max - d_min)) #np.arange(len(tick_labels)) + 1) / (1. + d_max - d_min))
    print("tick_labels", tick_labels)
    print("ticks", ticks) 
    cbar.ax.set_yticklabels(tick_labels) #"bad", "good", "other", "more", "what"])
    # plt.xlabel("confused labels: " + str(confused_kmeans_labels))
    plt.xlabel(str("".join([ str( x) for x in ["n_nan ", n_nan, " n_points ", n_points]])))
    print("confused labels:", confused_kmeans_labels)
plt.tight_layout()
plt_fn = fn + ".png"
print("+w", plt_fn)

plt.savefig(plt_fn,
            dpi=300,
            pad_inches =0.)

if show_plot:
    plt.show()
