import os
import sys
import math
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def err(m):
    print('error: ' + m); sys.exit(1)

def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    for line in open(hdr).readlines():
        line = line.strip()
        words = line.split('=')
        if len(words) == 2:
            f, g = words[0].strip(), words[1].strip()
            if f == 'samples': samples = g
            if f == 'lines': lines = g
            if f == 'bands': bands = g
    print('nrow ' + str(lines) + ' ncol ' + str(samples) + ' nband ' + str(bands))
    return [int(x) for x in [samples, lines, bands]]

# use numpy to read a floating-point data file (4 bytes per float, byte order 0)
def read_float(fn):
    print("+r", fn)
    return np.fromfile(fn, dtype = np.float32) # "float32") # '<f4')

def imshow_100(rgb, samples, lines, fn, grey=False):
    dpi = 80
    margin = 0.05 # (5% of the width/height of the figure...)
    xpixels, ypixels = lines, samples # samples, lines
    figsize = (1 + margin) * ypixels / dpi, (1 + margin) * xpixels / dpi
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    if grey:
        ax.imshow(rgb, interpolation='none', cmap='gray')
    else:
        ax.imshow(rgb, interpolation='none')
    print("+w " + fn)
    plt.savefig(fn)

fuels_f = 'ftl2018.bin'  # 2018 fuels data
fuels_h = 'ftl2018.hdr'  # header file for raster
lookup_f = 'FM_FUEL_TYPE_GRID_BC_2018.tif.clr'  # colour lookup file

lines = [[int(i) for i in x.strip().split()] for x in open(lookup_f).readlines()]
print(lines)

lookup_rgb = {}
for line in lines:
    lookup_rgb[line[0]] = line[1:]
print(lookup_rgb)

samples, lines, bands = read_hdr(fuels_h) # int(s) for s in read_hdr(fuels_h)]
print(samples, lines, bands)

d = read_float(fuels_f)
d = np.reshape(d, (lines, samples))

def apply_lookup(d):
    rgb = np.zeros((lines, samples, 3))
    for i in range(lines):
        for j in range(samples):
            di = d[i, j]
            dj = lookup_rgb[int(di)]
            rgb[i, j, 0] = dj[0]
            rgb[i, j, 1] = dj[1]
            rgb[i, j, 2] = dj[2]
    for i in range(3):
        rgb[:, :, i] /= 255.
    return rgb

rgb = apply_lookup(d)

if not os.path.exists('ftl.png'):
    imshow_100(rgb,
               samples, lines,
              'ftl.png')

''' load before image '''
samples2, lines2, bands2 = read_hdr('before.hdr')
img_before = read_float('before.bin')

''' load after image '''
samples3, lines3, bands3 = read_hdr('after.hdr')
img_after = read_float('after.bin')

if samples2 != samples or samples3 != samples or lines2 != lines or lines3 != lines:
    err('check image raster dimensions')

if bands2 != bands3:
    err('band count must match for before and after images')

npx = lines * samples
def reformat(d, samples, lines, bands):
    d2 = np.zeros((lines, samples, bands))
    for i in range(lines):
        for j in range(samples):
            for k in range(bands):
                d2[i,j,k] = d[k * npx + i * samples + j]
    return d2

before = reformat(img_before, samples, lines, bands2)
after  = reformat(img_after,  samples, lines, bands2)

def two_p(dat, samples, lines, bands):
    band_select = [0, 1, 2]  # assume data is already rgb for now ..
    rgb = np.zeros((lines, samples, 3))

    for i in range(0, 3):
        rgb[:, :, i] = dat[:, :, band_select[i]]

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
        if rng > 0.: rgb[:, :, i] /= (rgb_max - rgb_min)

        # need to add this update in misc.py as well, and move this code out
        d = rgb[:, :, i]

        for j in range(rgb.shape[0]):
            for k in range(rgb.shape[1]):
                d = rgb[j,k,i]
                if d < 0.: rgb[j,k,i] = 0.
                if d > 1.: rgb[j,k,i] = 1.
    return rgb

if not os.path.exists('before.png'):
    imshow_100(two_p(before, samples, lines, bands2),
               samples, lines,
               'before.png')

if not os.path.exists('after.png'):
    imshow_100(two_p(after, samples, lines, bands2),
               samples, lines,
              'after.png')

def to_sk(dat):  # convert image format to sklearn expected format
    ret, lines, samples, bands = [], None, None, None
    try:
        lines, samples, bands = dat.shape
    except Exception:
        lines, samples = dat.shape
        bands = 1
    try:
        for i in range(lines):
            for j in range(samples):
                ret.append(dat[i, j, :].tolist())
    except Exception: # in case we don't have a bands coordinate (i.e., 1d case)
        for i in range(lines):
            for j in range(samples):
                ret.append(dat[i,j])
    return ret

'''
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsClassifier
>>> neigh = KNeighborsClassifier(n_neighbors=3)
>>> neigh.fit(X, y)
KNeighborsClassifier(...)
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[0.666... 0.333...]]
'''

'''===================================================='''
# don't forget to set the K through optimizing
'''===================================================='''
K = 1111

def from_sk(dat, samples, lines):
    # linear array to (lines, samples) np.array
    return np.reshape(dat, (lines, samples))

'''===================================================='''
# consistency: predict on the "before" input data..
'''===================================================='''
y = to_sk(d)
X = to_sk(before)
# K = int(math.floor(math.sqrt(npx)))

knn = KNeighborsClassifier(n_neighbors=K); print("fitting K=" + str(K))
knn.fit(X, y); print("predict..")
z = knn.predict(X)
p = knn.predict_proba(X)

rgb2 = apply_lookup(from_sk(z, samples, lines))

if not os.path.exists('ftl2.png'):
    imshow_100(rgb2, samples, lines, 'ftl2.png')

# calculate entropy
if p.shape[0] != npx:
    err('predict_proba returned unexpected shape')

pi2 = 1.584962500721156
p2 = np.zeros(npx)
n_class = p.shape[1]
for i in range(npx):
    for j in range(n_class): # https://en.wikipedia.org/wiki/Entropy_(information_theory)
        pp = p[i, j]
        pp += pi2
        pp /= (2. * pi2)
        if pp != 0.:
            p2[i] -= pp * math.log(pp) / math.log(2) 
        p2[i] = 1. - p2[i]
p2 = from_sk(p2, samples, lines)
print("min", np.min(p2))
print("max", np.max(p2))
if os.path.exists('ftl2p.png'):
    os.remove('ftl2p.png')

# would probability of class selected be better?
if not os.path.exists('ftl2p.png'):
    imshow_100(p2, samples, lines, 'ftl2p.png', True)
# print('p.shape', p.shape) # print(npx)

'''===================================================='''
# now actually do the forecast
'''===================================================='''
y = to_sk(d)
X = to_sk(after)
# K = int(math.floor(math.sqrt(npx)))

knn = KNeighborsClassifier(n_neighbors=K); print("fitting K=" + str(K))
knn.fit(X, y); print("predict..")
z = knn.predict(X)
p = knn.predict_proba(X)

rgb2 = apply_lookup(from_sk(z, samples, lines))

if not os.path.exists('ftl3.png'):
    imshow_100(rgb2, samples, lines, 'ftl3.png')

# calculate entropy
if p.shape[0] != npx:
    err('predict_proba returned unexpected shape')

pi2 = 1.584962500721156
p2 = np.zeros(npx)
n_class = p.shape[1]
for i in range(npx):
    for j in range(n_class): # https://en.wikipedia.org/wiki/Entropy_(information_theory)
        pp = p[i, j]
        pp += pi2
        pp /= (2. * pi2)
        if pp != 0.:
            p2[i] -= pp * math.log(pp) / math.log(2) 
        p2[i] = 1. - p2[i]
p2 = from_sk(p2, samples, lines)
print("min", np.min(p2))
print("max", np.max(p2))
if os.path.exists('ftl3p.png'):
    os.remove('ftl3p.png')

# would probability of class selected be better?
if not os.path.exists('ftl3p.png'):
    imshow_100(p2, samples, lines, 'ftl3p.png', True)
# print('p.shape', p.shape) # print(npx)



'''===================================================='''

# calculate accuracy
# calculate truthiness !!! 
# don't forget, perform spatial mode filter
# find out why the proba is between -pi/2 and pi/2!

'''
end: 
    increase number of bands
    find most consistent K 
    plot modelling time as function of K
    parallelize KNN ?????
    how does the proba work?????
    balanced representation of consistent samples...
    parallelisation..
look at adding slope, aspect, lat, long, time of year, age of fire, etc. other variables beyond just colour...
'''
