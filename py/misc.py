# functions for reuse, for image processing, etc.
import os
import sys
import copy
import math
import struct
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

args = sys.argv

# print message and exit
def err(c):
    print('Error: ' + c)
    sys.exit(1)


def run(c):
    print('run("' + str(c) + '")')
    a = os.system(c)
    if a != 0:
        err("command failed to run:\n\t" + c)
    return a

def exist(f):
    return os.path.exists(f)

def exists(f):
    return exist(f)

def hdr_fn(bin_fn):
    # return filename for header file, given name for bin file
    hfn = bin_fn[:-4] + '.hdr'
    if not exist(hfn):
        hfn2 = bin_fn + '.hdr'
        if not exist(hfn2):
            err("didn't find header file at: " + hfn + " or: " + hfn2)
        return hfn2
    return hfn

def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    for line in open(hdr).readlines():
        line = line.strip()
        words = line.split('=')
        if len(words) == 2:
            f, g = words[0].strip(), words[1].strip()
            if f == 'samples':
                samples = g
            if f == 'lines':
                lines = g
            if f == 'bands':
                bands = g
    return samples, lines, bands

def band_names(hdr): # read band names from header file
    names, lines = [], open(hdr).readlines()
    for i in range(0, len(lines)):
        line = lines[i].strip()
        x = line.split(' = ')
        if len(x) > 1:
            if x[0] == 'band names':
                names.append(x[1].split('{')[1].strip(','))
                for j in range(i + 1, len(lines)):
                    line = lines[j].strip()
                    names.append(line.strip(',').strip('}'))
                return names
    return []

# require a filename, or list of filenames, to exist
def assert_exists(fn):
    try:
        if type(fn) != str:
            iterator = iter(fn)
            for f in fn:
                assert_exists(f)
            return
    except:
        # not iterable
        pass

    if not exists(fn):
        err("couldn't find required file: " + fn)

# use numpy to read a floating-point data file (4 bytes per float, byte order 0)
def read_float(fn):
    print("+r", fn)
    return np.fromfile(fn, dtype = np.float32) # "float32") # '<f4')

def wopen(fn):
    f = open(fn, "wb")
    if not f:
        err("failed to open file for writing: " + fn)
    print("+w", fn)
    return f

def read_binary(fn):
    hdr = hdr_fn(fn)

    # read header and print parameters
    samples, lines, bands = read_hdr(hdr)
    samples, lines, bands = int(samples), int(lines), int(bands)
    print("\tsamples", samples, "lines", lines, "bands", bands)

    data = read_float(fn)
    return samples, lines, bands, data


def write_binary(np_ndarray, fn):
    of = wopen(fn)
    np_ndarray.tofile(of, '', '<f4')
    of.close()


def write_hdr(hfn, samples, lines, bands):
    lines = ['ENVI',
             'samples = ' + str(samples),
             'lines = ' + str(lines),
             'bands = ' + str(bands),
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 4',
             'interleave = bsq',
             'byte order = 0']
    open(hfn, 'wb').write('\n'.join(lines).encode())

# counts of each data instance
def hist(data):
    count = {}
    for d in data:
        count[d] = 1 if d not in count else count[d] + 1
    return count


# two-percent linear, histogram stretch. N.b. this impl. does not preserve colour ratios
def twop_str(data, band_select = [3, 2, 1]):
    samples, lines, bands = data.shape
    rgb = np.zeros((samples, lines, 3))

    for i in range(0, 3):
        # extract a channel
        rgb[:, :, i] = data[:, :, band_select[i]]
        
        # slice, reshape and sort
        values = rgb[:, :, i].reshape(samples * lines).tolist() 
        values.sort()
        
        # sanity check
        if values[-1] < values[0]:
            err("failed to sort")

        # so-called "2% linear stretch
        npx = len(values) # number of pixels
        rgb_mn = values[int(math.floor(float(npx) * 0.02))]
        rgb_mx = values[int(math.floor(float(npx) * 0.98))]
        rgb[:, :, i] -= rgb_mn
        rng = rgb_mx - rgb_mn
        mask = rgb[:, :, i] < 0
        rgb[mask] = 0.
        if rng > 0.:
            rgb[:, :, i] /= rng
    return rgb

def parfor(my_function, my_inputs):
    # evaluate a function in parallel, and collect the results
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_function, my_inputs)
    return(result)


def bsq_to_scikit(ncol, nrow, nband, d):
    # convert image to a format expected by sgd / scikit learn

    npx = nrow * ncol # number of pixels

    # convert the image data to a numpy array of format expected by sgd
    img_np = np.zeros((npx, nband))
    for i in range(0, nrow):
        ii = i * ncol
        for j in range(0, ncol):
            for k in range(0, nband):
                # don't mess up the indexing
                img_np[ii + j, k] = d[(k * npx) + ii + j]
    return(img_np)
