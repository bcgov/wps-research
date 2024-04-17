''' 20231230 view.py e.g. Usage:
      python3 view.py G80223_20230513.bin
      python3 view.py G90292_20230514.bin
'''
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def exist(f): return os.path.exists(f)

def exists(f): return os.path.exists(f)

def hdr_fn(bin_fn):  # return filename for hdr file, given binfile name
    hfn = bin_fn[:-4] + '.hdr'
    if not exist(hfn):
        hfn2 = bin_fn + '.hdr'
        if not exist(hfn2):
            err("header not found at:" + hfn + " or: " + hfn2)
        return hfn2
    return hfn

def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    # print('+r', hdr)
    for line in open(hdr).readlines():
        line = line.strip()
        words = line.split('=')
        if len(words) == 2:
            f, g = words[0].strip(), words[1].strip()
            if f == 'samples': samples = g
            if f == 'lines': lines = g
            if f == 'bands': bands = g
    return samples, lines, bands

# use numpy to read a floating-point data file (4 bytes per float, byte order 0)
def read_float(fn):
    print("+r", fn)
    return np.fromfile(fn, dtype = np.float32) # "float32") # '<f4')

def wopen(fn):
    f = open(fn, "wb")
    if not f: err("failed to open file for writing: " + fn)
    print("+w", fn)
    return f

def read_binary(fn):
    hdr = hdr_fn(fn) # read header and print parameters
    samples, lines, bands = read_hdr(hdr)
    samples, lines, bands = int(samples), int(lines), int(bands)
    print("\tsamples", samples, "lines", lines, "bands", bands)
    data = read_float(fn)
    return samples, lines, bands, data

header_file = hdr_fn(sys.argv[1])
bin_file = sys.argv[1]
samples, lines, bands, dataset = read_binary(bin_file)
print(samples, lines, bands, dataset)

def scale(X):
    # default: scale a band to [0, 1]  and then clip
    mymin = np.nanmin(X) # np.nanmin(X))
    mymax = np.nanmax(X) # np.nanmax(X))
    X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    X[X < 0.] = 0.  # clip
    X[X > 1.] = 1.

    # use histogram trimming / turn it off to see what this step does!
    if False: # True:
        values = X.ravel().tolist()
        values.sort()
        n_pct = 1. # percent for stretch value
        frac = n_pct / 100.
        lower = int(math.floor(float(len(values))*frac))
        upper = int(math.floor(float(len(values))*(1. - frac)))
        mymin, mymax = values[lower], values[upper]
        X = (X-mymin) / (mymax - mymin)  # perform the linear transformation

    return X

# Check if the dataset was successfully opened
if dataset is None:
    print(f"Failed to open the BIN file: {bin_file_path}")
else:
    # image dimensions
    width = int(samples) # dataset.RasterXSize)
    height = int(lines) # dataset.RasterYSize)
    rgb = np.zeros((height, width, 3))
    n_pix = width * height

    # Read the data from the raster bands (assuming RGB bands are 1, 2, and 3)
    rgb[:, :, 0] = scale(dataset[n_pix * 2: n_pix * 3].reshape((height, width))) # .GetRasterBand(1).ReadAsArray().reshape((height, width)))
    rgb[:, :, 1] = scale(dataset[n_pix * 1: n_pix * 2].reshape((height, width))) # .GetRasterBand(2).ReadAsArray().reshape((height, width)))
    rgb[:, :, 2] = scale(dataset[n_pix * 0: n_pix * 1].reshape((height, width))) # .GetRasterBand(3).ReadAsArray().reshape((height, width)))
    ''' A data cube indexed by row, column and band index (band index is in 1,2,3 rather: 0,1,2 from 0) 

    0,1,2 are not actually red, green blue. They are B12, B11, B9 from Sentinel-2:
        B12: 2190 nm
        B11: 1610 nm
        B9: 940 nm 
    which are in the short-wave infrared (SWIR)

    We chose the false-color encoding (R,G,B) = (B12, B11, B9) because fire looks orange/red/brown ish
    band names = {
        20230513 60m: B9 945nm,
        20230513 20m: B11 1610nm,
        20230513 20m: B12 2190nm}

    Because the bands are stored in order of increasing wavelength, we used the indices in reverse order in the lines:
        rgb[:, :, 0] = scale(dataset[n_pix * 2: n_pix * 3].reshape((height, width))) ...
        etc. 
        (compare with view.py) 
    to get the chosen false-color encoding (in which fire is "red/orange")
    ''' 
   
    # Close the dataset
    dataset = None

    # Plot the RGB image using Matplotlib
    plt.figure()
    plt.imshow(rgb)
    plt.title(sys.argv[1] + " with encoding R,G,B =(B12, B11, B9)")
    plt.axis('off')  # Turn off axis labels
    plt.tight_layout()
    plt.show()


