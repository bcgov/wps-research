# functions for reuse, for image processing, etc.
import os
import sys
import copy
import math
import struct
import numpy as np
import os.path as path
from osgeo import gdal
import multiprocessing as mp
try:
    import matplotlib.pyplot as plt
except:
    pass
args = sys.argv
sep = os.path.sep
abspath = os.path.abspath
def get_pd():
    return os.path.abspath(sep.join(abspath(__file__).split(sep)[:-1])) + sep  # python directory i.e. path to here
pd = get_pd()
def get_cd():
    return os.path.abspath(sep.join(abspath(__file__).split(sep)[:-2]) + sep + 'cpp') + sep
cd = get_cd()

def file_size(f): # get size of a file
    return os.stat(f).st_size

def me():  # my user name
    return os.popen('whoami').read().strip()

# print message and exit
def err(c):
    print('Error:', c); sys.exit(1)

def run(c, quit_on_nonzero=True):
    print('run(' + str(c) + ')')
    if type(c) == list:
        c = [str(i) for i in c]
        for i in range(1, len(c)):  # quote both sides of the argument if it's a parameter
            if c[i][0] != '"':
                c[i] = '"' + c[i]
            if c[i][-1] != '"':
                c[i] = c[i] + '"'
    c = ' '.join(c) if type(c) == list else c
    a = os.system(c)
    if a != 0 and quit_on_nonzero:
        err("command failed to run:\n\t" + c)
    return a

def exist(f):
    return os.path.exists(f)

def exists(f):
    return os.path.exists(f)

def hdr_fn(bin_fn):  # return filename for hdr file, given binfile name
    hfn = bin_fn[:-4] + '.hdr'
    if not exist(hfn):
        hfn2 = bin_fn + '.hdr'
        if not exist(hfn2):
            err("header not found at:", hfn, "or:", hfn2)
        return hfn2
    return hfn

def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    print('+r', hdr)
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

def get_band_names_line_idx(hdr):  # input: file data
    samples, lines, bands = read_hdr(hdr)
    lines = [x.strip() for x in open(hdr).readlines()]
    #  Output: line idx of lines with band names data in them!
    for i in range(len(lines)):
        if len(lines[i].split("band names =")) > 1:
            # print("lines[i]", lines[i])
            return list(range(i, i + int(bands)))
    return []

def get_map_info_lines_idx(hdr):
    # given filename, get indices of map info lines (should be two of those)
    result = [None, None]
    lines = [x.strip() for x in open(hdr).readlines()]
    for i in range(len(lines)):
        w = [x.strip() for x in lines[i].split('=')]
        if w[0] == 'map info':
            result[0] = i
        if w[0] == 'coordinate system string':
            result[1] = i
    return result

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
    hdr = hdr_fn(fn) # read header and print parameters
    samples, lines, bands = read_hdr(hdr)
    samples, lines, bands = int(samples), int(lines), int(bands)
    print("\tsamples", samples, "lines", lines, "bands", bands)
    data = read_float(fn)
    return samples, lines, bands, data

def write_binary(np_ndarray, fn): # write a numpy array to ENVI format type 4
    of = wopen(fn)
    np_ndarray = np_ndarray.astype(np.float32)
    np_ndarray.tofile(of, '', '<f4')
    of.close()

def write_hdr(hfn, samples, lines, bands, band_names = None):
    print('+w', hfn)
    lines = ['ENVI',
             'samples = ' + str(samples),
             'lines = ' + str(lines),
             'bands = ' + str(bands),
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 4',
             'interleave = bsq',
             'byte order = 0']
    if band_names is not None:
        lines += ['']
        bs1 = "band names = {" + band_names[0] + ','
        lines += [bs1]
        for i in range(1, len(band_names)):
            bsi = (band_names[i] + ',')
            lines += [bsi]
        lines[-1] = lines[-1][:-1] + '}'
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
        rgb[:, :, i] = data[:, :, band_select[i]]  # pull a channel
        values = rgb[:, :, i].reshape(samples * lines).tolist()  # slice, reshape
        values.sort()  # sort

        if values[-1] < values[0]:  # sanity
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


def parfor(my_function,  # function to run in parallel
           my_inputs,  # inputs evaluated by worker pool
           n_thread=mp.cpu_count()): # cpu threads to use
    
    if n_thread == 1:  # don't use multiprocessing for 1-thread
        return [my_function(my_inputs[i])
                for i in range(len(my_inputs))]
    else:
        n_thread = (mp.cpu_count() if n_thread is None
                    else n_thread)
        return mp.Pool(n_thread).map(my_function, my_inputs)


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

def add_commas(number):
    return "{:,}".format(number)

def discrete_cmap(N, base_cmap=None): # https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

'''method for reading a csv file. Returns a list of fields, and a list of lists..
indexed by field index e.g. data[0] is columnar representation of the first field'''
def read_csv(f):
    import csv
    data, i = [], 0
    reader = csv.reader(open(f),
                        delimiter=',',
                        quotechar='"')
    for row in reader:
        row = [x.strip() for x in row]
        if i == 0:
            N = len(row)
            I = range(N)
            fields, data = row, [[] for j in I]
        else:
            for j in I:
                data[j].append(row[j])
        i += 1
        if i % 1000 == 0:
            print(i)
    fields = [x.strip() for x in fields]
    return fields, data

'''returns a list of colors for use with matplotlib'''
def colors():
    import matplotlib
    mcolors = matplotlib.colors
    cols = list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys())
    return cols[0:7] + cols[8:]  # skipped one that looked indistinct

'''returns a list of marker patterns for use with matplotlib'''
def markers():
    return [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s",
            "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1,
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def xy_to_pix_lin(fn, x, y, nb):  # raster fn, lat/lon, number of bands (assume we read it already)
    cmd = ["gdallocationinfo",
           fn, # input image
           '-wgs84', # specify lat long input
           str(x), # lat
           str(y)] # long
    cmd = ' '.join(cmd)
    print(cmd)
    lines = [x.strip() for x in os.popen(cmd).readlines()]
    count = 0
    if len(lines) >= 2 * (1 + nb):
        w = lines[1].split()
        if w[0] != "Location:":
            err("unexpected field")
        pix_i, lin_i = w[1].strip('(').strip(')').split(',')
        if pix_i[-1] != 'P' or lin_i[-1] != 'L':
            err('unexpected data')

        pix_i, lin_i = int(pix_i[:-1]), int(lin_i[:-1])
        print(str(pix_i) + 'P ' + str( lin_i) + 'L')
        count += 1
        data = []
        for j in range(0, nb): # for each band
            bn = lines[2 * (1 + j)].strip(":").strip().split()
            if int(bn[1]) != j + 1:
                err("expected: Band: " + str(j + 1) + "; found: " + lines[2 * (1 + j)])
            value = float(lines[3 + (2*j)].split()[1].strip())
            data.append(value)
        print(data)

        row, col = lin_i, pix_i
        return row, col, data  # return the goods!
    else:
        err("misc.py: unexpected output from gdallocationinfo: number of lines: " + str(len(lines)))

def pix_lin_to_xy(fn, col, row):
    err('fix this with code from raster_pixels_location.py')

def utc_to_pst(YYYY, MM, DD, hh, mm, ss, single_string=True):
    from datetime import datetime
    from pytz import timezone
    PST = timezone('US/Pacific')

    d = datetime(YYYY, MM, DD, hh, mm, ss)
    x = PST.localize(d)
    time_diff = x.tzinfo.utcoffset(x)
    local_time = d + time_diff

    L = local_time
    if(single_string):
        return ''.join([str(L.year).zfill(4),
                        str(L.month).zfill(2),
                        str(L.day).zfill(2),
                        str(L.hour).zfill(2),
                        str(L.minute).zfill(2),
                        str(L.second).zfill(2)])
    else:
        return [L.year, L.month, L.day, L.hour, L.minute, L.second]

def write_band_gtiff(output_data,  # 2d numpy array
                     ref_dataset, # gdal dataset to copy map info from
                     output_fn,  # output filename to write
                     gdal_datatype=gdal.GDT_Float32):  # output datatype
    print('+w', output_fn)
    from osgeo import gdal
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = output_data.shape  # assumed one-band
    outd = driver.Create(output_fn, cols, rows, 1, gdal_datatype)
    outd.SetGeoTransform(ref_dataset.GetGeoTransform())##sets same geotransform as input
    outd.SetProjection(ref_dataset.GetProjection())##sets same projection as input
    outd.GetRasterBand(1).WriteArray(output_data)
    outd.GetRasterBand(1).SetNoDataValue(0) #
    outd.FlushCache() ##saves to disk!!
    outd = None
    band=None
    ds=None
