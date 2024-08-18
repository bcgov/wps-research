'''20240203 assume first date is cloud-free / non-fire date

'''
from misc import err, args, exist, run, parfor
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import copy
import sys
import os

if len(args) < 2:
    err("arguments: [swir image sequence filename, multidate in order B12, B11, B9]")   
file_name = args[1]

print(file_name)
out_fn = file_name + '_bs.bin'
out_hn = file_name + '_bs.hdr'

def ignore_warnings(x, y, z):
    pass
gdal.PushErrorHandler(ignore_warnings)  # suppress warnings

d = gdal.Open(file_name)
n_bands = d.RasterCount

if n_bands % 3 != 0:
    err("band count must be multiple of 3")

data = []
dates = []
for i in range(1, n_bands + 1):
    band = d.GetRasterBand(i)
    s = band.GetDescription().split()
    try:
        s[0] = int(s[0])
    except:
        err("unexpected string in band name:", band.GetDescription())
    print(s)
    if i % 3 == 1:
        dates += [s[0]]
        if s[2] != 'B12':
           err('B12 expected in position ', i)
    if i % 3 == 2 and s[2] != 'B11':
        err('B11 expected in position ', i)
    if i % 3 == 0 and s[2] != 'B9':
        err('B9 expected in position ', i)
    data += [band.ReadAsArray().astype(np.float32)]

print(dates)
for i in range(1, len(dates)):
    if dates[i-1] >= dates[i]:
        err('dates should be non-duplicate and in order (increasing)')

driver = gdal.GetDriverByName('ENVI')
stack = driver.Create(out_fn,
                      d.RasterXSize,
                      d.RasterYSize,
                      n_bands,
                      gdal.GDT_Float32)

stack.SetProjection(d.GetProjection())
stack.SetGeoTransform(d.GetGeoTransform())

x = None
data2 = []
for i in range(0, int(n_bands/3)): # , n_bands + 1):
    a = data[i * 3] # a = data[i-1]
    b = data[i * 3 + 1]
    c = data[i * 3 + 2]
    x = np.logical_and(a > b, a > c)      
    x = np.multiply(x, (a - c)  - (data[0] - data[2]))# np.divide(a - c, data[1] - data[3]))

    for j in range(3):
        ix = i * 3 + j + 1  # band to write
        rb = stack.GetRasterBand(ix)
        b = np.zeros(d.RasterXSize * d.RasterYSize)
        b = copy.deepcopy(x) # np.multiply(x, data[i + j])
        if i > 0:
            b += data2[(i - 1) * 3 + j]
        data2 += [b]
        b = b.reshape((d.RasterYSize, d.RasterXSize))
        rb.WriteArray(b)
        band_name = d.GetRasterBand(ix).GetDescription()
        print(band_name)
        rb.SetDescription(band_name)
stack = None  # close dataset
run('envi_header_cleanup.py ' + out_hn)
