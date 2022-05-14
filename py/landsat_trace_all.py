'''get timestamps from landsat and sentinel2
and include those in them in result filenames

run this after:
    raster_accumulate.exe
and 
    raster_ts_dedup.exe

THIS VERSION IS FOR ATTRIBUTING MOST RECENT DETECTION

'''

import os
import time
import datetime
import matplotlib
import numpy as np
from osgeo import gdal
FOOT_H = 'footprint3.hdr'
import matplotlib.pyplot as plt
from misc import band_names, run, err, hdr_fn, sep, exist, utc_to_pst
args = ['', 'landsat.bin'] # bin_accumulate.bin_dedup.hdr']
bn = band_names(hdr_fn('landsat.bin')) # args[1]))

out_d = 'results_landsat_trace'
if not exist(out_d):
    os.mkdir(out_d)

ci = 1  # gdal uses band from 1
dataset = gdal.Open('landsat.bin', gdal.GA_ReadOnly)
nrow, ncol, latest = None, None, None
cumulative = None

for b in bn:
    w = b.strip().strip('.').strip(sep).split(sep) # print(w)
    base, TS, TYPE = '', '', ''
    if w[1][0] == 'L':
        TYPE = 'landsats'
        base = sep.join(w[:2])
        txt = base + sep + w[1] + '_MTL.txt'
        if not exist(txt):
            err('file not found')
        d, t = None, None
        lines = [x.strip() for x in open(txt).readlines()]
        for line in lines:
            w = [x.strip() for x in line.split('=')]
            if w[0] == 'DATE_ACQUIRED':
                d = w[1]
            if w[0] == 'SCENE_CENTER_TIME':
                t = w[1]
        yyyy, mm, dd = d.split('-')
        h, m, s = t.strip('"').split('.')[0].split(':')
        TS = '-'.join([yyyy, mm, dd, h, m, s])
    elif w[1][0] == 'S':
        # sentinel2
        TYPE = 'sentinel'
        d,t  = w[1].split('_')[2].split('T')
        yyyy, mm, dd = d[0:4], d[4:6], d[6:8]
        h, m, s = t[:2], t[2:4], t[4:6]
        TS = '-'.join([yyyy, mm, dd, h, m, s])
    else:
        err('unrecognized')
    x = [int(i) for i in TS.split('-')]
    L = utc_to_pst(x[0], x[1], x[2], x[3], x[4], x[5])
    if False:
        print(x, '-->', L) # '   **', L)
    zs = str(ci).zfill(3)
    fn = 'landsat.bin_accumulate.bin_dedup.bin_' + zs + '.bin'
    # print(fn)
    # hf = hdr_fn(fn)
    # print(fn, hf)
    
    pre = 'K61884' # 'results'
    of = pre + '_' + zs + '_' + TYPE + '_' + TS + '.bin'
    oh = pre + '_' + zs + '_' + TYPE + '_' + TS + '.hdr'
    ot = pre + '_' + zs + '_' + TYPE + '_' + TS + '.tif'
    out_file = out_d + sep + str(ci).zfill(3) + '_' + L + of + '.tif'
    cmd = ('find results_landsat/ -name "*' + TS + '*.tif"') # .read()
    match = os.popen(cmd).read().strip()
    print(out_file)

    print("+r", bn[ci-1])
    band = dataset.GetRasterBand(ci)
    arr = band.ReadAsArray()
    rows, cols = arr.shape
    if nrow is None:
        nrow, ncol = int(rows), int(cols)
        latest = np.zeros((nrow, ncol), np.float32) # int32)
    else:
        if nrow != rows or ncol != cols:
            err('unexpected number of rows/cols')
    
    yyyy, mm, dd, hour, minute, second = utc_to_pst(x[0], x[1], x[2], x[3], x[4], x[5], single_string=False)
    unix_t = time.mktime(datetime.datetime(yyyy, mm, dd, hour, minute, second).timetuple())
    # unix_t = int(unix_t)
    print(unix_t, type(unix_t))

    fire = (arr > 0)   # fire detection result (this step)
    if ci == 1:
        cumulative = np.zeros((nrow, ncol), np.int32)
        cumulative[fire] = 1  # start with the first dataset
        latest +=  float('nan')
    else:
        cumulative = np.logical_or(fire, cumulative)  # anywhere fire has been detected until now.
    latest[fire] = unix_t

    print('+w', out_file)
    if False:
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(out_file, cols, rows, 1, gdal.GDT_UInt32)
        outdata.SetGeoTransform(dataset.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(dataset.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(latest)
        outdata.GetRasterBand(1).SetNoDataValue(0) # 
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None
    else:
        if match != '':
            # cmap = matplotlib.cm.jet
            # cmap.set_bad('black', 0.)
            # plt.imshow(latest, interpolation='nearest', cmap=cmap)
            # plt.imshow(latest)
            # plt.tight_layout()
            # plt.savefig('time_' + str(ci).zfill(3) + '.png')

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            ax = plt.subplot()
            # cmap = matplotlib.cm.cool
            # cmap.set_bad('black', 0.)
            cmap = matplotlib.cm.get_cmap("rainbow").copy()
            cmap.set_bad('black', 0.)
            im = ax.imshow(latest, interpolation='nearest', cmap=cmap) # np.arange(100).reshape((10, 10)))
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            # plt.show()
            ax.set_xlabel('Time since last detect: ' + L)
            # plt.title('Time since last detection: ' + L) 
            plt.tight_layout()
            plt.savefig('time_' + str(ci).zfill(3) + '.png')
            plt.clf()
            plt.close()
        

    #if exist('results_landsat' + sep + ot):
    #   print(of)
    '''
    if not exist(of):
        run('cp ' + fn + ' ' + of)
    if not exist(oh):
        run('cp ' + hf + ' ' + oh)
    if not exist(ot):
        run('gdal_translate -of GTiff -ot Float32 ' + of + ' ' + ot)

    run('python3 ' + pd + 'envi_header_copy_mapinfo.py ' + FOOT_H + ' ' + oh)
    '''
    ci += 1
