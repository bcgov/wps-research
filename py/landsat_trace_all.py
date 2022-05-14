'''get timestamps from landsat and sentinel2
and include those in them in result filenames

landsat_get_time.py instructions said:
run this after:
    raster_accumulate.exe
and 
    raster_ts_dedup.exe

THIS VERSION IS FOR ATTRIBUTING MOST RECENT DETECTION and deprecates landsat_get_time.py
Therefore, this version does not need raster_accumulate, not raster_ts_dedup'''
PIXEL_CHANGE_THRES = 333 # magic number !!!!!!!!!!!

import os
import sys
import time
import datetime
import matplotlib
import numpy as np
from osgeo import gdal
import matplotlib.ticker as ticker
FOOT_H = 'footprint3.hdr'
import matplotlib.pyplot as plt
from misc import read_binary, write_hdr, band_names, run, err, hdr_fn, sep, exist, utc_to_pst, write_band_gtiff, write_binary
from mpl_toolkits.axes_grid1 import make_axes_locatable
args = ['', 'landsat.bin'] # bin_accumulate.bin_dedup.hdr']
bn = band_names(hdr_fn('landsat.bin')) # args[1]))

out_d = 'results_landsat_trace'
if not exist(out_d):
    os.mkdir(out_d)

ci = 1  # band index, gdal uses band from 1
out_i = 1  # index of output file

dataset = gdal.Open('landsat.bin', gdal.GA_ReadOnly)
nrow, ncol, latest = None, None, None
cumulative = None
target_row, target_col = None, None


# calculate the closest detected point to centroid of the same points
def detection_centroid(detect):
    nrow, ncol = detect.shape
    target_row, target_col = None, None
    if True:
        # calculate centroid of detection
        indices = np.indices((nrow, ncol))
        row_ix = indices[0]
        col_ix = indices[1]

        target_n = np.sum(detect)
        target_row = np.sum(np.multiply(row_ix, detect))
        target_col = np.sum(np.multiply(col_ix, detect))
        target_row /= target_n
        target_col /= target_n

        target_row = int(target_row + .5)
        target_col = int(target_col + .5)
        target_row = max(target_row, 0)
        target_col = max(target_col, 0)
        target_row = min(target_row, nrow - 1)
        target_col = min(target_col, ncol - 1)
        print("target row", target_row)
        print("target_col", target_col)
        
    # now get nearest element that's selected
    fire_row, fire_col = np.where(detect)
    min_row, min_col = fire_row[0], fire_col[0]

    fx = fire_row[0] - target_row
    fy = fire_col[0] - target_col
    min_d = fx * fx + fy * fy
    for i in range(len(fire_row)):
        fx = fire_row[i] - target_row
        fy = fire_col[i] - target_col
        d = fx * fx + fy * fy

        if d < min_d:
            min_row = fire_row[i]
            min_col = fire_col[i]
            min_d = d
    target_row = min_row
    target_col = min_col
    print("target row", target_row)
    print("target_col", target_col)
    if not detect[target_row, target_col]:
        err("detection centroid not detected")
    return target_row, target_col

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
    
    zs = str(out_i).zfill(3)
    fn = 'landsat.bin_accumulate.bin_dedup.bin_' + zs + '.bin'
    # print(fn)
    # hf = hdr_fn(fn)
    # print(fn, hf)
    
    pre = out_d + sep + 'K61884' # 'results'
    of = pre + '_' + zs + '_' + TYPE + '_' + TS + '.bin'
    oh = pre + '_' + zs + '_' + TYPE + '_' + TS + '.hdr'
    ot = pre + '_' + zs + '_' + TYPE + '_' + TS + '.tif'
    of2 = pre + '_' + zs + '_' + TYPE + '_' + TS + '_unixtime.tif' #  out_d + sep + str(ci).zfill(3) + '_' + L + of + '.tif'
    op = pre + '_' + zs + '_' + TYPE + '_' + TS + '.png'
    of2e=pre + '_' + zs + '_' + TYPE + '_' + TS + '_unixtime.bin'
    # print(out_file)

    # print("+r", bn[ci-1])
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
    # print(unix_t, type(unix_t))

    fire = (arr > 0)   # fire detection result (this step)

    if ci == 1:
        cumulative = np.zeros((nrow, ncol), np.int32)

    if ci == 1:
        cumulative[fire] = 1  # start with the first dataset
        latest +=  float('nan')
    else:
        cumulative = np.logical_or(fire, cumulative)  # anywhere fire has been detected until now.
    # latest[fire] = unix_t

    # find cumulative extent
    target_row, target_col = detection_centroid(cumulative)
    # run flood fill
    run("rm cum.bin*")
    write_binary(cumulative.astype(np.float32), "cum.bin")
    write_hdr("cum.hdr", str(ncol), str(nrow), str(1))
    run('ulimit -s 1000000; flood.exe cum.bin')
    # now, run class linking (on nearest point to centroid that is a detection, as target)
    cmd= ('class_link.exe cum.bin_flood4.bin 111 ' + str(target_row) + ' ' + str(target_col))
    run(cmd)
    print(cmd)
    p_re = out_d + sep + str(ci).zfill(4) + '_' + str(out_i).zfill(4)
    run('cp cum.bin_flood4.bin ' + p_re + '_flood4.bin')
    run('cp cum.bin_flood4.hdr ' + p_re + '_flood4.hdr')
    run('cp cum.bin_flood4.bin_link_target.bin ' + p_re + '_link_target.bin')
    run('cp cum.bin_flood4.bin_link_target.hdr ' + p_re + '_link_target.hdr')
    [f_samp, f_lines, f_bands, f_d] = read_binary('cum.bin_flood4.bin_link_target.bin')
    f_d = f_d.reshape(rows, cols)
    # fire = f_d > 0  # now we revised the fire detection result, to include only this connected component
    cumulative = f_d > 0

    fire = np.logical_and(cumulative, fire)

    changed = np.logical_and(fire, np.logical_not(cumulative)) # detected this step and not detected before 
    n_changed = np.count_nonzero(changed)

    latest[fire] = unix_t

    if n_changed > PIXEL_CHANGE_THRES:
        print("\n\nn_changed", n_changed, "**********************")
        if True:

            # BEGIN WRITE PLOT**************************************************
            plt.figure(figsize=(16, 12))
            ax = plt.subplot()
            cmap = matplotlib.cm.get_cmap("rainbow").copy()
            cmap.set_bad('black', 0.)
            im = ax.imshow(latest, interpolation='nearest', cmap=cmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            def fmt(x, pos):
                a, b = '{:.5e}'.format(x).split('e')
                b = int(b)
                return r'${} \times 10^{{{}}}$'.format(a, b)

            plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(fmt))
            ax.set_xlabel('Seconds (since 00:00:00 PST / Jan 1 1970) of last detection circa: ' + L + ' PST')
            plt.tight_layout()
            print('+w', op)
            plt.savefig(op) # output png file 
            plt.clf()
            plt.close()
            # END WRITE PLOT **************************************************
    
            write_band_gtiff(cumulative, dataset, ot)  # gdal_datatype=gdal.GDT_Float32):
            write_band_gtiff(latest, dataset, of2) # , gdal_datatype=gdal.GDT_UInt32)
            run('gdal_translate -of ENVI -ot Float32 ' + ot + ' ' + of) # + ' &')
            run('gdal_translate -of ENVI -ot Float32 ' + of2 + ' ' + of2e) # + ' &')
        out_i += 1
    ci += 1


# the flood-fill/connect algorithm... if the updates don't connect to previous ones, might need to run this step at the end instead of at every step! 
