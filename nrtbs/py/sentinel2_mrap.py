'''20230727 extract most recent available pixel (MRAP) from sentinel2 series.. assume tile based
(*) Difference from sentinel2_extract_swir.py: 
    can't process in parallel (By tile, at least)
NOTE: assumes sentinel2_extract_cloudfree.py has been run.

20240618 NOTE: need to make this fully incremental. I.e., initialize buffer (per tile) with last avaiable MRAP date'''
from envi import envi_update_band_names
from envi import envi_header_cleanup
from misc import args, run, hdr_fn, err, parfor, exist
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import copy
import sys
import os
import shutil

my_bands_gid, my_proj_gid, my_geo_gid, my_xsize_gid, my_ysize_gid, nbands_gid = {}, {}, {}, {}, {}, {} 

def extract(file_name, gid):
    stack_fn = '.'.join(file_name.split('.')[:-1]) + '.bin_MRAP.bin' # output stack filename

    print(stack_fn)
    if exist(stack_fn):
        print("Exists:", stack_fn, "skipping..")
        return

    global my_bands_gid, my_proj_gid, my_geo_gid, my_xsize_gid, my_ysize_gid, nbands_gid
    print("+r", file_name)
    d = gdal.Open(file_name)  # open the file brought in for this update step
    
    for i in range(1, d.RasterCount + 1):
        band = d.GetRasterBand(i)
        new_data = band.ReadAsArray().astype(np.float32)
        
        if gid not in my_bands_gid:
            my_bands_gid[gid] = {}
            my_proj_gid[gid] = None
            my_geo_gid[gid] = None
            my_xsize_gid[gid] = None
            my_ysize_gid[gid] = None

        if i not in my_bands_gid[gid]:  #initialize the band / first time
            my_bands_gid[gid][i] = new_data # new data for this band becomes the band
        else:
            nans, update = np.isnan(new_data), copy.deepcopy(my_bands_gid[gid][i])  # forgot copy before?
            update[~nans] = new_data[~nans]
            my_bands_gid[gid][i] = update
            # my_bands[i][~nans] = new_data[~nans]

    my_proj_gid[gid] = d.GetProjection() if my_proj_gid[gid] == None else my_proj_gid[gid]
    my_geo_gid[gid] = d.GetGeoTransform() if my_geo_gid[gid] is None else my_geo_gid[gid]
    if my_xsize_gid[gid] is None:
        my_xsize_gid[gid], my_ysize_gid[gid], nbands_gid[gid] = d.RasterXSize, d.RasterYSize, d.RasterCount 

    d = None  # close input file brought in for this update step

    # write output file
    out_file_name, driver = file_name + '_MRAP.bin', gdal.GetDriverByName('ENVI')
    print(out_file_name, my_xsize_gid[gid], my_ysize_gid[gid], nbands_gid[gid], gdal.GDT_Float32)

    stack_ds = driver.Create(out_file_name,
                             my_xsize_gid[gid],
                             my_ysize_gid[gid],
                             nbands_gid[gid],
                             gdal.GDT_Float32)
    
    stack_ds.SetProjection(my_proj_gid[gid])
    stack_ds.SetGeoTransform(my_geo_gid[gid])

    for i in range(1, nbands_gid[gid] + 1):
        stack_ds.GetRasterBand(i).WriteArray(my_bands_gid[gid][i])
    stack_ds = None

    hdr_file = f'{file_name[:-4]}.hdr'
    out_hdr_name = f'{out_file_name[:-4]}.hdr'
    
    # Copy the file..anything lost by doing this?
    shutil.copy2(hdr_file, out_hdr_name)

    # print(f'File name: {file_name}!!!!!!!!!!!!!!')
    # print(f'Out file: {out_file_name}!!!!!!!!!!!!!!!!!')
    # #envi_header_cleanup(['',out_file_name])
    # envi_update_band_names(['', 
    #                         hdr_fn(file_name),
    #                         hdr_fn(out_file_name)])

def run_mrap(gid):  # run MRAP on one tile
    if True:
        # look for all the dates in this tile's folder and sort on aquisition time
        
        def get_filename_lines(search_cmd):
            lines = [x.strip() for x in os.popen(search_cmd).readlines()]
            lines = [x.split(os.path.sep)[-1].split('_') for x in lines]
            lines = [[x[2], x] for x in lines]
            lines.sort()
            return ['_'.join(x[1]) for x in lines]

        lines = get_filename_lines("ls -1r L2_" + gid + os.path.sep + "S2*_cloudfree.bin")

        for line in lines:
            gid = line.split("_")[5]
            extract("L2_" +  gid + os.path.sep + line, gid)

        print("check sorting order")
        for line in lines:
            print("mrap " + line)


if __name__ == "__main__":
    if len(args) < 2:
        gids = []
        dirs = [x.strip() for x in os.popen('ls -1d L2_*').readlines()]
        for d in dirs:
            print(d)
            w = d.split('_')
            if len(w) != 2:
                err('unexpected folder name')

            gid = w[1]
            gids += [gid]
            # should run on this frame here. 
            # but check for cloud-free data first
        # err("python3 sentinel2_mrap.py [sentinel-2 gid] # [optional: yyyymmdd 'maxdate' parameter] ")
        
        # run mrap on one tile
        def f(gid):
            run_mrap(gid)

        # run mrap in parallel (by tile)
        parfor(f, gids, int(mp.cpu_count()))
    else:
        gids = []
        dirs = [x.strip() for x in os.popen(f'ls -1d {args[1]}').readlines()]
        for d in dirs:
            print(d)
            w = d.split('_')
            if len(w) != 2:
                err('unexpected folder name')

            gid = w[1]
            gids += [gid]
            # should run on this frame here. 
            # but check for cloud-free data first
        # err("python3 sentinel2_mrap.py [sentinel-2 gid] # [optional: yyyymmdd 'maxdate' parameter] ")
        
        # run mrap on one tile
        def f(gid):
            run_mrap(gid)

        # run mrap in parallel (by tile)
        parfor(f, gids, int(mp.cpu_count()))
