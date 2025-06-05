'''20230727 extract most recent available pixel (MRAP) from sentinel2 series.. assume tile based
(*) Difference from sentinel2_extract_swir.py: 
    can't process in parallel (By tile, at least)
NOTE: assumes sentinel2_extract_cloudfree.py has been run.

20240618 NOTE: need to make this fully incremental. I.e., initialize buffer (per tile) with last avaiable MRAP date

20250602: note: the program can't be run in parallel currently ( will try scoping the extract() function inside the run_mrap function)
20250602: need to do the above before ____ ( DONE ) 

20250602: NB need to run sentinel2_extract_cloudfree_swir_nir.py BEFORE running this script ( if we've updated with sync_recent.py )

'''
from misc import args, run, hdr_fn, err, parfor
from envi import envi_update_band_names
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import copy
import sys
import os


def run_mrap(gid):  # run MRAP on one tile
    my_bands, my_proj, my_geo, my_xsize, my_ysize, nbands, file_name  = {}, None, None, None, None, None, None

    
    def extract(file_name):
        nonlocal my_proj, my_geo, my_bands, my_xsize, my_ysize, nbands
        print("+r", file_name)
        d = gdal.Open(file_name)  # open the file brought in for this update step
    
        for i in range(1, d.RasterCount + 1):
            band = d.GetRasterBand(i)
            new_data = band.ReadAsArray().astype(np.float32)
    
            if i not in my_bands:  #initialize the band / first time
                my_bands[i] = new_data # new data for this band becomes the band
            else:
                nans, update = np.isnan(new_data), copy.deepcopy(my_bands[i])  # forgot copy before?
                update[~nans] = new_data[~nans]
                my_bands[i] = update
                # my_bands[i][~nans] = new_data[~nans]
    
        my_proj = d.GetProjection() if my_proj == None else my_proj
        my_geo = d.GetGeoTransform() if my_geo is None else my_geo
        if my_xsize is None:
            my_xsize, my_ysize, nbands = d.RasterXSize, d.RasterYSize, d.RasterCount
    
        d = None  # close input file brought in for this update step
    
        # write output file
        out_file_name, driver = file_name + '_MRAP.bin', gdal.GetDriverByName('ENVI')

        # skip files that already exist! 
        if not os.path.exists(out_file_name):
            print(out_file_name, my_xsize, my_ysize, nbands, gdal.GDT_Float32)
            stack_ds = driver.Create(out_file_name, my_xsize, my_ysize, nbands, gdal.GDT_Float32)
            stack_ds.SetProjection(my_proj)
            stack_ds.SetGeoTransform(my_geo)
    
            for i in range(1, nbands + 1):
                stack_ds.GetRasterBand(i).WriteArray(my_bands[i])
            stack_ds = None
    
            run('fh ' + out_file_name)  # fix envi header, then reproduce the band names
            envi_update_band_names(['envi_update_band_names.py',
                                    hdr_fn(file_name),
                                    hdr_fn(out_file_name)])
        else:
            print(out_file_name, 'exists [SKIP WRITE]')

    # look for all the dates in this tile's folder and sort them in aquisition time

    def get_filename_lines(search_cmd):
        lines = [x.strip() for x in os.popen(search_cmd).readlines()]
        lines = [x.split(os.path.sep)[-1].split('_') for x in lines]
        lines = [[x[2], x] for x in lines]
        lines.sort()
        return [[x[0], '_'.join(x[1])] for x in lines]
    
    lines = get_filename_lines("ls -1r L2_" + gid + os.path.sep + "S2*_cloudfree.bin")
    
    mrap_lines = get_filename_lines("ls -1r L2_" + gid + os.path.sep + "S2*_cloudfree.bin_MRAP.bin")
    # before this step, check for latest completed MRAP file and "seed" with that..if possible!

    for [mrap_date, line] in mrap_lines:
        gid = line.split("_")[5]
        extract_path = "L2_" +  gid + os.path.sep + line
        print('**' + mrap_date + " " + extract_path)
        if False:
            extract(extract_path)

    last_mrap_date = None
    if len(mrap_lines) > 0:
        last_mrap_date = mrap_lines[-1][0][:8]
        last_mrap_file = "L2_" + gid + os.path.sep + mrap_lines[-1][1]

        # load the last MRAP file here!  Seed from "most recent" in timestamp, MRAP file..
        print("last_mrap_date", last_mrap_date)
        # -----------------------------------------------------------------------------
        print("+r", last_mrap_file)
        d = gdal.Open(last_mrap_file)  # open the file brought in for this update step
    
        for i in range(1, d.RasterCount + 1):
            band = d.GetRasterBand(i)
            my_bands[i] = band.ReadAsArray().astype(np.float32)
    
        my_proj = d.GetProjection()
        my_geo = d.GetGeoTransform()
        my_xsize, my_ysize, nbands = d.RasterXSize, d.RasterYSize, d.RasterCount
        print(my_proj, my_geo, my_xsize, my_ysize, nbands)
        #-------------------------------------------------------------------------------

        
    
    for [line_date, line] in lines:
        gid = line.split("_")[5]
        extract_path = "L2_" +  gid + os.path.sep + line
                
        line_date_short = line_date[:8]

        if ( last_mrap_date is not None and line_date_short > last_mrap_date) or last_mrap_date is None:
            print('  ' + line_date_short + " " + extract_path)

            if True:
                extract(extract_path)

    # THIS PART SHOULD MARK ( NEEDING REFRESHING ) MRAP COMPOSITES ( MERGED PRODUCTS ) THAT NEED REFRESHING / BY DELETING THEM !!!!

    '''
    print("check sorting order")
    for [line_date, line] in lines:
        print("mrap " + line)
    '''

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
        def f(fn):
            run_mrap(fn)
        parfor(f, gids, 1) #  int(mp.cpu_count()))
    else:
        run_mrap(args[1])  # single tile mode: no mosaicing
