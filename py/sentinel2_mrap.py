'''20230727 extract most recent available pixel (MRAP) from sentinel2 series.. assume tile based

Difference from sentinel2_extract_swir.py: 
can't process in parallel (By tile, at least)

NOTE: assumes sentinel2_extract_cloudfree.py has been run.'''
from envi import envi_update_band_names
from envi import envi_header_cleanup
from misc import args, run, hdr_fn
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import copy
import sys
import os

my_bands, my_proj, my_geo, my_xsize, my_ysize, nbands, file_name  = {}, None, None, None, None, None, None

def extract(file_name):
    global my_proj, my_geo, my_bands, my_xsize, my_ysize, nbands
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
   
    my_proj = d.GetProjection() if my_proj == None else my_proj
    my_geo = d.GetGeoTransform() if my_geo is None else my_geo
    if my_xsize is None:
        my_xsize, my_ysize, nbands = d.RasterXSize, d.RasterYSize, d.RasterCount 

    d = None  # close input file brought in for this update step

    # write output file
    out_file_name, driver = file_name + '_MRAP.bin', gdal.GetDriverByName('ENVI')
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


if __name__ == "__main__":
    if len(args) < 2:
        err("python3 sentinel2_mrap.py [sentinel-2 gid] # [optional: yyyymmdd 'maxdate' parameter] ")
    else:
        gid = args[1]
        lines = [x.strip() for x in os.popen("ls -1r S2*.bin").readlines()]  # sort dates in time
        lines = [x.split('_') for x in lines]
        lines = [[x[2], x] for x in lines]
        lines.sort()
        lines = ['_'.join(x[1]) for x in lines]
        
        for line in lines:
            extract(line)
