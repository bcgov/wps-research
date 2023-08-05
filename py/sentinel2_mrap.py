'''20230727 extract most recent available pixel (MRAP) from sentinel2 series.. assume tile based

Difference from sentinel2_extract_swir.py: 
can't process in parallel (By tile, at least)

NOTE: assumes sentinel2_extract_cloudfree.py has been run.
'''
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
from misc import args, run
import numpy as np
import sys
import os

my_bands = {}
my_proj, my_geo, my_xsize, my_ysize, nbands  = None, None, None, None, None

def extract(file_name):
    global my_proj, my_geo, my_bands, my_xsize, my_ysize, nbands
    print("+r", file_name)
    d = gdal.Open(file_name)
    
    for i in range(1, d.RasterCount + 1):
        band = d.GetRasterBand(i)
        array_data = band.ReadAsArray().astype(np.float32)
        
        if i not in my_bands:
            my_bands[i] = array_data

        else:
            nans = np.isnan(array_data)
            update = my_bands[i]
            update[~nans] = array_data[~nans]
            my_bands[i] = update
   
    if my_proj is None:
        my_proj = d.GetProjection()
    if my_geo is None:
        my_geo = d.GetGeoTransform()
    if my_xsize is None:
        my_xsize, my_ysize = d.RasterXSize, d.RasterYSize    
        nbands = d.RasterCount 

if __name__ == "__main__":
    file_name = None
    if len(args) != 2:
        err("python3 sentinel2_mrap.py [sentinel-2 gid]")
    else:
        gid = args[1]

        # sort the dates in time...
        lines = [x.strip() for x in os.popen("ls -1r S*.bin")]
        lines2 = [x.split('_') for x in lines]
        lines3 = [[x[2], x] for x in lines2]
        lines3.sort()

        lines = ['_'.join(x[1]) for x in lines3]

        for line in lines:
            extract(line)
        # now save the output


    driver = gdal.GetDriverByName('ENVI')

    print("mrap_" + args[1] + ".bin",
                             my_xsize,
                             my_ysize,
                             nbands,
                             gdal.GDT_Float32)

    stack_ds = driver.Create("mrap_" + args[1] + ".bin",
                             my_xsize,
                             my_ysize,
                             nbands,
                             gdal.GDT_Float32)

    stack_ds.SetProjection(my_proj)
    stack_ds.SetGeoTransform(my_geo)

    for i in range(1, nbands + 1):
        rb = stack_ds.GetRasterBand(i)
        rb.WriteArray(my_bands[i])

    stack_ds = None

    run('fh mrap_' + args[1] + '.hdr')
