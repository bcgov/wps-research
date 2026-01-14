'''
polygon.py
'''

from raster import Raster
import numpy as np


def split_in_out(
        raster_filename,
        polygon_filename        
):
    
    '''
    Extract data from inside and outside of the polygon.

    
    Parameters
    ----------
    raster_filename: the main data where polygon was drawn onto.

    polygon_filename: file name to the polygon, .bin format (use rasterization from shape file).
        polygon needs to be of 1 band.

    
    Returns
    -------
    4 things:

    2 "all" data read from the files

    2 matrices of inside and outside.


    Notes
    -----
    Use rasterize_onto.py to have similar matrix shape for raster and polygon.
    '''

    from exceptions.matrix_exception import Shape_Mismatched_Error
    

    raster = Raster(raster_filename)
    polygon = Raster(polygon_filename)

    raster_dat  = raster.read_bands()
    polygon_dat = polygon.read_bands(band_lst=[1]) #1 band is enough, we use it to mask raster data

    #The masking we need
    #boolean casting is essential.
    polygon_mask = polygon_dat[:, :, 0].astype(np.bool)

    #Distinguishing inside and outside
    raster_shape = raster_dat.shape
    polygon_shape = polygon_dat.shape

    if (raster_shape[0] != polygon_shape[0]) or (raster_shape[1] != polygon_shape[1]):

        raise Shape_Mismatched_Error(f"Masking of size {polygon_shape} can't work with object of shape {raster_shape}")

    inside  = raster_dat[polygon_mask]
    outside = raster_dat[~polygon_mask]

    return raster_dat, polygon_dat, inside, outside

