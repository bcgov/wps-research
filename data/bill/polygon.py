'''
polygon.py

This method is to work with fire perimeter or other type of polygon.
'''

from raster import Raster
import numpy as np


def split_in_out(
        *,
        raster_dat=None,
        polygon_dat=None,

        raster_filename = None,
        polygon_filename = None       
):
    
    '''
    Extract data from inside and outside of the polygon.

    
    Parameters
    ----------
    raster_dat: data read from raster.

    poly_dat: data read from polygon.

    raster_filename: the main data where polygon was drawn onto.

    polygon_filename: file name to the polygon, .bin format (use rasterization from shape file).
        polygon needs to be of 1 band.

    
    Returns
    -------
    2 sets of data, containing the extracted data, and its corresponding row indices from the ORIGNAL DATA. 

    One for Inside, One for Outside.


    Notes
    -----
    Only works with 1 polygon file.
    
    Use rasterize_onto.py to have similar matrix shape for raster and polygon.
    '''

    from exceptions.matrix import Shape_Mismatched_Error
    
    if raster_dat is not None:
        xSize = raster_dat.shape[1]
        ySize = raster_dat.shape[0]


    if raster_filename is not None:
        #Override raster data with the one read from file
        raster = Raster(raster_filename)
        xSize = raster._xSize
        ySize = raster._ySize
        raster_dat = raster.read_bands()

    if polygon_filename is not None:
        #Override polygon data with the one read from file
        polygon = Raster(polygon_filename)
        polygon_dat = polygon.read_bands(band_lst=[1]) #1 band is enough, we use it to mask raster data

    #The masking we need, where boolean casting is essential.

    polygon_mask = polygon_dat[..., 0].astype(np.bool)

    #Distinguishing inside and outside

    raster_shape = raster_dat.shape
    polygon_shape = polygon_dat.shape

    if (raster_shape[0] != polygon_shape[0]) or (raster_shape[1] != polygon_shape[1]):

        raise Shape_Mismatched_Error(f"Masking of size {polygon_shape} can't work with object of shape {raster_shape}")

    inside  = raster_dat[polygon_mask]
    outside = raster_dat[~polygon_mask]


    #Indices of Inside, and outside
    indices_all = np.arange(0, xSize * ySize)

    flat_mask = polygon_mask.flatten()

    inside_idx = indices_all[flat_mask]
    outside_idx = indices_all[~flat_mask]

    return inside, inside_idx, outside, outside_idx
