'''
misc/sen2.py

Mostly miscellaneous functions for sentinel-2 data processing
'''

import re


def read_raster_timestamp(
        path
):
    '''
    Extract the timestamp when the image was acquired.

    Parameters
    ----------
    path: path to file (relative or absolute).

    
    Returns
    -------
    The timestamp of acquisition.

    
    Note
    ----
    Very temporary, 
    accuracy depends on the current file naming convention.
    '''

    file_name = re.split(r"[\\/]", path).pop()

    #The index can change, check once before using.

    timestamp = file_name[11:26]

    return timestamp



def band_index(
        band_info_list: str,
        band: int
):
    '''
    The data opened by gdal is of different layers (rasters),
    we might want to know the index of each band.
        E.g: Band 8 is of index 2, so let's get it by data[:, :, 2]


    Parameters
    ----------
    band_info_list: List of bands and information.
    band:           Which band we want to find its index.


    Returns
    -------
    The index of that band.

    Notes
    -----
    In the case there are 2 identical bands in the same data, it returns the first index.
    '''

    from exceptions.sen2_exception import No_Band_Error

    pattern = rf"\bB{band}\b"
    
    for i, I in enumerate(band_info_list):

        if re.search(pattern, I):

            return i
        
    raise No_Band_Error(f'Band {band} is not in the data.')

    




    