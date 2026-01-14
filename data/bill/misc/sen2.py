'''
misc/sen2.py

Mostly miscellaneous functions for sentinel-2 data processing
'''

import re



def read_raster_date(
        path
):
    '''
    Extract the date when the image was acquired.

    
    Parameters
    ----------
    path: path to file (relative or absolute).

    
    Returns
    -------
    The date of acquisition (no time) in date format.

    
    Note
    ----
    Very temporary, 
    accuracy depends on the current file naming convention.
    '''

    from datetime import datetime

    file_name = re.split(r"[\\/]", path).pop()

    #The index can change, check once before using.

    date = file_name[11:19]

    return datetime.strptime(date, "%Y%m%d").date()



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
    The timestamp of acquisition in string format.

    
    Note
    ----
    Very temporary, 
    accuracy depends on the current file naming convention.
    '''

    file_name = re.split(r"[\\/]", path).pop()

    #The index can change, check once before using.

    timestamp = file_name[11:26]

    return timestamp



def get_date_dict(
        folder: str,
        descending = False
):
    '''
    Description
    -----------
    A dictionary where each date acts as a key to all corresponding file names.

    E.g:     {datetime.date(2025, 4, 20): ['fire_C11659/S2C_MSIL2A_20250420T192931_N0511_R142_T09UYU_20250421T000400_cloudfree.bin_MRAP_C11659.bin'],
              datetime.date(2025, 4, 22): ['fire_C11659/S2B_MSIL2A_20250422T191909_N0511_R099_T09UYU_20250422T224118_cloudfree.bin_MRAP_C11659.bin'],
              ...}
    '''

    from collections import defaultdict

    from misc.files import iter_binary_files

    groups = defaultdict(list)

    for p in iter_binary_files(folder):
        d = read_raster_date(p)
        groups[d].append(p)

    sorted_dates = sorted(groups.keys(), reverse=descending)

    return {d: groups[d] for d in sorted_dates}




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


    




    