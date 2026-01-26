'''
This misc file focuses on working with file management.
'''

from pathlib import Path

import numpy as np


def iter_files(
        folder_name,
        file_type
    ):
    '''
    Description
    -----------
    Retrieve all file names of format ... from a folder.

    
    Parameters
    ----------
    folder_name: folder name to iterate through

    file_type: e.g .bin, .hdr, .py...


    Returns
    -------
    An iterator of file names
    '''
    
    folder = Path(folder_name)

    for p in folder.iterdir():
        if p.is_file() and p.suffix == file_type:
            yield str(p)



def writeENVI(
        output_filename: str,
        ref_filename: str,
        data: np.ndarray
):
    
    from osgeo import gdal
    
    ref_ds = gdal.Open(ref_filename, gdal.GA_ReadOnly)

    H = ref_ds.RasterYSize
    W = ref_ds.RasterXSize
    B = ref_ds.RasterCount

    driver = gdal.GetDriverByName("ENVI")

    out_ds = driver.Create(
        output_filename,
        W,
        H,
        B,
        gdal.GDT_Float32
    )

    # copy spatial metadata
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())

    if data.ndim == 2:
        out_ds.GetRasterBand(1).WriteArray(data)
    
    else:
        for i in range(data.shape[2]):

            out_ds.GetRasterBand(i + 1).WriteArray(data[..., i])
            
            out_ds.GetRasterBand(i + 1).SetDescription(
                ref_ds.GetRasterBand(i + 1).GetDescription()
            )