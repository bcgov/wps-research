'''
Miscellaneous file for sentinel-2 data processing.
'''

import re
import numpy as np


def band_index(
        band_info_list: str,
        band: int
):
    '''
    Description
    ------------
    The data opened by gdal is of different layers (rasters),

    we might want to know the index of each band.

        E.g: Band 8 is of index 2, so let's get it by data[:, :, 2]


    Parameters
    ----------
    band_info_list: List of bands and information.

    band: Which band we want to find its index.


    Returns
    -------
    The index of that band.
    

    Notes
    -----
    In the case there are 2 identical bands in the same data, it returns the first index.
    '''

    pattern = rf"\bB{band}\b"
    
    for i, I in enumerate(band_info_list):

        if re.search(pattern, I):

            return i
        
    raise LookupError(f'Band {band} is not in the data.')



def band_name(
        band_info_list: str,
        band_index: int
):
    '''
    Description
    -----------
    Get the band name at the index.
    '''

    try:
        that_band = band_info_list[band_index]

        return re.search(r"\bB\d{1,2}\b", that_band).group()
    
    except Exception:
        
        return that_band



def writeENVI(
    output_filename: str,
    data: np.ndarray,
    *,
    mode: str = "new",
    ref_filename: str | None = None,
    band_names: list[str] | None = None,
    copy_geo: bool = True
):
    """
    mode="new":
        - creates a brand-new ENVI file
        - ignores pixel values in ref_filename
        - optionally copies georeferencing

    mode="add":
        - appends band(s) to ref_filename
        - creates a NEW file (ENVI limitation)
    """

    from osgeo import gdal
    import numpy as np

    if data.ndim == 2:
        data = data[..., None]

    H, W, K = data.shape

    if band_names is not None and len(band_names) != K:
        raise ValueError("band_names length mismatch")

    driver = gdal.GetDriverByName("ENVI")

    # -------------------------------
    # MODE: NEW FILE
    # -------------------------------
    if mode == "new":
        if ref_filename is None:
            raise ValueError("ref_filename required to define geometry")

        ref = gdal.Open(ref_filename)
        if ref is None:
            raise RuntimeError("Failed to open reference file")

        out = driver.Create(
            output_filename,
            W, H,
            K,
            gdal.GDT_Float32
        )

        if copy_geo:
            out.SetGeoTransform(ref.GetGeoTransform())
            out.SetProjection(ref.GetProjection())

        for i in range(K):
            band = out.GetRasterBand(i + 1)
            band.WriteArray(data[..., i])
            if band_names:
                band.SetDescription(band_names[i])

        out.FlushCache()
        out = ref = None
        return

    # -------------------------------
    # MODE: ADD BANDS
    # -------------------------------
    if mode == "add":
        if ref_filename is None:
            raise ValueError("ref_filename required for add mode")

        ref = gdal.Open(ref_filename)
        if ref is None:
            raise RuntimeError("Failed to open reference file")

        H0, W0, B0 = ref.RasterYSize, ref.RasterXSize, ref.RasterCount

        if (H, W) != (H0, W0):
            raise ValueError("New band shape mismatch with reference")

        out = driver.Create(
            output_filename,
            W0, H0,
            B0 + K,
            ref.GetRasterBand(1).DataType
        )

        out.SetGeoTransform(ref.GetGeoTransform())
        out.SetProjection(ref.GetProjection())

        # copy existing bands
        for i in range(B0):
            band = out.GetRasterBand(i + 1)
            band.WriteArray(ref.GetRasterBand(i + 1).ReadAsArray())
            band.SetDescription(ref.GetRasterBand(i + 1).GetDescription())

        # append new bands
        for j in range(K):
            band = out.GetRasterBand(B0 + j + 1)
            band.WriteArray(data[..., j])
            if band_names:
                band.SetDescription(band_names[j])

        out.FlushCache()
        out = ref = None

        print("Saved new data !")
        return

    raise ValueError("mode must be 'new' or 'add'")

    










    










    



    

