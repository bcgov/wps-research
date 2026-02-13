'''
Dedicated functions to read data from SAFE file.
'''


########################################

from pathlib import Path

from .misc import (
    _get_ENVI_paths_L2,
    _find_band_file_L2,
    _read_acquisition_time
)

from osgeo import gdal

from .info import (
    S2_L2_BANDS
)

#########################################
    

#EXTRACTS SPECTRAL BANDS (CLOUD IS OPTIONAL)
def ENVI_band_stack_L2(
        safe_dir: str,
        band_list: list[str] = 'all',
        *,
        cloud_prob = False,
        resolution: int = 20,
        out_dir: str = None
):
    '''
    Description
    -----------
    This function reads bands of the criteria and stack from .SAFE folder from sentinel 2 raw data.

    Data of Level 2


    Parameters
    ----------
    safe_dir: SAFE folder
        e.g C11659/L2/S2B_MSIL2A_20250830T191909_N0511_R099_T09UYU_20250830T225737.SAFE

    band_list

    scl: Scence Classification Layer, if True and band_list = None, will get this only.

    resolution: spatial resolution (defaul is 60m -> lowest quality)
        e.g: 10, 20 or 60m
    

    Guide
    -------
    There will be a path to L2 folder, where .SAFE folders can be found there. Use yaml file.
    '''
    safe_path = Path(safe_dir)

    img_dir = _get_ENVI_paths_L2(
        safe_path, 
        resolution
    )

    out_path = safe_path.name.replace(
        '.SAFE', 
        '.bin'
    )

    if (out_dir is not None):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / out_path

    # ------------------------------------------------------------------
    # Validate IMG bands
    # ------------------------------------------------------------------
    if band_list == 'all':
        band_list = list(S2_L2_BANDS[resolution])
        
    else:
        allowed = S2_L2_BANDS[resolution]
        missing = [b for b in band_list if b not in allowed]

        if missing:
            raise ValueError(f"Bands {missing} not available at {resolution} m, choose from\n{allowed}")
    
    band_files = [_find_band_file_L2(img_dir, b) for b in band_list]


    # ------------------------------------------------------------------
    # Resolve CLDPRB (resolution-exact)
    # ------------------------------------------------------------------
    cloud_file = None
    if cloud_prob:
        if resolution == 10:
            cloud_file = None  # Sentinel-2 has no 10 m cloud prob
        else:
            granule = next((safe_path / "GRANULE").iterdir())
            qi_dir = granule / "QI_DATA"

            candidate = qi_dir / f"MSK_CLDPRB_{resolution}m.jp2"
            if candidate.exists():
                cloud_file = candidate


    # ------------------------------------------------------------------
    # Open reference band
    # ------------------------------------------------------------------
    ref = gdal.Open(str(band_files[0]))
    xsize, ysize = ref.RasterXSize, ref.RasterYSize
    gt = ref.GetGeoTransform()
    proj = ref.GetProjection()

    # ------------------------------------------------------------------
    # Create ENVI output
    # ------------------------------------------------------------------
    n_bands = len(band_files) + (1 if cloud_file else 0)

    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(
        out_path,
        xsize,
        ysize,
        n_bands,
        gdal.GDT_Float32,
        options=["INTERLEAVE=BSQ"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)


    # ------------------------------------------------------------------
    # Write IMG_DATA bands
    # ------------------------------------------------------------------
    for i, (band, jp2) in enumerate(zip(band_list, band_files), start=1):
        src = gdal.Open(str(jp2))
        data = src.GetRasterBand(1).ReadAsArray()

        ds.GetRasterBand(i).WriteArray(data)
        ds.GetRasterBand(i).SetDescription(band)

    # ------------------------------------------------------------------
    # Write cloud probability (same resolution only)
    # ------------------------------------------------------------------
    if cloud_file:
        src = gdal.Open(str(cloud_file))
        data = src.GetRasterBand(1).ReadAsArray()

        b = len(band_list) + 1
        ds.GetRasterBand(b).WriteArray(data)
        ds.GetRasterBand(b).SetDescription(f"CLDPRB_{resolution}m")


    # ------------------------------------------------------------------
    # ENVI metadata
    # ------------------------------------------------------------------
    acq_time = _read_acquisition_time(safe_path, level=2)

    band_names = band_list.copy()
    if cloud_file:
        band_names.append(f"CLDPRB_{resolution}m")

    ds.SetMetadata({
        "band names": ", ".join(band_names),
        "acquisition_time": acq_time,
        "sensor": "Sentinel-2 MSI",
        "processing_level": "L2A",
        "interleave": "bsq"
    }, "ENVI")

    ds.FlushCache()
    ds = None

    return out_path




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    #Read directory

    parser.add_argument(
        "safe_dir",
        type=str,
        help="Input SAFE dir"
    )   

    parser.add_argument(
        "--res",
        type=int,
        default=20,
        help="Spatial resolution, default is 20m"
    )   

    def comma_separated_list(s: str):
        s = s.strip().lower()
        if s == "all":
            return "all"   # sentinel, handled later
        return [b.strip().upper() for b in s.split(",") if b.strip()]
    
    parser.add_argument(
        "--band_list",
        help='The band info you want in that resolution. Eg: B01,B02,B04',
        type=comma_separated_list,
        default='all'
    )   

    parser.add_argument(
        "--cloud",
        type=bool,
        default=False,
        help='Include cloud probability of that resolution.'
    )


    parser.add_argument(
        "--outdir",
        type=str,
        help="Directory to save files to."
    )

    args = parser.parse_args()

    safe_dir = args.safe_dir
    band_list = args.band_list
    resolution = args.res
    cloud = args.cloud
    out_dir = args.outdir


    ENVI_band_stack_L2(
        safe_dir = safe_dir,
        band_list = band_list,
        resolution = resolution,
        cloud_prob=cloud,
        out_dir = out_dir
    )



