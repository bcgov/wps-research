'''
Dedicated functions to read data from SAFE file.
'''


########################################

from pathlib import Path

from .misc import (
    _get_ENVI_paths_L1,
    _find_band_file_L1,
    _read_acquisition_time
)

from osgeo import gdal

from .info import (
    S2_L1_BANDS
)

#########################################



def ENVI_band_stack_L1(
    safe_dir: str,
    band_list: list[str] = "all",
    *,
    out_dir: Path | None = None
):
    """

    Read Sentinel-2 L1C bands from SAFE and write ENVI BSQ stack.

    Parameters
    ----------
    safe_dir : str
        Path to L1C .SAFE directory
    band_list : list[str] or 'all'
        Bands to extract (e.g. ['B04', 'B08'])
    out_dir : Path | None
        Output directory

    """

    safe_path = Path(safe_dir)
    img_dir = _get_ENVI_paths_L1(safe_path)

    if band_list == "all":
        band_list = S2_L1_BANDS
    else:
        missing = [b for b in band_list if b not in S2_L1_BANDS]
        if missing:
            raise ValueError(f"Invalid L1 bands: {missing}, choose from\n{S2_L1_BANDS}")

    band_files = [_find_band_file_L1(img_dir, b) for b in band_list]

    # --------------------------------------------------
    # Reference band
    # --------------------------------------------------
    ref = gdal.Open(str(band_files[0]))
    xsize, ysize = ref.RasterXSize, ref.RasterYSize
    gt = ref.GetGeoTransform()
    proj = ref.GetProjection()

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------
    out_path = safe_path.name.replace(".SAFE", ".bin")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_path

    # --------------------------------------------------
    # Create ENVI dataset
    # --------------------------------------------------
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(
        out_path,
        xsize,
        ysize,
        len(band_files),
        gdal.GDT_Float32,
        options=["INTERLEAVE=BSQ"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)

    # --------------------------------------------------
    # Write bands
    # --------------------------------------------------
    for i, (band, jp2) in enumerate(zip(band_list, band_files), start=1):
        src = gdal.Open(str(jp2))
        data = src.GetRasterBand(1).ReadAsArray()
        ds.GetRasterBand(i).WriteArray(data)
        ds.GetRasterBand(i).SetDescription(band)

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    acq_time = _read_acquisition_time(safe_path, level=1)

    ds.SetMetadata({
        "band names": ", ".join(band_list),
        "acquisition_time": acq_time,
        "sensor": "Sentinel-2 MSI",
        "processing_level": "L1C",
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

    def comma_separated_list(s: str):
        s = s.strip().lower()
        if s == "all":
            return "all"   # sentinel, handled later
        return [b.strip().upper() for b in s.split(",") if b.strip()]
    
    parser.add_argument(
        "band_list",
        help='The band info you want. Eg b01,b03,b04. They must match in resolution. Use resample if not sure.',
        type=comma_separated_list
    )   


    parser.add_argument(
        "--outdir",
        type=str,
        help="Directory to save files to."
    )

    args = parser.parse_args()

    safe_dir = args.safe_dir
    band_list = args.band_list
    out_dir = args.outdir


    ENVI_band_stack_L1(
        safe_dir = safe_dir,
        band_list = band_list,
        out_dir = out_dir
    )

