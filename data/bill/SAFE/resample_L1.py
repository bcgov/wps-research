'''
Resample Level 1 data.
'''

##########################

from pathlib import Path

from osgeo import gdal

from .info import (
    S2_L1_NATIVE_RES, 
    S2_L1_BANDS
)

from .misc import (
    _get_ENVI_paths_L1,
    _read_acquisition_time,
    _find_band_file_L1
)

##########################



def _resample_to_match(
        src_ds: gdal.Dataset,
        ref_ds: gdal.Dataset,
        *,
        resample_alg: int
) -> gdal.Dataset:
    """
    Resample src_ds onto ref_ds grid (extent, transform, size).
    Returns an in-memory GDAL dataset.
    """

    gt = ref_ds.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * ref_ds.RasterXSize
    ymin = ymax + gt[5] * ref_ds.RasterYSize  # gt[5] is negative

    return gdal.Warp(
        "",
        src_ds,
        format="MEM",
        outputBounds=(xmin, ymin, xmax, ymax),
        width=ref_ds.RasterXSize,
        height=ref_ds.RasterYSize,
        dstSRS=ref_ds.GetProjection(),
        resampleAlg=resample_alg
    )



def ENVI_band_stack_L1_resampled(
        safe_dir: str,
        band_list: list[str] = "all",
        *,
        target_resolution: int = 20,
        out_dir: Path | None = None
):
    """
    Read Sentinel-2 L1C bands and resample all bands to a common resolution
    before writing an ENVI BSQ stack.

    Parameters
    ----------
    safe_dir : str
        Path to L1C .SAFE directory
    band_list : list[str] or 'all'
        Bands to extract
    target_resolution : int
        Target grid resolution (10, 20, or 60)
    out_dir : Path | None
        Output directory
    """

    if target_resolution not in (10, 20, 60):
        raise ValueError("target_resolution must be 10, 20, or 60")

    safe_path = Path(safe_dir)
    img_dir = _get_ENVI_paths_L1(safe_path)

    # --------------------------------------------------
    # Band validation
    # --------------------------------------------------
    if band_list == "all":
        band_list = S2_L1_BANDS.copy()
    else:
        missing = [b for b in band_list if b not in S2_L1_BANDS]
        if missing:
            raise ValueError(f"Invalid L1 bands: {missing}, choose from\n{S2_L1_BANDS}")

    band_files = [_find_band_file_L1(img_dir, b) for b in band_list]

    # --------------------------------------------------
    # Select reference band at target resolution if possible
    # --------------------------------------------------
    ref_idx = None
    for i, b in enumerate(band_list):
        if S2_L1_NATIVE_RES[b] == target_resolution:
            ref_idx = i
            break

    if ref_idx is None:
        ref_idx = 0  # fallback

    ref_src = gdal.Open(str(band_files[ref_idx]))

    # --------------------------------------------------
    # Build reference grid (identity if native matches)
    # --------------------------------------------------
    if S2_L1_NATIVE_RES[band_list[ref_idx]] == target_resolution:
        ref_ds = ref_src
    else:
        scale = target_resolution / S2_L1_NATIVE_RES[band_list[ref_idx]]
        ref_ds = gdal.Warp(
            "",
            ref_src,
            format="MEM",
            xRes=ref_src.GetGeoTransform()[1] * scale,
            yRes=abs(ref_src.GetGeoTransform()[5]) * scale,
            resampleAlg=gdal.GRA_Average
        )


    xsize, ysize = ref_ds.RasterXSize, ref_ds.RasterYSize
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------
    out_path = safe_path.name.replace(".SAFE", f"_{target_resolution}m.bin")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_path

    # --------------------------------------------------
    # Create ENVI dataset
    # --------------------------------------------------
    print(f"Resampling {band_list} to {resolution} m")
    
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(
        out_path,
        xsize,
        ysize,
        len(band_list),
        gdal.GDT_Float32,
        options=["INTERLEAVE=BSQ"]
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)

    # --------------------------------------------------
    # Write bands (with resolution-aware resampling)
    # --------------------------------------------------
    for i, (band, jp2) in enumerate(zip(band_list, band_files), start=1):
        src = gdal.Open(str(jp2))
        native_res = S2_L1_NATIVE_RES[band]

        if native_res == target_resolution:
            data = src.GetRasterBand(1).ReadAsArray()
        else:
            if native_res < target_resolution:
                alg = gdal.GRA_Average      # downsample
            else:
                alg = gdal.GRA_Bilinear    # upsample

            warped = _resample_to_match(src, ref_ds, resample_alg=alg)
            data = warped.GetRasterBand(1).ReadAsArray()

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
        "target_resolution_m": str(target_resolution),
        "interleave": "bsq"
    }, "ENVI")

    ds.FlushCache()
    ds = None

    print(f"Done resampling, find file at {out_path}")



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
        help='The band info you want. Eg: B01,B02,B04',
        type=comma_separated_list,
        default='all'
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="Directory to save files to."
    )

    args = parser.parse_args()

    safe_dir = args.safe_dir
    resolution = args.res
    band_list = args.band_list
    out_dir = args.outdir


    ENVI_band_stack_L1_resampled(
        safe_dir = safe_dir,
        target_resolution = resolution,
        band_list=band_list,
        out_dir = Path(out_dir)
    )
