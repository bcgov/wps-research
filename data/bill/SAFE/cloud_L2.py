'''
Reads cloud from Level 2A data.
'''

#####################

from pathlib import Path

from .misc import (
    unzip_safe,
    _read_acquisition_time
)

from osgeo import gdal

import shutil

######################


def ENVI_cloud_L2(
        safe_dir: str,
        resolution: int = 20,
        out_dir: str = None
):
    '''
    Extract Sentinel-2 L2A cloud probability (CLDPRB) only.

    Parameters
    ----------
    safe_dir : str
        Path to .SAFE directory
    resolution : int
        20 or 60 only

    Returns
    -------
    out_path : str
        Output ENVI .bin path
    '''

    if resolution not in (20, 60):
        raise ValueError("Cloud probability only available at 20 m or 60 m")

    safe_path = Path(safe_dir)

    # --------------------------------------------------
    # Locate CLDPRB
    # --------------------------------------------------
    granule_root = safe_path / "GRANULE"
    granule = next(granule_root.iterdir())

    qi_dir = granule / "QI_DATA"
    cloud_file = qi_dir / f"MSK_CLDPRB_{resolution}m.jp2"

    if not cloud_file.exists():
        raise FileNotFoundError(
            f"Cloud probability file not found: {cloud_file}"
        )

    # --------------------------------------------------
    # Open source
    # --------------------------------------------------
    src = gdal.Open(str(cloud_file))
    xsize, ysize = src.RasterXSize, src.RasterYSize
    gt = src.GetGeoTransform()
    proj = src.GetProjection()

    data = src.GetRasterBand(1).ReadAsArray()

    # --------------------------------------------------
    # Output path
    # --------------------------------------------------
    out_path = safe_path.name.replace(
        ".SAFE",
        f"_CLDPRB_{resolution}m.bin"
    )

    if (out_dir is not None):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / out_path

    # --------------------------------------------------
    # Create ENVI dataset
    # --------------------------------------------------
    driver = gdal.GetDriverByName("ENVI")
    ds = driver.Create(
        out_path,
        xsize,
        ysize,
        1,
        gdal.GDT_Float32,
        options=["INTERLEAVE=BSQ"]
    )

    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)

    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetDescription(f"CLDPRB_{resolution}m")

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    acq_time = _read_acquisition_time(safe_path)

    ds.SetMetadata({
        "band names": f"CLDPRB_{resolution}m",
        "acquisition_time": acq_time,
        "sensor": "Sentinel-2 MSI",
        "processing_level": "L2A",
        "interleave": "bsq"
    }, "ENVI")

    ds.FlushCache()
    ds = None

    return out_path



def ENVI_cloud_L2_from_zip_root(
        zip_root: str,
        resolution: int = 20,
        out_root: str = None
):
    '''
    Read whole root, extracts zip and write ENVI files for cloud.

    Example
    -------
    ENVI_cloud_L2_from_zip_root(
        zip_root="/data/bill/mrap/Level2", #this root contains zip files.
        resolution=60, #60m of spatial resolution
        out_root=paths['l2a_dir'] #If none, will save to zip root, with extra root of 'cloud_{resolution}_m'
    )
    '''
    
    ZIP_ROOT = Path(zip_root)

    if out_root is None:

        out_root = zip_root

    OUT_DIR = Path(out_root) / f'cloud_{resolution}m'

    for zip_file in sorted(ZIP_ROOT.glob("*.zip")):

        safe_path = None

        try:
            safe_path = unzip_safe(zip_file)

            out_file = OUT_DIR / safe_path.name.replace(
                ".SAFE",
                f"_CLDPRB_{resolution}m.bin"
            )

            if out_file.exists():
                print(f"Skipping (exists): {out_file.name}")
                continue

            ENVI_cloud_L2(
                safe_dir=str(safe_path),
                resolution=resolution,
                out_dir=OUT_DIR
            )

            print(f"✓ Extracted cloud: {out_file.name}")

            # -----------------------------------------
            # SAFE cleanup (after success)
            # -----------------------------------------
            if safe_path.exists():
                shutil.rmtree(safe_path)
                print(f"Removed SAFE: {safe_path.name}")

        except Exception as e:
            print(f"✗ Failed on {zip_file.name}: {e}")

            # Optional: keep SAFE for debugging
            if safe_path and safe_path.exists():
                print(f"Keeping SAFE for inspection: {safe_path.name}")