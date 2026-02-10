'''
Dedicated functions to read data from SAFE file.
'''

from pathlib import Path
import xml.etree.ElementTree as ET
from osgeo import gdal
import shutil


#------------------------------------------
#
#               FOR LEVEL 2 data
#
#------------------------------------------

S2_L2_BANDS = {
    10: ["B02", "B03", "B04", "B08"],
    20: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL"],
    60: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B09", "B11", "B12", "SCL"],
}



def unzip_safe(zip_path: Path) -> Path:

    import zipfile

    safe_name = zip_path.stem + ".SAFE"
    safe_path = zip_path.parent / safe_name

    if safe_path.exists():
        return safe_path

    print(f"Unzipping {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(zip_path.parent)

    return safe_path



def _get_ENVI_paths_L2(
        safe_path: str,
        resolution: int
):
    '''
    Resolve IMG_DATA and optional QI_DATA mask paths
    for a Sentinel-2 L2A SAFE product.

    Parameters
    ----------
    safe_dir : str
        Path to .SAFE directory
    resolution : int
        10, 20, or 60

    Returns
    -------
    img_dir : Path
        IMG_DATA/R{resolution}m
    mask_paths : dict[str, Path] | None
        Mapping of mask name -> file path
    '''

    granule_root = safe_path / "GRANULE"
    granule = next(granule_root.iterdir())

    img_dir = granule / "IMG_DATA" / f"R{resolution}m"

    return img_dir
    


def _find_band_file(img_path: Path, band: str) -> Path:
    '''
    Checks if band is in img_dir
    '''
    matches = list(img_path.glob(f"*_{band}_*.jp2"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Band {band} not found in {img_path}")
    return matches[0]



def _read_acquisition_time(safe_path: Path) -> str:
    '''
    Acquisition time, very useful.
    '''
    xml_file = safe_path / "MTD_MSIL2A.xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()

    elem = root.find(".//PRODUCT_START_TIME")

    if elem is None:
        raise RuntimeError("PRODUCT_START_TIME not found in metadata XML")

    return elem.text

    

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
            raise ValueError(f"Bands {missing} not available at {resolution} m")
    
    band_files = [_find_band_file(img_dir, b) for b in band_list]


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
    acq_time = _read_acquisition_time(safe_path)

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
    


#EXTRACTS CLOUD ONLY
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




