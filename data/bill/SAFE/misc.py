'''
Miscellaneous for SAFE directory.
'''

###################

from pathlib import Path

import xml.etree.ElementTree as ET

####################


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



def _get_ENVI_paths_L1(
        safe_path: Path
) -> Path:
    """
    Resolve IMG_DATA directory for Sentinel-2 L1C SAFE.
    """
    granule_root = safe_path / "GRANULE"

    granule = next(granule_root.iterdir())

    return granule / "IMG_DATA"



def _get_ENVI_paths_L2(
        safe_path: str,
        resolution: int
) -> Path:
    """
    Resolve IMG_DATA directory for Sentinel-2 L2A SAFE.
    """

    granule_root = safe_path / "GRANULE"
    granule = next(granule_root.iterdir())

    img_dir = granule / "IMG_DATA" / f"R{resolution}m"

    return img_dir
    


def _find_band_file_L1(img_path: Path, band: str) -> Path:
    '''
    Checks if band is in img_dir
    '''
    matches = list(img_path.glob(f"*_{band}.jp2"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Band {band} not found in {img_path}")
    return matches[0]



def _find_band_file_L2(img_path: Path, band: str) -> Path:
    '''
    Checks if band is in img_dir
    '''
    matches = list(img_path.glob(f"*_{band}_*.jp2"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Band {band} not found in {img_path}")
    return matches[0]



def _read_acquisition_time(
        safe_path: Path,
        level: int
) -> str:
    
    xml_main = "MTD_MSIL2A.xml" if level == 2 else "MTD_MSIL1C.xml"

    xml_file = safe_path / xml_main
    tree = ET.parse(xml_file)
    root = tree.getroot()

    elem = root.find(".//PRODUCT_START_TIME")

    if elem is None:
        raise RuntimeError("PRODUCT_START_TIME not found in metadata XML")

    return elem.text