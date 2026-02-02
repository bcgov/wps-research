'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror
'''

import sys
import s3fs
import rasterio
import xml.etree.ElementTree as ET
from datetime import datetime, date
from typing import List, Set

BUCKET = "sentinel-products-ca-mirror"
PREFIX_ROOT = "Sentinel-2/S2MSI2A"


def parse_yyyymmdd(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format (expected yyyymmdd): {s}")


def extract_cloud_percentage(zip_url: str) -> float:
    """
    Reads CLOUDY_PIXEL_PERCENTAGE from MTD_MSIL2A.xml inside a ZIP on S3.
    """
    xml_path = f"/vsizip//vsis3/{zip_url}/MTD_MSIL2A.xml"

    with rasterio.open(xml_path) as ds:
        xml_text = ds.read().decode("utf-8")

    root = ET.fromstring(xml_text)
    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")

    if cloud is None:
        raise RuntimeError("CLOUDY_PIXEL_PERCENTAGE not found")

    return float(cloud.text)


def iterate_products_by_date(fs, start: date, end: date):
    """
    Yield all product ZIP paths in the date range.
    """
    current = start
    one_day = date.fromordinal(start.toordinal() + 1) - start

    while current <= end:
        yyyy = current.strftime("%Y")
        mm = current.strftime("%m")
        dd = current.strftime("%d")

        prefix = f"{PREFIX_ROOT}/{yyyy}/{mm}/{dd}/"

        try:
            for obj in fs.ls(prefix):
                if obj.endswith(".zip"):
                    yield obj
        except FileNotFoundError:
            pass

        current += one_day


def extract_tile_id(product_path: str) -> str:
    """
    Extract tile ID (e.g. T10UFB) from Sentinel-2 product filename.
    """
    name = product_path.split("/")[-1]
    parts = name.split("_")
    for p in parts:
        if p.startswith("T") and len(p) == 6:
            return p
    return "UNKNOWN"


def main():
    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python sentinel2_cloud.py <yyyymmdd_start> <yyyymmdd_end> <TILE_ID> [TILE_ID ...]\n\n"
            "Example:\n"
            "  python sentinel2_cloud.py 20240501 20240505 T10UFB T10UFA"
        )
        sys.exit(1)

    start_date = parse_yyyymmdd(sys.argv[1])
    end_date = parse_yyyymmdd(sys.argv[2])
    requested_tiles = sys.argv[3:]

    if start_date > end_date:
        raise ValueError("Start date must be <= end date")

    fs = s3fs.S3FileSystem(anon=True)

    # --------------------------------------------------
    # 1. Discover available tiles
    # --------------------------------------------------
    available_tiles: Set[str] = set()
    products = []

    for product in iterate_products_by_date(fs, start_date, end_date):
        tile = extract_tile_id(product)
        available_tiles.add(tile)
        products.append(product)

    print("\nAvailable tiles in date range:")
    for t in sorted(available_tiles):
        print(f"  {t}")

    print("\nRequested tiles:")
    print("  " + ", ".join(requested_tiles))
    print()

    # --------------------------------------------------
    # 2. Process requested tiles only
    # --------------------------------------------------
    for product in products:
        tile = extract_tile_id(product)
        if tile not in requested_tiles:
            continue

        pid = product.split("/")[-1]
        try:
            cloud = extract_cloud_percentage(product)
            print(f"{pid} → {cloud:.2f}%")
        except Exception as e:
            print(f"FAILED {pid}: {e}")


if __name__ == "__main__":
    main()

