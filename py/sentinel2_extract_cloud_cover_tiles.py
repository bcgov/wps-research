'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror
'''
import s3fs
import rasterio
import xml.etree.ElementTree as ET
from datetime import date
from typing import List

BUCKET = "sentinel-products-ca-mirror"
PREFIX_ROOT = "Sentinel-2/S2MSI2A"


def extract_cloud_percentage(zip_url: str) -> float:
    """
    Reads CLOUDY_PIXEL_PERCENTAGE from MTD_MSIL2A.xml inside a ZIP on S3.
    """
    xml_path = (
        f"/vsizip//vsis3/{zip_url}/MTD_MSIL2A.xml"
    )

    with rasterio.open(xml_path) as ds:
        xml_text = ds.read().decode("utf-8")

    root = ET.fromstring(xml_text)
    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")

    return float(cloud.text)


def list_products_for_tiles(
    fs,
    tiles: List[str],
    start: date,
    end: date,
):
    """
    Generator yielding product ZIP paths matching tile IDs and date range.
    """
    current = start
    while current <= end:
        yyyy = current.strftime("%Y")
        mm = current.strftime("%m")
        dd = current.strftime("%d")

        prefix = f"{PREFIX_ROOT}/{yyyy}/{mm}/{dd}/"

        try:
            for obj in fs.ls(prefix):
                if not obj.endswith(".zip"):
                    continue
                if any(tile in obj for tile in tiles):
                    yield obj
        except FileNotFoundError:
            pass

        current = current.fromordinal(current.toordinal() + 1)


def main():
    tiles = ["T10UFB", "T10UFA"]
    start_date = date(2024, 5, 1)
    end_date   = date(2024, 5, 5)

    fs = s3fs.S3FileSystem(anon=True)

    results = []

    for product in list_products_for_tiles(
        fs, tiles, start_date, end_date
    ):
        try:
            cloud = extract_cloud_percentage(product)
            results.append((product.split("/")[-1], cloud))
            print(f"{product.split('/')[-1]} → {cloud:.2f}%")
        except Exception as e:
            print(f"Failed {product}: {e}")

    print("\nSummary")
    for pid, cloud in results:
        print(f"{pid:80s} {cloud:6.2f}%")


if __name__ == "__main__":
    main()

