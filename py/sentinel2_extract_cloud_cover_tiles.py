'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror
'''

import sys
import s3fs
import rasterio
import xml.etree.ElementTree as ET
from datetime import datetime, date
from typing import Set

BUCKET = "sentinel-products-ca-mirror"
PREFIX_ROOT = "Sentinel-2/S2MSI2A"


def parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def extract_cloud_percentage(zip_url: str) -> float:
    xml_path = f"/vsizip//vsis3/{zip_url}/MTD_MSIL2A.xml"
    with rasterio.open(xml_path) as ds:
        xml_text = ds.read().decode("utf-8")

    root = ET.fromstring(xml_text)
    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")
    return float(cloud.text)


def extract_tile_id(product_path: str) -> str:
    name = product_path.split("/")[-1]
    parts = name.split("_")
    for p in parts:
        if p.startswith("T") and len(p) == 6:
            return p
    return "UNKNOWN"


def iterate_products_by_date(fs, start: date, end: date):
    current = start
    one_day = date.fromordinal(start.toordinal() + 1) - start

    while current <= end:
        yyyy = current.strftime("%Y")
        mm = current.strftime("%m")
        dd = current.strftime("%d")

        prefix = f"{PREFIX_ROOT}/{yyyy}/{mm}/{dd}/"

        print(f"\n### DEBUG: Listing prefix: {prefix}")

        try:
            objs = fs.ls(prefix)
            print(f"### DEBUG: {len(objs)} entries found")

            for obj in objs:
                print(f"### DEBUG: raw entry -> {obj}")
                if obj.endswith(".zip"):
                    yield obj

        except Exception as e:
            print(f"### DEBUG: fs.ls failed for {prefix}: {e}")

        current += one_day


def main():
    if len(sys.argv) < 4:
        print("Usage: python sentinel2_cloud.py yyyymmdd1 yyyymmdd2 TILE_ID ...")
        sys.exit(1)

    start_date = parse_yyyymmdd(sys.argv[1])
    end_date = parse_yyyymmdd(sys.argv[2])
    requested_tiles = sys.argv[3:]

    fs = s3fs.S3FileSystem(anon=True)

    available_tiles: Set[str] = set()
    products = []

    print("\n### DEBUG: Starting tile discovery")

    for product in iterate_products_by_date(fs, start_date, end_date):
        print(f"### DEBUG: Found ZIP: {product}")
        tile = extract_tile_id(product)
        print(f"### DEBUG: Extracted tile: {tile}")

        available_tiles.add(tile)
        products.append(product)

    print("\nAvailable tiles in date range:")
    if not available_tiles:
        print("  (NONE FOUND)")
    else:
        for t in sorted(available_tiles):
            print(f"  {t}")

    print("\nRequested tiles:")
    print("  " + ", ".join(requested_tiles))
    print()

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

