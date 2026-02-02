'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror '''

import sys
import s3fs
import rasterio
import xml.etree.ElementTree as ET
from datetime import datetime, date
from typing import List
import csv

BUCKET = "sentinel-products-ca-mirror"
PREFIX_ROOT = "Sentinel-2/S2MSI2A"


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def parse_yyyymmdd(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format (expected yyyymmdd): {s}")


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


# ------------------------------------------------------------
# S3 discovery by date
# ------------------------------------------------------------
def iterate_products_by_date(fs, start: date, end: date, tiles: List[str]):
    """
    Yield product ZIP paths for the requested tile IDs in the date range.
    """
    current = start
    one_day = date.fromordinal(start.toordinal() + 1) - start

    while current <= end:
        yyyy = current.strftime("%Y")
        mm = current.strftime("%m")
        dd = current.strftime("%d")

        prefix = f"{PREFIX_ROOT}/{yyyy}/{mm}/{dd}/"
        s3_path = f"{BUCKET}/{prefix}"

        print(f"\n### DEBUG: Listing S3 path: {s3_path}")

        try:
            objs = fs.ls(s3_path)
            print(f"### DEBUG: {len(objs)} entries found")

            for obj in objs:
                if obj.endswith(".zip") and any(tile in obj for tile in tiles):
                    print(f"### DEBUG: Found ZIP for requested tile -> {obj}")
                    yield obj

        except Exception as e:
            print(f"### DEBUG: fs.ls failed for {s3_path}: {e}")

        current += one_day


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python sentinel2_cloud_tiles.py <yyyymmdd_start> <yyyymmdd_end> <TILE_ID> [TILE_ID ...]\n\n"
            "Example:\n"
            "  python sentinel2_cloud_tiles.py 20240501 20240505 T10UFB T10UFA"
        )
        sys.exit(1)

    start_date = parse_yyyymmdd(sys.argv[1])
    end_date = parse_yyyymmdd(sys.argv[2])
    requested_tiles = sys.argv[3:]

    if start_date > end_date:
        raise ValueError("Start date must be <= end date")

    fs = s3fs.S3FileSystem(anon=True)

    print(f"\nRequested tiles: {', '.join(requested_tiles)}")
    print(f"Date range: {start_date} → {end_date}")

    # --------------------------------------------------------
    # Process requested tiles
    # --------------------------------------------------------
    results = []
    for product in iterate_products_by_date(fs, start_date, end_date, requested_tiles):
        tile = extract_tile_id(product)
        pid = product.split("/")[-1]
        try:
            cloud = extract_cloud_percentage(product)
            results.append((pid, cloud))
            print(f"{pid} → {cloud:.2f}%")
        except Exception as e:
            print(f"FAILED {pid}: {e}")

    if results:
        print("\nSummary:")
        for pid, cloud in results:
            print(f"{pid:80s} {cloud:6.2f}%")

        # --------------------------------------------------------
        # Output to CSV
        # --------------------------------------------------------
        csv_filename = "_".join(sys.argv[1:]) + ".csv"
        print(f"\nWriting results to CSV: {csv_filename}")
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Product", "CloudPercentage"])
            for pid, cloud in results:
                writer.writerow([pid, cloud])
    else:
        print("\nNo products found for the requested tiles and date range.")


if __name__ == "__main__":
    main()

