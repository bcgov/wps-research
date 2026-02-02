'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror '''

import sys
import s3fs
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from typing import List
import csv
import time

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
    name = product_path.split("/")[-1]
    parts = name.split("_")
    for p in parts:
        if p.startswith("T") and len(p) == 6:
            return p
    return "UNKNOWN"


def get_safe_folder_name(zip_filename: str) -> str:
    """
    Convert ZIP filename to the SAFE folder name inside the ZIP.
    Example: S2C_MSIL2A_20250726T191931_N0511_R099_T09UYS_20250727T000316.zip
          -> S2C_MSIL2A_20250726T191931_N0511_R099_T09UYS_20250727T000316.SAFE
    """
    if zip_filename.endswith(".zip"):
        return zip_filename[:-4] + ".SAFE"
    return zip_filename + ".SAFE"


def extract_cloud_percentage(fs: s3fs.S3FileSystem, s3_path: str) -> float:
    """
    Extract cloud percentage from the MTD_MSIL2A.xml file inside the ZIP.

    The ZIP structure is:
    <product_name>.zip/
        <product_name>.SAFE/
            MTD_MSIL2A.xml
    """
    zip_filename = s3_path.split("/")[-1]
    safe_folder = get_safe_folder_name(zip_filename)

    # Path to the XML file inside the ZIP on S3
    xml_internal_path = f"{safe_folder}/MTD_MSIL2A.xml"

    # Use s3fs to open the ZIP and read the XML
    # We need to read from s3://<bucket>/<path>.zip/<internal_path>
    full_zip_path = f"s3://{s3_path}"

    # Open the ZIP file and read the XML content
    with fs.open(full_zip_path, 'rb') as f:
        import zipfile
        import io

        # Read the ZIP into memory (or stream it)
        zip_data = io.BytesIO(f.read())

        with zipfile.ZipFile(zip_data, 'r') as zf:
            # Try to find the MTD_MSIL2A.xml file
            xml_content = None
            for name in zf.namelist():
                if name.endswith('MTD_MSIL2A.xml'):
                    xml_content = zf.read(name).decode('utf-8')
                    break

            if xml_content is None:
                raise RuntimeError(f"MTD_MSIL2A.xml not found in {zip_filename}")

    root = ET.fromstring(xml_content)
    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")
    if cloud is None:
        raise RuntimeError("CLOUDY_PIXEL_PERCENTAGE not found in XML")
    return float(cloud.text)


# ------------------------------------------------------------
# S3 discovery by date
# ------------------------------------------------------------
def iterate_products_by_date(fs, start: date, end: date, tiles: List[str]):
    """
    Iterate through S3 bucket by date, yielding matching product paths.
    Returns a tuple of (product_path, current_date, entries_found, matches_found)
    for progress tracking.
    """
    current = start
    one_day = timedelta(days=1)

    total_days = (end - start).days + 1
    day_count = 0

    while current <= end:
        day_count += 1
        yyyy = current.strftime("%Y")
        mm = current.strftime("%m")
        dd = current.strftime("%d")

        prefix = f"{PREFIX_ROOT}/{yyyy}/{mm}/{dd}/"
        s3_path = f"{BUCKET}/{prefix}"

        entries_found = 0
        matches_found = 0

        try:
            objs = fs.ls(s3_path)
            entries_found = len(objs)

            for obj in objs:
                if obj.endswith(".zip") and any(tile in obj for tile in tiles):
                    matches_found += 1
                    yield obj

        except Exception as e:
            # Suppress AWS credential errors, just report as "not found"
            if "AccessDenied" in str(e) or "NoCredentialsError" in str(e):
                pass  # Skip silently
            else:
                pass  # Skip other errors silently during discovery

        # Print progress for the discovery phase
        print(
            f"Discovery: {day_count}/{total_days} days "
            f"({day_count/total_days*100:.1f}%) - "
            f"Date: {current} - Entries: {entries_found}, Matches: {matches_found}"
        )

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
    print(f"\n{'='*60}")
    print("Phase 1: Discovering products...")
    print(f"{'='*60}\n")

    # Collect all products first to estimate progress
    products = list(iterate_products_by_date(fs, start_date, end_date, requested_tiles))
    total_products = len(products)

    print(f"\n{'='*60}")
    print(f"Phase 2: Extracting cloud percentages from {total_products} products...")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for idx, product in enumerate(products, start=1):
        pid = product.split("/")[-1]
        tile = extract_tile_id(product)
        try:
            cloud = extract_cloud_percentage(fs, product)
            results.append((pid, cloud))
            status = f"{cloud:.2f}%"
        except Exception as e:
            print(f"FAILED {pid}: {e}")
            status = "FAILED"

        # --------------------------------------------------------
        # Progress / ETA
        # --------------------------------------------------------
        elapsed = time.time() - start_time
        progress = idx / total_products * 100
        if idx > 0:
            remaining = (elapsed / idx) * (total_products - idx)
        else:
            remaining = 0
        print(
            f"Progress: {idx}/{total_products} "
            f"({progress:.1f}%) - Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s - {status}"
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    if results:
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}\n")
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

        print(f"Successfully processed {len(results)}/{total_products} products.")
    else:
        print("\nNo products found for the requested tiles and date range.")


if __name__ == "__main__":
    main()
