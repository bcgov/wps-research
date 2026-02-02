'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror '''

import sys
import s3fs
import rasterio
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from typing import List
import csv
import time

BUCKET = "sentinel-products-ca-mirror"
PREFIX_ROOT = "Sentinel-2/S2MSI2A"
S3_BASE_URL = f"https://{BUCKET}.s3.amazonaws.com"


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


def extract_cloud_percentage(s3_path: str) -> float:
    """
    Extract cloud percentage from the MTD_MSIL2A.xml file inside the ZIP.
    Uses GDAL's /vsizip//vsicurl/ for efficient HTTP range requests.

    s3_path format: sentinel-products-ca-mirror/Sentinel-2/S2MSI2A/2025/07/26/file.zip
    """
    zip_filename = s3_path.split("/")[-1]

    # Remove bucket name to get the path portion
    path_without_bucket = s3_path[len(BUCKET) + 1:]

    # Build the vsicurl URL
    # Format: /vsizip//vsicurl/https://<bucket>.s3.amazonaws.com/<path>/MTD_MSIL2A.xml
    http_url = f"{S3_BASE_URL}/{path_without_bucket}"
    vsi_url = f"/vsizip//vsicurl/{http_url}/MTD_MSIL2A.xml"

    print(f"      [DEBUG] Opening: {vsi_url}")

    t_open_start = time.time()
    with rasterio.open(vsi_url) as ds:
        t_open_end = time.time()
        print(f"      [DEBUG] rasterio.open() took {t_open_end - t_open_start:.2f}s")

        t_read_start = time.time()
        xml_bytes = ds.read()
        t_read_end = time.time()
        print(f"      [DEBUG] ds.read() took {t_read_end - t_read_start:.2f}s, got {len(xml_bytes)} bytes")

    t_decode_start = time.time()
    xml_text = xml_bytes.decode("utf-8")
    t_decode_end = time.time()
    print(f"      [DEBUG] decode() took {t_decode_end - t_decode_start:.4f}s")

    t_parse_start = time.time()
    root = ET.fromstring(xml_text)
    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")
    t_parse_end = time.time()
    print(f"      [DEBUG] XML parse took {t_parse_end - t_parse_start:.4f}s")

    if cloud is None:
        raise RuntimeError("CLOUDY_PIXEL_PERCENTAGE not found in XML")

    return float(cloud.text)


# ------------------------------------------------------------
# S3 discovery by date
# ------------------------------------------------------------
def iterate_products_by_date(fs, start: date, end: date, tiles: List[str]):
    """
    Iterate through S3 bucket by date, yielding matching product paths.
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
    failed = []
    start_time = time.time()

    for idx, product in enumerate(products, start=1):
        pid = product.split("/")[-1]
        tile = extract_tile_id(product)

        print(f"\n  [{idx}/{total_products}] Processing: {pid}")
        print(f"    [DEBUG] Tile: {tile}")
        print(f"    [DEBUG] Full S3 path: {product}")

        t_start = time.time()
        try:
            cloud = extract_cloud_percentage(product)
            results.append((pid, cloud))
            status = f"OK - Cloud: {cloud:.2f}%"
        except Exception as e:
            failed.append((pid, str(e)))
            status = f"FAILED: {e}"
            print(f"    [DEBUG] Exception type: {type(e).__name__}")

        t_end = time.time()

        # --------------------------------------------------------
        # Progress / ETA
        # --------------------------------------------------------
        elapsed = time.time() - start_time
        progress = idx / total_products * 100
        if idx > 0:
            avg_per_item = elapsed / idx
            remaining = avg_per_item * (total_products - idx)
        else:
            remaining = 0

        print(f"    [DEBUG] This product took: {t_end - t_start:.2f}s")
        print(
            f"  Progress: {idx}/{total_products} "
            f"({progress:.1f}%) - Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s - {status}"
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}\n")

    print(f"Successfully processed: {len(results)}/{total_products}")
    print(f"Failed: {len(failed)}/{total_products}")

    if results:
        print(f"\nResults:")
        print("-" * 90)
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

        print(f"CSV written successfully.")

    if failed:
        print(f"\nFailed products:")
        print("-" * 90)
        for pid, error in failed[:10]:  # Show first 10 failures
            print(f"{pid}: {error[:60]}...")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more failures")


if __name__ == "__main__":
    main()
