'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror '''

import sys
import s3fs
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from typing import List
import csv
import time
import pickle
import hashlib
import os
import re

# Try to import GDAL - needed for VSI file access
try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    print("WARNING: osgeo.gdal not available, will try alternative method")

BUCKET = "sentinel-products-ca-mirror"
PREFIX_ROOT = "Sentinel-2/S2MSI2A"
S3_BASE_URL = f"https://{BUCKET}.s3.amazonaws.com"
CACHE_DIR = ".sentinel2_cache"


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
    """
    if zip_filename.endswith(".zip"):
        return zip_filename[:-4] + ".SAFE"
    return zip_filename + ".SAFE"


def get_cache_filename(start_date: date, end_date: date, tiles: List[str]) -> str:
    """
    Generate a cache filename based on query parameters.
    """
    tiles_sorted = sorted(tiles)
    tiles_str = "_".join(tiles_sorted)

    if len(tiles_str) > 50:
        tiles_hash = hashlib.md5(tiles_str.encode()).hexdigest()[:8]
        tiles_part = f"{len(tiles)}tiles_{tiles_hash}"
    else:
        tiles_part = tiles_str

    filename = f"discovery_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{tiles_part}.pkl"
    return os.path.join(CACHE_DIR, filename)


def load_discovery_cache(cache_file: str) -> List[str]:
    """Load cached discovery results if available."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded {len(data)} products from cache: {cache_file}")
            return data
        except Exception as e:
            print(f"  Cache load failed: {e}")
    return None


def save_discovery_cache(cache_file: str, products: List[str]):
    """Save discovery results to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(products, f)
        print(f"  Saved {len(products)} products to cache: {cache_file}")
    except Exception as e:
        print(f"  Cache save failed: {e}")


# ------------------------------------------------------------
# GDAL VSI file reading
# ------------------------------------------------------------
def read_vsi_file(vsi_path: str) -> bytes:
    """
    Read a file using GDAL's virtual file system.
    """
    f = gdal.VSIFOpenL(vsi_path, 'rb')
    if f is None:
        raise RuntimeError(f"Failed to open: {vsi_path}")

    try:
        gdal.VSIFSeekL(f, 0, 2)
        size = gdal.VSIFTellL(f)
        gdal.VSIFSeekL(f, 0, 0)
        data = gdal.VSIFReadL(1, size, f)
        return data
    finally:
        gdal.VSIFCloseL(f)


def debug_xml_structure(root: ET.Element, xml_text: str):
    """
    Print debug information about the XML structure to help find cloud percentage.
    """
    print(f"      [DEBUG] XML root tag: {root.tag}")

    # Strip namespace for easier searching
    ns_match = re.match(r'\{(.+?)\}', root.tag)
    ns = ns_match.group(1) if ns_match else None
    print(f"      [DEBUG] XML namespace: {ns}")

    # List all unique element tags (first 50)
    all_tags = set()
    for elem in root.iter():
        # Remove namespace prefix for readability
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        all_tags.add(tag)

    print(f"      [DEBUG] Total unique tags: {len(all_tags)}")
    print(f"      [DEBUG] All tags: {sorted(all_tags)[:50]}")
    if len(all_tags) > 50:
        print(f"      [DEBUG]   ... and {len(all_tags) - 50} more")

    # Search for anything containing "cloud" (case insensitive)
    cloud_elements = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if 'cloud' in tag.lower():
            cloud_elements.append((tag, elem.text[:100] if elem.text else None))

    print(f"      [DEBUG] Elements containing 'cloud': {len(cloud_elements)}")
    for tag, text in cloud_elements[:20]:
        print(f"      [DEBUG]   {tag}: {text}")

    # Search for anything containing "percent" (case insensitive)
    percent_elements = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if 'percent' in tag.lower():
            percent_elements.append((tag, elem.text[:100] if elem.text else None))

    print(f"      [DEBUG] Elements containing 'percent': {len(percent_elements)}")
    for tag, text in percent_elements[:20]:
        print(f"      [DEBUG]   {tag}: {text}")

    # Also search in raw text for CLOUDY
    cloudy_matches = re.findall(r'<([^>]*[Cc]loudy[^>]*)>([^<]*)<', xml_text)
    print(f"      [DEBUG] Raw regex matches for 'cloudy': {len(cloudy_matches)}")
    for tag, value in cloudy_matches[:10]:
        print(f"      [DEBUG]   <{tag}>: {value[:50]}")

    # Try different XPath patterns
    test_patterns = [
        ".//CLOUDY_PIXEL_PERCENTAGE",
        ".//{*}CLOUDY_PIXEL_PERCENTAGE",
        ".//Cloud_Coverage_Assessment",
        ".//{*}Cloud_Coverage_Assessment",
        ".//CLOUDY_PIXEL_OVER_LAND_PERCENTAGE",
        ".//{*}CLOUDY_PIXEL_OVER_LAND_PERCENTAGE",
    ]

    print(f"      [DEBUG] Testing XPath patterns:")
    for pattern in test_patterns:
        try:
            result = root.find(pattern)
            if result is not None:
                print(f"      [DEBUG]   {pattern}: FOUND -> {result.text}")
            else:
                print(f"      [DEBUG]   {pattern}: not found")
        except Exception as e:
            print(f"      [DEBUG]   {pattern}: error - {e}")


def extract_cloud_percentage_gdal(s3_path: str) -> float:
    """
    Extract cloud percentage using GDAL's VSI file system.
    """
    zip_filename = s3_path.split("/")[-1]
    safe_folder = get_safe_folder_name(zip_filename)

    path_without_bucket = s3_path[len(BUCKET) + 1:]

    http_url = f"{S3_BASE_URL}/{path_without_bucket}"
    vsi_url = f"/vsizip//vsicurl/{http_url}/{safe_folder}/MTD_MSIL2A.xml"

    print(f"      [DEBUG] VSI path: {vsi_url}")

    t_read_start = time.time()
    xml_bytes = read_vsi_file(vsi_url)
    t_read_end = time.time()
    print(f"      [DEBUG] VSI read took {t_read_end - t_read_start:.2f}s, got {len(xml_bytes)} bytes")

    xml_text = xml_bytes.decode("utf-8")

    t_parse_start = time.time()
    root = ET.fromstring(xml_text)
    t_parse_end = time.time()
    print(f"      [DEBUG] XML parse took {t_parse_end - t_parse_start:.4f}s")

    # Try to find cloud percentage with various patterns
    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")

    # If not found, try with namespace wildcard
    if cloud is None:
        cloud = root.find(".//{*}CLOUDY_PIXEL_PERCENTAGE")

    # If still not found, try Cloud_Coverage_Assessment
    if cloud is None:
        cloud = root.find(".//Cloud_Coverage_Assessment")

    if cloud is None:
        cloud = root.find(".//{*}Cloud_Coverage_Assessment")

    if cloud is None:
        # Debug the XML structure
        print(f"      [DEBUG] Cloud element not found, debugging XML structure...")
        debug_xml_structure(root, xml_text)
        raise RuntimeError("CLOUDY_PIXEL_PERCENTAGE not found in XML")

    return float(cloud.text)


def extract_cloud_percentage_s3fs(fs: s3fs.S3FileSystem, s3_path: str) -> float:
    """
    Fallback method using s3fs + zipfile.
    """
    import zipfile
    import io

    zip_filename = s3_path.split("/")[-1]

    print(f"      [DEBUG] Using s3fs fallback method")
    print(f"      [DEBUG] Opening s3://{s3_path}")

    t_download_start = time.time()

    with fs.open(f"s3://{s3_path}", 'rb') as f:
        zip_data = io.BytesIO(f.read())

    t_download_end = time.time()
    print(f"      [DEBUG] Download took {t_download_end - t_download_start:.2f}s")

    t_extract_start = time.time()
    with zipfile.ZipFile(zip_data, 'r') as zf:
        xml_content = None
        for name in zf.namelist():
            if name.endswith('MTD_MSIL2A.xml'):
                xml_content = zf.read(name).decode('utf-8')
                print(f"      [DEBUG] Found XML at: {name}")
                break

        if xml_content is None:
            raise RuntimeError(f"MTD_MSIL2A.xml not found in {zip_filename}")

    t_extract_end = time.time()
    print(f"      [DEBUG] Extract took {t_extract_end - t_extract_start:.2f}s")

    root = ET.fromstring(xml_content)

    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")
    if cloud is None:
        cloud = root.find(".//{*}CLOUDY_PIXEL_PERCENTAGE")
    if cloud is None:
        cloud = root.find(".//Cloud_Coverage_Assessment")
    if cloud is None:
        cloud = root.find(".//{*}Cloud_Coverage_Assessment")

    if cloud is None:
        debug_xml_structure(root, xml_content)
        raise RuntimeError("CLOUDY_PIXEL_PERCENTAGE not found in XML")

    return float(cloud.text)


def extract_cloud_percentage(s3_path: str, fs: s3fs.S3FileSystem = None) -> float:
    """
    Extract cloud percentage, using GDAL if available, otherwise s3fs fallback.
    """
    if HAS_GDAL:
        return extract_cloud_percentage_gdal(s3_path)
    elif fs is not None:
        return extract_cloud_percentage_s3fs(fs, s3_path)
    else:
        raise RuntimeError("No method available to read ZIP files (need GDAL or s3fs)")


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
            if "AccessDenied" in str(e) or "NoCredentialsError" in str(e):
                pass
            else:
                pass

        print(
            f"Discovery: {day_count}/{total_days} days "
            f"({day_count/total_days*100:.1f}%) - "
            f"Date: {current} - Entries: {entries_found}, Matches: {matches_found}"
        )

        current += one_day


def discover_products(fs, start_date: date, end_date: date, tiles: List[str], use_cache: bool = True) -> List[str]:
    """
    Discover products, using cache if available.
    """
    cache_file = get_cache_filename(start_date, end_date, tiles)

    if use_cache:
        cached = load_discovery_cache(cache_file)
        if cached is not None:
            return cached

    print(f"  Running fresh discovery...")
    products = list(iterate_products_by_date(fs, start_date, end_date, tiles))

    if use_cache:
        save_discovery_cache(cache_file, products)

    return products


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python sentinel2_cloud_tiles.py <yyyymmdd_start> <yyyymmdd_end> <TILE_ID> [TILE_ID ...]\n\n"
            "Options:\n"
            "  --no-cache    Skip discovery cache (force fresh discovery)\n\n"
            "Example:\n"
            "  python sentinel2_cloud_tiles.py 20240501 20240505 T10UFB T10UFA"
        )
        sys.exit(1)

    use_cache = True
    args = []
    for arg in sys.argv[1:]:
        if arg == "--no-cache":
            use_cache = False
        else:
            args.append(arg)

    if len(args) < 3:
        print("Error: Need at least start_date, end_date, and one tile ID")
        sys.exit(1)

    start_date = parse_yyyymmdd(args[0])
    end_date = parse_yyyymmdd(args[1])
    requested_tiles = args[2:]

    if start_date > end_date:
        raise ValueError("Start date must be <= end date")

    fs = s3fs.S3FileSystem(anon=True)

    print(f"\nRequested tiles: {', '.join(requested_tiles)}")
    print(f"Date range: {start_date} → {end_date}")
    print(f"GDAL available: {HAS_GDAL}")
    print(f"Discovery cache: {'enabled' if use_cache else 'disabled'}")
    print(f"\n{'='*60}")
    print("Phase 1: Discovering products...")
    print(f"{'='*60}\n")

    products = discover_products(fs, start_date, end_date, requested_tiles, use_cache)
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
            cloud = extract_cloud_percentage(product, fs)
            results.append((pid, cloud))
            status = f"OK - Cloud: {cloud:.2f}%"
        except Exception as e:
            failed.append((pid, str(e)))
            status = f"FAILED: {e}"
            print(f"    [DEBUG] Exception type: {type(e).__name__}")

        t_end = time.time()

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

    # Summary
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

        csv_filename = "_".join(args) + ".csv"
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
        for pid, error in failed[:10]:
            print(f"{pid}: {error[:80]}...")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more failures")


if __name__ == "__main__":
    main()
