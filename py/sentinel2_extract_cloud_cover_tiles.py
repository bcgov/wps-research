'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror '''

import sys
import s3fs
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional
import csv
import time
import pickle
import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")

# Global settings
N_WORKERS = 8
SINGLE_THREAD = False


# ------------------------------------------------------------
# Thread-safe progress tracking
# ------------------------------------------------------------
class ProgressTracker:
    def __init__(self, name: str, total: int):
        self.name = name
        self.total = total
        self.count = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, message: str = ""):
        with self.lock:
            self.count += 1
            elapsed = time.time() - self.start_time
            pct = self.count / self.total * 100 if self.total > 0 else 0
            eta = (elapsed / self.count) * (self.total - self.count) if self.count > 0 else 0
            print(f"  {self.name}: {self.count}/{self.total} ({pct:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s - {message}")


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


# ------------------------------------------------------------
# Discovery Cache
# ------------------------------------------------------------
def get_discovery_cache_filename(start_date: date, end_date: date, tiles: List[str]) -> str:
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


def load_discovery_cache(cache_file: str) -> Optional[List[str]]:
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
# Results Cache (per-product)
# ------------------------------------------------------------
def get_result_cache_filename(product_id: str) -> str:
    """
    Generate a cache filename for a single product result.
    """
    safe_name = product_id.replace("/", "_").replace("\\", "_")
    return os.path.join(RESULTS_CACHE_DIR, f"{safe_name}.pkl")


def load_result_cache(product_id: str) -> Optional[Tuple[str, float]]:
    """Load cached result for a single product."""
    cache_file = get_result_cache_filename(product_id)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def save_result_cache(product_id: str, cloud_pct: float):
    """Save result for a single product."""
    os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)
    cache_file = get_result_cache_filename(product_id)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((product_id, cloud_pct), f)
    except Exception:
        pass


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

    ns_match = re.match(r'\{(.+?)\}', root.tag)
    ns = ns_match.group(1) if ns_match else None
    print(f"      [DEBUG] XML namespace: {ns}")

    all_tags = set()
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        all_tags.add(tag)

    print(f"      [DEBUG] Total unique tags: {len(all_tags)}")
    print(f"      [DEBUG] All tags: {sorted(all_tags)[:50]}")

    cloud_elements = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if 'cloud' in tag.lower():
            cloud_elements.append((tag, elem.text[:100] if elem.text else None))

    print(f"      [DEBUG] Elements containing 'cloud': {len(cloud_elements)}")
    for tag, text in cloud_elements[:20]:
        print(f"      [DEBUG]   {tag}: {text}")

    cloudy_matches = re.findall(r'<([^>]*[Cc]loudy[^>]*)>([^<]*)<', xml_text)
    print(f"      [DEBUG] Raw regex matches for 'cloudy': {len(cloudy_matches)}")
    for tag, value in cloudy_matches[:10]:
        print(f"      [DEBUG]   <{tag}>: {value[:50]}")


def extract_cloud_from_xml(xml_text: str) -> float:
    """
    Parse XML and extract cloud percentage.
    """
    root = ET.fromstring(xml_text)

    # Try various patterns
    cloud = root.find(".//CLOUDY_PIXEL_PERCENTAGE")
    if cloud is None:
        cloud = root.find(".//{*}CLOUDY_PIXEL_PERCENTAGE")
    if cloud is None:
        cloud = root.find(".//Cloud_Coverage_Assessment")
    if cloud is None:
        cloud = root.find(".//{*}Cloud_Coverage_Assessment")

    if cloud is None:
        debug_xml_structure(root, xml_text)
        raise RuntimeError("CLOUDY_PIXEL_PERCENTAGE not found in XML")

    return float(cloud.text)


def extract_cloud_percentage_gdal(s3_path: str) -> float:
    """
    Extract cloud percentage using GDAL's VSI file system.
    """
    zip_filename = s3_path.split("/")[-1]
    safe_folder = get_safe_folder_name(zip_filename)

    path_without_bucket = s3_path[len(BUCKET) + 1:]

    http_url = f"{S3_BASE_URL}/{path_without_bucket}"
    vsi_url = f"/vsizip//vsicurl/{http_url}/{safe_folder}/MTD_MSIL2A.xml"

    xml_bytes = read_vsi_file(vsi_url)
    xml_text = xml_bytes.decode("utf-8")

    return extract_cloud_from_xml(xml_text)


def extract_cloud_percentage_s3fs(s3_path: str) -> float:
    """
    Fallback method using s3fs + zipfile.
    """
    import zipfile
    import io

    fs = s3fs.S3FileSystem(anon=True)

    with fs.open(f"s3://{s3_path}", 'rb') as f:
        zip_data = io.BytesIO(f.read())

    with zipfile.ZipFile(zip_data, 'r') as zf:
        xml_content = None
        for name in zf.namelist():
            if name.endswith('MTD_MSIL2A.xml'):
                xml_content = zf.read(name).decode('utf-8')
                break

        if xml_content is None:
            raise RuntimeError(f"MTD_MSIL2A.xml not found in ZIP")

    return extract_cloud_from_xml(xml_content)


def extract_cloud_percentage(s3_path: str) -> float:
    """
    Extract cloud percentage, using GDAL if available, otherwise s3fs fallback.
    """
    if HAS_GDAL:
        return extract_cloud_percentage_gdal(s3_path)
    else:
        return extract_cloud_percentage_s3fs(s3_path)


# ------------------------------------------------------------
# Discovery worker function
# ------------------------------------------------------------
def discover_single_date(current_date: date, tiles: List[str], progress: ProgressTracker) -> List[str]:
    """
    Worker function to discover products for a single date.
    """
    fs = s3fs.S3FileSystem(anon=True)

    yyyy = current_date.strftime("%Y")
    mm = current_date.strftime("%m")
    dd = current_date.strftime("%d")

    prefix = f"{PREFIX_ROOT}/{yyyy}/{mm}/{dd}/"
    s3_path = f"{BUCKET}/{prefix}"

    matches = []
    entries_found = 0

    try:
        objs = fs.ls(s3_path)
        entries_found = len(objs)

        for obj in objs:
            if obj.endswith(".zip") and any(tile in obj for tile in tiles):
                matches.append(obj)
    except Exception:
        pass

    # Thread-safe progress update
    progress.update(f"Date: {current_date} - Entries: {entries_found}, Matches: {len(matches)}")

    return matches


def discover_products_parallel(start_date: date, end_date: date, tiles: List[str], use_cache: bool = True) -> List[str]:
    """
    Discover products in parallel using ThreadPoolExecutor.
    """
    cache_file = get_discovery_cache_filename(start_date, end_date, tiles)

    if use_cache:
        cached = load_discovery_cache(cache_file)
        if cached is not None:
            return cached

    print(f"  Running fresh discovery in parallel with {N_WORKERS} workers...")

    # Generate list of dates
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    # Initialize progress tracker
    progress = ProgressTracker("Discovery", len(dates))

    # Run in parallel using ThreadPoolExecutor
    products = []

    if SINGLE_THREAD:
        for d in dates:
            matches = discover_single_date(d, tiles, progress)
            products.extend(matches)
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            # Submit all tasks
            futures = {executor.submit(discover_single_date, d, tiles, progress): d for d in dates}

            # Collect results as they complete (workers pick up new jobs immediately)
            for future in as_completed(futures):
                try:
                    matches = future.result()
                    products.extend(matches)
                except Exception as e:
                    d = futures[future]
                    print(f"  ERROR processing date {d}: {e}")

    # Save to cache
    if use_cache:
        save_discovery_cache(cache_file, products)

    return products


# ------------------------------------------------------------
# Extraction worker function
# ------------------------------------------------------------
def extract_single_product(s3_path: str, progress: ProgressTracker) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Worker function to extract cloud percentage for a single product.
    Returns (product_id, cloud_pct, error_msg)
    """
    pid = s3_path.split("/")[-1]

    try:
        cloud = extract_cloud_percentage(s3_path)
        # Save to cache
        save_result_cache(pid, cloud)
        result = (pid, cloud, None)
        status = f"{pid[:50]}... Cloud: {cloud:.2f}%"
    except Exception as e:
        result = (pid, None, str(e))
        status = f"{pid[:50]}... FAILED: {str(e)[:30]}"

    # Thread-safe progress update
    progress.update(status)

    return result


def extract_cloud_percentages_parallel(products: List[str], use_cache: bool = True) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str]]]:
    """
    Extract cloud percentages in parallel using ThreadPoolExecutor.
    Returns (results, failures)
    """
    # Get product IDs
    product_ids = [p.split("/")[-1] for p in products]

    # Load cached results
    cached_results = {}
    if use_cache:
        print(f"  Checking cache for {len(products)} products...")
        for pid in product_ids:
            result = load_result_cache(pid)
            if result is not None:
                cached_results[pid] = result[1]
        print(f"  Found {len(cached_results)} cached results")

    # Filter out already-cached products
    products_to_process = []
    for s3_path in products:
        pid = s3_path.split("/")[-1]
        if pid not in cached_results:
            products_to_process.append(s3_path)

    print(f"  Need to process {len(products_to_process)} new products")

    # Process new products in parallel
    new_results = []
    new_failures = []

    if products_to_process:
        # Initialize progress tracker
        progress = ProgressTracker("Extraction", len(products_to_process))

        print(f"  Starting parallel extraction with {N_WORKERS} workers...")

        if SINGLE_THREAD:
            for s3_path in products_to_process:
                pid, cloud, error = extract_single_product(s3_path, progress)
                if cloud is not None:
                    new_results.append((pid, cloud))
                else:
                    new_failures.append((pid, error))
        else:
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                # Submit all tasks
                futures = {executor.submit(extract_single_product, s3_path, progress): s3_path
                          for s3_path in products_to_process}

                # Collect results as they complete (workers pick up new jobs immediately)
                for future in as_completed(futures):
                    try:
                        pid, cloud, error = future.result()
                        if cloud is not None:
                            new_results.append((pid, cloud))
                        else:
                            new_failures.append((pid, error))
                    except Exception as e:
                        s3_path = futures[future]
                        pid = s3_path.split("/")[-1]
                        new_failures.append((pid, str(e)))

    # Combine cached + new results
    all_results = [(pid, cloud) for pid, cloud in cached_results.items()]
    all_results.extend(new_results)

    return all_results, new_failures


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    global N_WORKERS, SINGLE_THREAD

    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python sentinel2_cloud_tiles.py <yyyymmdd_start> <yyyymmdd_end> <TILE_ID> [TILE_ID ...]\n\n"
            "Options:\n"
            "  --no-cache       Skip all caching (force fresh discovery and extraction)\n"
            "  --workers=N      Set number of parallel workers (default: 8)\n"
            "  --single-thread  Run single-threaded (for debugging)\n\n"
            "Example:\n"
            "  python sentinel2_cloud_tiles.py 20240501 20240505 T10UFB T10UFA\n"
            "  python sentinel2_cloud_tiles.py 20240501 20240505 T10UFB --workers=16"
        )
        sys.exit(1)

    # Parse arguments
    use_cache = True
    args = []
    for arg in sys.argv[1:]:
        if arg == "--no-cache":
            use_cache = False
        elif arg == "--single-thread":
            SINGLE_THREAD = True
        elif arg.startswith("--workers="):
            N_WORKERS = int(arg.split("=")[1])
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

    print(f"\n{'='*70}")
    print(f"Sentinel-2 Cloud Cover Extraction")
    print(f"{'='*70}")
    print(f"Requested tiles: {', '.join(requested_tiles)}")
    print(f"Date range: {start_date} → {end_date}")
    print(f"GDAL available: {HAS_GDAL}")
    print(f"Workers: {N_WORKERS}")
    print(f"Single-thread mode: {SINGLE_THREAD}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"{'='*70}\n")

    # Phase 1: Discovery
    print(f"{'='*70}")
    print("Phase 1: Discovering products...")
    print(f"{'='*70}\n")

    t_discovery_start = time.time()
    products = discover_products_parallel(start_date, end_date, requested_tiles, use_cache)
    t_discovery_end = time.time()

    print(f"\n  Discovery complete: {len(products)} products found in {t_discovery_end - t_discovery_start:.1f}s")

    # Phase 2: Extraction
    print(f"\n{'='*70}")
    print(f"Phase 2: Extracting cloud percentages from {len(products)} products...")
    print(f"{'='*70}\n")

    t_extraction_start = time.time()
    results, failures = extract_cloud_percentages_parallel(products, use_cache)
    t_extraction_end = time.time()

    print(f"\n  Extraction complete in {t_extraction_end - t_extraction_start:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}\n")

    print(f"Total products discovered: {len(products)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"Total time: {t_extraction_end - t_discovery_start:.1f}s")

    if results:
        # Sort results by product name for consistent output
        results_sorted = sorted(results, key=lambda x: x[0])

        print(f"\nResults (sorted by product name):")
        print("-" * 90)
        for pid, cloud in results_sorted[:20]:
            print(f"{pid:80s} {cloud:6.2f}%")
        if len(results_sorted) > 20:
            print(f"  ... and {len(results_sorted) - 20} more")

        # Write to CSV
        csv_filename = "_".join(args) + ".csv"
        print(f"\nWriting {len(results_sorted)} results to CSV: {csv_filename}")
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Product", "CloudPercentage"])
            for pid, cloud in results_sorted:
                writer.writerow([pid, cloud])

        print(f"CSV written successfully.")

        # Stats
        cloud_values = [c for _, c in results]
        avg_cloud = sum(cloud_values) / len(cloud_values)
        min_cloud = min(cloud_values)
        max_cloud = max(cloud_values)
        print(f"\nCloud cover statistics:")
        print(f"  Min: {min_cloud:.2f}%")
        print(f"  Max: {max_cloud:.2f}%")
        print(f"  Avg: {avg_cloud:.2f}%")

    if failures:
        print(f"\nFailed products ({len(failures)}):")
        print("-" * 90)
        for pid, error in failures[:10]:
            print(f"{pid}: {error[:70]}...")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more failures")


if __name__ == "__main__":
    main()
