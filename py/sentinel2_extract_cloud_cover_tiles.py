'''20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror '''

#!/usr/bin/env python3
"""
Sentinel-2 Cloud Cover Extraction and Visualization

Discovers Sentinel-2 products, extracts cloud cover percentages, and generates
a plot showing cloud cover over time by tile with the best 5 days highlighted.

Usage:
    python sentinel2_cloud_tiles.py <yyyymmdd_start> <yyyymmdd_end> <TILE_ID> [TILE_ID ...]

Options:
    --no-cache       Skip all caching (force fresh discovery and extraction)
    --workers=N      Set number of parallel workers (default: 8)
    --single-thread  Run single-threaded (for debugging)
    --output=FILE    Output plot filename (default: cloud_cover_by_tile.png)

Example:
    python sentinel2_cloud_tiles.py 20240501 20240505 T10UFB T10UFA
    python sentinel2_cloud_tiles.py 20240501 20240505 T10UFB --workers=16
"""

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
from collections import defaultdict

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, plotting disabled")

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


def extract_date_from_product(product_id: str) -> Optional[datetime]:
    """
    Extract sensing date from product filename.
    Format: S2A_MSIL2A_20240503T202851_N0510_R114_T09WWQ_20240504T013252.zip
    """
    parts = product_id.split("_")
    for p in parts:
        if len(p) == 15 and p[8] == "T":
            try:
                return datetime.strptime(p[:8], "%Y%m%d")
            except ValueError:
                continue
    return None


def get_safe_folder_name(zip_filename: str) -> str:
    """Convert ZIP filename to the SAFE folder name inside the ZIP."""
    if zip_filename.endswith(".zip"):
        return zip_filename[:-4] + ".SAFE"
    return zip_filename + ".SAFE"


# ------------------------------------------------------------
# Discovery Cache
# ------------------------------------------------------------
def get_discovery_cache_filename(start_date: date, end_date: date, tiles: List[str]) -> str:
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
    safe_name = product_id.replace("/", "_").replace("\\", "_")
    return os.path.join(RESULTS_CACHE_DIR, f"{safe_name}.pkl")


def load_result_cache(product_id: str) -> Optional[Tuple[str, float]]:
    cache_file = get_result_cache_filename(product_id)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def save_result_cache(product_id: str, cloud_pct: float):
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
    root = ET.fromstring(xml_text)

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
    zip_filename = s3_path.split("/")[-1]
    safe_folder = get_safe_folder_name(zip_filename)

    path_without_bucket = s3_path[len(BUCKET) + 1:]

    http_url = f"{S3_BASE_URL}/{path_without_bucket}"
    vsi_url = f"/vsizip//vsicurl/{http_url}/{safe_folder}/MTD_MSIL2A.xml"

    xml_bytes = read_vsi_file(vsi_url)
    xml_text = xml_bytes.decode("utf-8")

    return extract_cloud_from_xml(xml_text)


def extract_cloud_percentage_s3fs(s3_path: str) -> float:
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
    if HAS_GDAL:
        return extract_cloud_percentage_gdal(s3_path)
    else:
        return extract_cloud_percentage_s3fs(s3_path)


# ------------------------------------------------------------
# Discovery worker function
# ------------------------------------------------------------
def discover_single_date(current_date: date, tiles: List[str], progress: ProgressTracker) -> List[str]:
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

    progress.update(f"Date: {current_date} - Entries: {entries_found}, Matches: {len(matches)}")

    return matches


def discover_products_parallel(start_date: date, end_date: date, tiles: List[str], use_cache: bool = True) -> List[str]:
    cache_file = get_discovery_cache_filename(start_date, end_date, tiles)

    if use_cache:
        cached = load_discovery_cache(cache_file)
        if cached is not None:
            return cached

    print(f"  Running fresh discovery in parallel with {N_WORKERS} workers...")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    progress = ProgressTracker("Discovery", len(dates))

    products = []

    if SINGLE_THREAD:
        for d in dates:
            matches = discover_single_date(d, tiles, progress)
            products.extend(matches)
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(discover_single_date, d, tiles, progress): d for d in dates}

            for future in as_completed(futures):
                try:
                    matches = future.result()
                    products.extend(matches)
                except Exception as e:
                    d = futures[future]
                    print(f"  ERROR processing date {d}: {e}")

    if use_cache:
        save_discovery_cache(cache_file, products)

    return products


# ------------------------------------------------------------
# Extraction worker function
# ------------------------------------------------------------
def extract_single_product(s3_path: str, progress: ProgressTracker) -> Tuple[str, Optional[float], Optional[str]]:
    pid = s3_path.split("/")[-1]

    try:
        cloud = extract_cloud_percentage(s3_path)
        save_result_cache(pid, cloud)
        result = (pid, cloud, None)
        status = f"{pid[:50]}... Cloud: {cloud:.2f}%"
    except Exception as e:
        result = (pid, None, str(e))
        status = f"{pid[:50]}... FAILED: {str(e)[:30]}"

    progress.update(status)

    return result


def extract_cloud_percentages_parallel(products: List[str], use_cache: bool = True) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str]]]:
    product_ids = [p.split("/")[-1] for p in products]

    cached_results = {}
    if use_cache:
        print(f"  Checking cache for {len(products)} products...")
        for pid in product_ids:
            result = load_result_cache(pid)
            if result is not None:
                cached_results[pid] = result[1]
        print(f"  Found {len(cached_results)} cached results")

    products_to_process = []
    for s3_path in products:
        pid = s3_path.split("/")[-1]
        if pid not in cached_results:
            products_to_process.append(s3_path)

    print(f"  Need to process {len(products_to_process)} new products")

    new_results = []
    new_failures = []

    if products_to_process:
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
                futures = {executor.submit(extract_single_product, s3_path, progress): s3_path
                          for s3_path in products_to_process}

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

    all_results = [(pid, cloud) for pid, cloud in cached_results.items()]
    all_results.extend(new_results)

    return all_results, new_failures


# ------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------
def organize_by_tile_and_date(results: List[Tuple[str, float]]) -> Dict[str, List[Tuple[datetime, float]]]:
    """Organize results by tile ID, with date and cloud percentage."""
    by_tile = defaultdict(list)

    for product_id, cloud_pct in results:
        tile_id = extract_tile_id(product_id)
        sensing_date = extract_date_from_product(product_id)

        if sensing_date is None:
            continue

        by_tile[tile_id].append((sensing_date, cloud_pct))

    for tile_id in by_tile:
        by_tile[tile_id].sort(key=lambda x: x[0])

    return dict(by_tile)


def aggregate_by_day(tile_data: List[Tuple[datetime, float]]) -> Tuple[List[datetime], List[float]]:
    """Aggregate multiple observations on the same day by averaging."""
    by_day = defaultdict(list)

    for dt, cloud_pct in tile_data:
        day = dt.date()
        by_day[day].append(cloud_pct)

    days = sorted(by_day.keys())
    dates = [datetime.combine(d, datetime.min.time()) for d in days]
    averages = [sum(by_day[d]) / len(by_day[d]) for d in days]

    return dates, averages


def compute_daily_totals(data_by_tile: Dict[str, List[Tuple[datetime, float]]]) -> Dict[date, Tuple[float, int]]:
    """Compute total cloud coverage across all tiles for each day."""
    daily_totals = defaultdict(lambda: [0.0, 0])

    for tile_id, tile_data in data_by_tile.items():
        for dt, cloud_pct in tile_data:
            day = dt.date()
            daily_totals[day][0] += cloud_pct
            daily_totals[day][1] += 1

    return {day: (vals[0], vals[1]) for day, vals in daily_totals.items()}


def find_best_days(daily_totals: Dict[date, Tuple[float, int]], n: int = 5) -> List[Tuple[date, float, int]]:
    """Find the N days with lowest total cloud coverage."""
    sorted_days = sorted(daily_totals.items(), key=lambda x: x[1][0])
    return [(day, total, count) for day, (total, count) in sorted_days[:n]]


def print_best_days(best_days: List[Tuple[date, float, int]], data_by_tile: Dict[str, List[Tuple[datetime, float]]]):
    """Print details about the best days."""
    print("\n" + "=" * 70)
    print("BEST 5 DAYS (Lowest Total Cloud Coverage Across All Tiles)")
    print("=" * 70)

    for rank, (day, total_cloud, num_obs) in enumerate(best_days, 1):
        avg_cloud = total_cloud / num_obs if num_obs > 0 else 0
        print(f"\n#{rank}: {day.strftime('%Y-%m-%d')}")
        print(f"    Total cloud sum: {total_cloud:.2f}%")
        print(f"    Number of observations: {num_obs}")
        print(f"    Average cloud per tile: {avg_cloud:.2f}%")

        print(f"    Per-tile breakdown:")
        for tile_id in sorted(data_by_tile.keys()):
            tile_data = data_by_tile[tile_id]
            for dt, cloud_pct in tile_data:
                if dt.date() == day:
                    print(f"      {tile_id}: {cloud_pct:.2f}%")


def plot_cloud_cover(data_by_tile: Dict[str, List[Tuple[datetime, float]]],
                     best_days: List[Tuple[date, float, int]],
                     output_file: str = "cloud_cover_by_tile.png"):
    """Create a line plot of cloud cover over time, one line per tile."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    if not data_by_tile:
        print("ERROR: No data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab20(range(len(data_by_tile)))

    for i, (tile_id, tile_data) in enumerate(sorted(data_by_tile.items())):
        dates, cloud_pcts = aggregate_by_day(tile_data)

        ax.plot(dates, cloud_pcts,
                marker='o',
                markersize=3,
                linewidth=1,
                alpha=0.8,
                color=colors[i],
                label=f"{tile_id} ({len(tile_data)} obs)")

    best_day_dates = [d for d, _, _ in best_days]

    for tile_id, tile_data in data_by_tile.items():
        for dt, cloud_pct in tile_data:
            if dt.date() in best_day_dates:
                ax.scatter([dt], [cloud_pct],
                          marker='X',
                          s=150,
                          c='red',
                          zorder=10,
                          edgecolors='darkred',
                          linewidths=1)

    ax.scatter([], [], marker='X', s=150, c='red', edgecolors='darkred',
               linewidths=1, label=f'Best 5 days (lowest total cloud)')

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cloud Cover (%)", fontsize=12)
    ax.set_title("Sentinel-2 Cloud Cover by Tile Over Time", fontsize=14)

    ax.set_ylim(0, 100)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    ax.grid(True, alpha=0.3)

    if len(data_by_tile) > 10:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, ncol=1)
    else:
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    global N_WORKERS, SINGLE_THREAD

    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    use_cache = True
    output_file = "cloud_cover_by_tile.png"
    args = []
    for arg in sys.argv[1:]:
        if arg == "--no-cache":
            use_cache = False
        elif arg == "--single-thread":
            SINGLE_THREAD = True
        elif arg.startswith("--workers="):
            N_WORKERS = int(arg.split("=")[1])
        elif arg.startswith("--output="):
            output_file = arg.split("=")[1]
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
    print(f"Matplotlib available: {HAS_MATPLOTLIB}")
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

        # Phase 3: Plotting
        print(f"\n{'='*70}")
        print("Phase 3: Generating plot...")
        print(f"{'='*70}\n")

        data_by_tile = organize_by_tile_and_date(results)
        print(f"Organized data for {len(data_by_tile)} tiles")

        # Print per-tile summary
        print("\nSummary by Tile:")
        print("-" * 70)
        print(f"{'Tile':<10} {'Obs':>6} {'Min':>8} {'Max':>8} {'Avg':>8} {'Date Range'}")
        print("-" * 70)

        for tile_id in sorted(data_by_tile.keys()):
            tile_data = data_by_tile[tile_id]
            tile_cloud_values = [c for _, c in tile_data]
            tile_dates = [d for d, _ in tile_data]

            tile_min = min(tile_cloud_values)
            tile_max = max(tile_cloud_values)
            tile_avg = sum(tile_cloud_values) / len(tile_cloud_values)
            date_range = f"{min(tile_dates).strftime('%Y-%m-%d')} to {max(tile_dates).strftime('%Y-%m-%d')}"

            print(f"{tile_id:<10} {len(tile_data):>6} {tile_min:>7.2f}% {tile_max:>7.2f}% {tile_avg:>7.2f}% {date_range}")

        # Find and print best days
        daily_totals = compute_daily_totals(data_by_tile)
        best_days = find_best_days(daily_totals, n=5)
        print_best_days(best_days, data_by_tile)

        # Generate plot
        plot_cloud_cover(data_by_tile, best_days, output_file)

    if failures:
        print(f"\nFailed products ({len(failures)}):")
        print("-" * 90)
        for pid, error in failures[:10]:
            print(f"{pid}: {error[:70]}...")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more failures")


if __name__ == "__main__":
    main()
