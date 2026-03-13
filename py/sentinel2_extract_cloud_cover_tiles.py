#!/usr/bin/env python3
"""
20260202 extract cloud cover percentage for given tiles for specified date range..using Canada AWS sentinel-2 products mirror

Sentinel-2 Cloud Cover Extraction and Visualization

Discovers Sentinel-2 products, extracts cloud cover percentages, and generates
a plot showing cloud cover over time by tile with the best 5 days highlighted.

Usage:
    sentinel2_extract_cloud_cover_tiles.py <yyyymmdd_start> <yyyymmdd_end> <TILE_ID> [TILE_ID ...]

Options:
    --L1             Query L1C data only
    --L2             Query L2A data only
                     (default: query both L1C and L2A)
    --no-cache       Skip all caching (force fresh discovery and extraction)
    --workers=N      Set number of parallel workers (default: 8)
    --single-thread  Run single-threaded (for debugging)
    --output=PREFIX  Output filename prefix (default: cloud_cover_by_tile)
                     Will generate PREFIX_L1C.png/csv and/or PREFIX_L2A.png/csv
    --average        Overlay a thick dashed black line showing the average of all
                     tiles' cloud cover curves on the plot
    --utm_zone=ZONE  Filter tiles to only those whose tile ID starts with the
                     given UTM zone prefix (e.g. --utm_zone=T10 keeps T10UFB,
                     T10UFA, etc. and drops T09VXE, T11VLE, ...).
                     The prefix is matched case-insensitively against the leading
                     characters of each tile ID (e.g. "T10", "t10", "T09" all
                     work).  Can be combined with an explicit tile-ID list and/or
                     a date-range filter; tiles that fail either check are omitted.
                     In normal (download) mode the filtered tile list is also used
                     for S3 discovery, so no unnecessary data is fetched.
    --restore_csv=FILE
                     Generate the plot from an existing CSV file (from a previous
                     run) instead of downloading data. The FILE argument is the
                     CSV path to restore from.
                     Optionally filter by date range and/or tile IDs provided at
                     the command line. If no date range is given, all dates in the
                     CSV are used. If no tile IDs are given, all tiles are plotted.
                     The level (L1C/L2A) is inferred from the CSV filename if it
                     ends with _L1C.csv or _L2A.csv, otherwise defaults to
                     'UNKNOWN'.

Example:
    sentinel2_extract_cloud_cover_tiles.py 20240501 20240505 T10UFB T10UFA
    sentinel2_extract_cloud_cover_tiles.py 20240501 20240505 T10UFB --L2 --workers=16 --average
    sentinel2_extract_cloud_cover_tiles.py --restore_csv=cloud_cover_by_tile_L2A.csv --average
    sentinel2_extract_cloud_cover_tiles.py 20240601 20240901 T10UFB T10UFA --restore_csv=cloud_cover_by_tile_L2A.csv
    sentinel2_extract_cloud_cover_tiles.py --restore_csv=cloud_cover_by_tile_L2A.csv --utm_zone=T10 --average
    sentinel2_extract_cloud_cover_tiles.py 20240601 20240901 --restore_csv=cloud_cover_by_tile_L2A.csv --utm_zone=T09
    sentinel2_extract_cloud_cover_tiles.py 20240501 20240901 T10UFB T10UFA T09VXE --utm_zone=T10 --L2

sentinel2_extract_cloud_cover_tiles.py 20230420 20251111 T10VDJ T09VXE T10UDD T10UEF T10VEH T10UCD T10UDG T10UFE T10UFG T10UEE T10VCH T10VDH T11VLE T11VLC T09UYU T10VCM T10VEM T09VXC T09VXG T10UDE T10VCJ T10VFM T10UFF T10VCL T10VEJ T10VFL T10UGE T11ULT T11UMU T10UDF T10VDM T09VXF T10UFD T09VWC T10VEL T09VVE T09VWE T10VDL T10VFK T09VWG T11UMT T10UGC T10UCE T11ULA T10VFJ T10UEG T09VVF T10UED T10VFH T10UGD T11ULU T09VWF T11ULB T09VWD T09UXB T10UCG T10VCK T11ULV T11VLD T10UCF T11VLF T10VDK T10VEK T09UYV T09VXD T11VLG --L2 --workers=16
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
S3_BASE_URL = f"https://{BUCKET}.s3.amazonaws.com"
CACHE_DIR = ".sentinel2_cache"
RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "results")

# Level prefixes
LEVEL_PREFIXES = {
    "L1C": "Sentinel-2/S2MSI1C",
    "L2A": "Sentinel-2/S2MSI2A",
}

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


def extract_level(product_path: str) -> str:
    """Extract processing level (L1C or L2A) from product path."""
    name = product_path.split("/")[-1]
    if "MSIL1C" in name:
        return "L1C"
    elif "MSIL2A" in name:
        return "L2A"
    return "UNKNOWN"


def extract_date_from_product(product_id: str) -> Optional[datetime]:
    """Extract sensing date from product filename."""
    parts = product_id.split("_")
    for p in parts:
        if len(p) == 15 and p[8] == "T":
            try:
                return datetime.strptime(p[:8], "%Y%m%d")
            except ValueError:
                continue
    return None


def tile_matches_utm_zone(tile_id: str, utm_zone: Optional[str]) -> bool:
    """Return True if tile_id starts with utm_zone prefix (case-insensitive),
    or if utm_zone is None/empty (no filter applied)."""
    if not utm_zone:
        return True
    return tile_id.upper().startswith(utm_zone.upper())


def get_safe_folder_name(zip_filename: str) -> str:
    """Convert ZIP filename to the SAFE folder name inside the ZIP."""
    if zip_filename.endswith(".zip"):
        return zip_filename[:-4] + ".SAFE"
    return zip_filename + ".SAFE"


# ------------------------------------------------------------
# Discovery Cache
# ------------------------------------------------------------
def get_discovery_cache_filename(start_date: date, end_date: date, tiles: List[str], level: str) -> str:
    tiles_sorted = sorted(tiles)
    tiles_str = "_".join(tiles_sorted)

    if len(tiles_str) > 50:
        tiles_hash = hashlib.md5(tiles_str.encode()).hexdigest()[:8]
        tiles_part = f"{len(tiles)}tiles_{tiles_hash}"
    else:
        tiles_part = tiles_str

    filename = f"discovery_{level}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{tiles_part}.pkl"
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


def get_metadata_filename(level: str) -> str:
    """Get the metadata XML filename for the given level."""
    if level == "L1C":
        return "MTD_MSIL1C.xml"
    else:
        return "MTD_MSIL2A.xml"


def extract_cloud_percentage_gdal(s3_path: str) -> float:
    zip_filename = s3_path.split("/")[-1]
    safe_folder = get_safe_folder_name(zip_filename)
    level = extract_level(s3_path)
    metadata_file = get_metadata_filename(level)

    path_without_bucket = s3_path[len(BUCKET) + 1:]

    http_url = f"{S3_BASE_URL}/{path_without_bucket}"
    vsi_url = f"/vsizip//vsicurl/{http_url}/{safe_folder}/{metadata_file}"

    xml_bytes = read_vsi_file(vsi_url)
    xml_text = xml_bytes.decode("utf-8")

    return extract_cloud_from_xml(xml_text)


def extract_cloud_percentage_s3fs(s3_path: str) -> float:
    import zipfile
    import io

    level = extract_level(s3_path)
    metadata_file = get_metadata_filename(level)

    fs = s3fs.S3FileSystem(anon=True)

    with fs.open(f"s3://{s3_path}", 'rb') as f:
        zip_data = io.BytesIO(f.read())

    with zipfile.ZipFile(zip_data, 'r') as zf:
        xml_content = None
        for name in zf.namelist():
            if name.endswith(metadata_file):
                xml_content = zf.read(name).decode('utf-8')
                break

        if xml_content is None:
            raise RuntimeError(f"{metadata_file} not found in ZIP")

    return extract_cloud_from_xml(xml_content)


def extract_cloud_percentage(s3_path: str) -> float:
    if HAS_GDAL:
        return extract_cloud_percentage_gdal(s3_path)
    else:
        return extract_cloud_percentage_s3fs(s3_path)


# ------------------------------------------------------------
# Discovery worker function
# ------------------------------------------------------------
def discover_single_date(current_date: date, tiles: List[str], level: str, progress: ProgressTracker) -> List[str]:
    fs = s3fs.S3FileSystem(anon=True)

    yyyy = current_date.strftime("%Y")
    mm = current_date.strftime("%m")
    dd = current_date.strftime("%d")

    prefix_root = LEVEL_PREFIXES[level]
    prefix = f"{prefix_root}/{yyyy}/{mm}/{dd}/"
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

    progress.update(f"{level} Date: {current_date} - Entries: {entries_found}, Matches: {len(matches)}")

    return matches


def discover_products_parallel(start_date: date, end_date: date, tiles: List[str], level: str, use_cache: bool = True) -> List[str]:
    cache_file = get_discovery_cache_filename(start_date, end_date, tiles, level)

    if use_cache:
        cached = load_discovery_cache(cache_file)
        if cached is not None:
            return cached

    print(f"  Running fresh {level} discovery in parallel with {N_WORKERS} workers...")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    progress = ProgressTracker(f"Discovery {level}", len(dates))

    products = []

    if SINGLE_THREAD:
        for d in dates:
            matches = discover_single_date(d, tiles, level, progress)
            products.extend(matches)
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(discover_single_date, d, tiles, level, progress): d for d in dates}

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
    level = extract_level(s3_path)

    try:
        cloud = extract_cloud_percentage(s3_path)
        save_result_cache(pid, cloud)
        result = (pid, cloud, None)
        status = f"[{level}] {pid[:45]}... Cloud: {cloud:.2f}%"
    except Exception as e:
        result = (pid, None, str(e))
        status = f"[{level}] {pid[:45]}... FAILED: {str(e)[:25]}"

    progress.update(status)

    return result


def extract_cloud_percentages_parallel(products: List[str], level: str, use_cache: bool = True) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str]]]:
    product_ids = [p.split("/")[-1] for p in products]

    cached_results = {}
    if use_cache:
        print(f"  Checking cache for {len(products)} {level} products...")
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

    print(f"  Need to process {len(products_to_process)} new {level} products")

    new_results = []
    new_failures = []

    if products_to_process:
        progress = ProgressTracker(f"Extraction {level}", len(products_to_process))

        print(f"  Starting parallel {level} extraction with {N_WORKERS} workers...")

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
# CSV restore
# ------------------------------------------------------------
def load_results_from_csv(csv_file: str,
                          filter_start: Optional[date] = None,
                          filter_end: Optional[date] = None,
                          filter_tiles: Optional[List[str]] = None,
                          utm_zone: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Load (product_id, cloud_pct) pairs from a previously written CSV file.
    Optionally filter by date range, explicit tile IDs, and/or UTM zone prefix.
    All active filters are ANDed together.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    results = []
    skipped = 0

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["Product"]
            try:
                cloud_pct = float(row["CloudPercentage"])
            except (KeyError, ValueError) as e:
                print(f"  WARNING: skipping row {row}: {e}")
                skipped += 1
                continue

            # Date filter
            if filter_start is not None or filter_end is not None:
                sensing_date = extract_date_from_product(pid)
                if sensing_date is None:
                    skipped += 1
                    continue
                if filter_start is not None and sensing_date.date() < filter_start:
                    skipped += 1
                    continue
                if filter_end is not None and sensing_date.date() > filter_end:
                    skipped += 1
                    continue

            # Tile ID filter
            if filter_tiles:
                tile_id = extract_tile_id(pid)
                if tile_id not in filter_tiles:
                    skipped += 1
                    continue

            # UTM zone prefix filter
            if utm_zone:
                tile_id = extract_tile_id(pid)
                if not tile_matches_utm_zone(tile_id, utm_zone):
                    skipped += 1
                    continue

            results.append((pid, cloud_pct))

    print(f"  Loaded {len(results)} records from {csv_file} (skipped {skipped})")
    return results


def infer_level_from_csv_filename(csv_file: str) -> str:
    """Infer L1C/L2A from CSV filename suffix, e.g. *_L1C.csv or *_L2A.csv."""
    base = os.path.basename(csv_file)
    if base.endswith("_L1C.csv"):
        return "L1C"
    if base.endswith("_L2A.csv"):
        return "L2A"
    return "UNKNOWN"


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


def build_mosaic_state(data_by_tile: Dict[str, List[Tuple[datetime, float]]]) -> Dict[date, Dict[str, float]]:
    """
    Build a "rolling mosaic" state: for every calendar day that has at least
    one new acquisition across any tile, return a snapshot of the most recent
    cloud cover value for *every* tile as of that day.

    Algorithm
    ---------
    1. Collect every unique observation date across all tiles.
    2. Walk those dates in order, maintaining a per-tile "last known" value.
    3. On each date where at least one tile has a new observation, update that
       tile's last-known value (using the mean if multiple acquisitions landed
       on the same day), then record the full mosaic state for that date.

    Only dates that actually carry at least one new observation are stored, so
    the returned dict has exactly as many keys as there are unique observation
    dates in the dataset.  Tiles that have never yet been observed by a given
    date are simply absent from that date's snapshot.

    Returns
    -------
    Dict mapping date -> {tile_id: most_recent_cloud_pct}
    """
    # Per-tile sorted observation lists (date -> mean cloud for that day)
    tile_daily: Dict[str, Dict[date, float]] = {}
    for tile_id, tile_data in data_by_tile.items():
        by_day: Dict[date, List[float]] = defaultdict(list)
        for dt, cloud_pct in tile_data:
            by_day[dt.date()].append(cloud_pct)
        tile_daily[tile_id] = {d: sum(v) / len(v) for d, v in by_day.items()}

    # All unique observation dates across all tiles, sorted
    all_obs_dates = sorted({d for td in tile_daily.values() for d in td})

    mosaic: Dict[date, Dict[str, float]] = {}
    last_known: Dict[str, float] = {}

    for obs_date in all_obs_dates:
        # Update last-known values for tiles that have data on this date
        for tile_id, daily in tile_daily.items():
            if obs_date in daily:
                last_known[tile_id] = daily[obs_date]
        # Snapshot the full mosaic (copy so later updates don't mutate it)
        mosaic[obs_date] = dict(last_known)

    return mosaic


def compute_average_curve(data_by_tile: Dict[str, List[Tuple[datetime, float]]]) -> Tuple[List[datetime], List[float]]:
    """
    Compute the per-day average cloud cover across all tiles using the rolling
    mosaic state: for each observation date, average the most-recent cloud
    cover for every tile that has been seen at least once by that date.
    Returns (sorted_dates, avg_cloud_pct_per_date).
    """
    mosaic = build_mosaic_state(data_by_tile)

    days = sorted(mosaic.keys())
    dates = [datetime.combine(d, datetime.min.time()) for d in days]
    averages = [
        sum(mosaic[d].values()) / len(mosaic[d])
        for d in days
    ]

    return dates, averages


def compute_daily_totals(data_by_tile: Dict[str, List[Tuple[datetime, float]]]) -> Dict[date, Tuple[float, int]]:
    """
    Compute the mosaic-based total cloud score for each observation date.

    For every date that has at least one new acquisition, the score is the
    *sum* of the most-recent cloud cover across all tiles seen so far
    (not just tiles that acquired on that exact day).  This means every day
    is scored on equal footing — a day with only one new acquisition still
    contributes the full mosaic state for ranking purposes.

    Returns dict mapping date -> (sum_of_mosaic_cloud, num_tiles_in_mosaic).
    """
    mosaic = build_mosaic_state(data_by_tile)

    return {
        day: (sum(tile_vals.values()), len(tile_vals))
        for day, tile_vals in mosaic.items()
    }


def find_best_days(daily_totals: Dict[date, Tuple[float, int]], n: int = 5) -> List[Tuple[date, float, int]]:
    """Find the N days with lowest mosaic-total cloud coverage."""
    sorted_days = sorted(daily_totals.items(), key=lambda x: x[1][0])
    return [(day, total, count) for day, (total, count) in sorted_days[:n]]


def print_best_days(best_days: List[Tuple[date, float, int]],
                    data_by_tile: Dict[str, List[Tuple[datetime, float]]],
                    level: str):
    """Print details about the best days using the mosaic state."""
    mosaic = build_mosaic_state(data_by_tile)

    print(f"\n" + "=" * 70)
    print(f"BEST 5 DAYS - {level} (Lowest Mosaic Cloud Coverage Across All Tiles)")
    print(f"  (Each tile contributes its most recent observation on or before that date)")
    print("=" * 70)

    for rank, (day, total_cloud, num_tiles) in enumerate(best_days, 1):
        avg_cloud = total_cloud / num_tiles if num_tiles > 0 else 0
        print(f"\n#{rank}: {day.strftime('%Y-%m-%d')}")
        print(f"    Mosaic cloud sum : {total_cloud:.2f}%  ({num_tiles} tiles)")
        print(f"    Average per tile : {avg_cloud:.2f}%")

        day_mosaic = mosaic.get(day, {})
        print(f"    Per-tile mosaic breakdown (most recent value as of this date):")
        for tile_id in sorted(day_mosaic.keys()):
            # Flag tiles that actually acquired on this exact day
            tile_daily = {dt.date(): cloud for dt, cloud in data_by_tile.get(tile_id, [])}
            tag = " ← new acquisition" if day in tile_daily else " (carried forward)"
            print(f"      {tile_id}: {day_mosaic[tile_id]:.2f}%{tag}")


def plot_cloud_cover(data_by_tile: Dict[str, List[Tuple[datetime, float]]],
                     best_days: List[Tuple[date, float, int]],
                     level: str,
                     output_file: str,
                     plot_average: bool = False):
    """Create a line plot of cloud cover over time, one line per tile.

    Parameters
    ----------
    data_by_tile : dict
        Mapping of tile_id -> list of (datetime, cloud_pct).
    best_days : list
        Output of find_best_days(), used to highlight best-day markers.
    level : str
        Level string used in plot title (e.g. 'L1C' or 'L2A').
    output_file : str
        Path to write the PNG to.
    plot_average : bool
        When True, overlays a thick dashed black line showing the rolling-mosaic
        average cloud cover across all tiles (most recent value per tile per day).
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    if not data_by_tile:
        print(f"No data to plot for {level}")
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

    # ------------------------------------------------------------------
    # Average curve overlay — uses rolling mosaic (most recent per tile)
    # ------------------------------------------------------------------
    if plot_average and len(data_by_tile) > 0:
        avg_dates, avg_values = compute_average_curve(data_by_tile)
        ax.plot(avg_dates, avg_values,
                linestyle='--',
                linewidth=3,
                color='black',
                zorder=9,
                label='Average (all tiles, mosaic)')

    # ------------------------------------------------------------------
    # Best-day markers — placed at each tile's mosaic cloud value on that
    # day (most recent observation on or before the best day), so markers
    # appear even for tiles that didn't acquire on that exact date.
    # ------------------------------------------------------------------
    best_day_dates = {d for d, _, _ in best_days}
    mosaic = build_mosaic_state(data_by_tile)

    # Build a lookup: tile_id -> sorted list of (date, cloud_pct) for step-plotting
    tile_sorted: Dict[str, List[Tuple[date, float]]] = {
        tile_id: sorted(
            {dt.date(): cloud for dt, cloud in tile_data}.items()
        )
        for tile_id, tile_data in data_by_tile.items()
    }

    for best_date in best_day_dates:
        day_mosaic = mosaic.get(best_date, {})
        for tile_id, mosaic_cloud in day_mosaic.items():
            ax.scatter(
                [datetime.combine(best_date, datetime.min.time())],
                [mosaic_cloud],
                marker='X',
                s=150,
                c='red',
                zorder=10,
                edgecolors='darkred',
                linewidths=1,
            )

    ax.scatter([], [], marker='X', s=150, c='red', edgecolors='darkred',
               linewidths=1, label='Best 5 days (lowest mosaic cloud)')

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cloud Cover (%)", fontsize=12)
    ax.set_title(f"Sentinel-2 {level} Cloud Cover by Tile Over Time", fontsize=14)

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


def process_level(level: str, start_date: date, end_date: date, requested_tiles: List[str],
                  use_cache: bool, output_prefix: str,
                  plot_average: bool = False,
                  utm_zone: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Process a single level (L1C or L2A): discovery, extraction, summary, CSV, and plot.
    Returns (num_products, num_success, num_failed)
    """
    print(f"\n{'#'*70}")
    print(f"# Processing {level}")
    print(f"{'#'*70}")

    # Apply UTM zone filter to the requested tile list up-front so that
    # discovery only searches for matching tiles and doesn't fetch extras.
    if utm_zone:
        filtered_tiles = [t for t in requested_tiles if tile_matches_utm_zone(t, utm_zone)]
        omitted = [t for t in requested_tiles if not tile_matches_utm_zone(t, utm_zone)]
        if omitted:
            print(f"\n  UTM zone filter '{utm_zone}': omitting {len(omitted)} tile(s): {', '.join(omitted)}")
        if not filtered_tiles:
            print(f"  No tiles remain after UTM zone filter '{utm_zone}'. Skipping {level}.")
            return 0, 0, 0
        requested_tiles = filtered_tiles

    # Phase 1: Discovery
    print(f"\n{'='*70}")
    print(f"Phase 1: Discovering {level} products...")
    print(f"{'='*70}\n")

    t_discovery_start = time.time()
    products = discover_products_parallel(start_date, end_date, requested_tiles, level, use_cache)
    t_discovery_end = time.time()

    print(f"\n  {level} Discovery complete: {len(products)} products found in {t_discovery_end - t_discovery_start:.1f}s")

    if not products:
        print(f"  No {level} products found.")
        return 0, 0, 0

    # Phase 2: Extraction
    print(f"\n{'='*70}")
    print(f"Phase 2: Extracting {level} cloud percentages from {len(products)} products...")
    print(f"{'='*70}\n")

    t_extraction_start = time.time()
    results, failures = extract_cloud_percentages_parallel(products, level, use_cache)
    t_extraction_end = time.time()

    print(f"\n  {level} Extraction complete in {t_extraction_end - t_extraction_start:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print(f"{level} Summary:")
    print(f"{'='*70}\n")

    print(f"Total {level} products discovered: {len(products)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {len(failures)}")

    if results:
        results_sorted = sorted(results, key=lambda x: x[0])

        print(f"\n{level} Results (sorted by product name):")
        print("-" * 90)
        for pid, cloud in results_sorted[:20]:
            print(f"{pid:80s} {cloud:6.2f}%")
        if len(results_sorted) > 20:
            print(f"  ... and {len(results_sorted) - 20} more")

        # Write to CSV
        csv_filename = f"{output_prefix}_{level}.csv"
        print(f"\nWriting {len(results_sorted)} {level} results to CSV: {csv_filename}")
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
        print(f"\n{level} Cloud cover statistics:")
        print(f"  Min: {min_cloud:.2f}%")
        print(f"  Max: {max_cloud:.2f}%")
        print(f"  Avg: {avg_cloud:.2f}%")

        # Phase 3: Plotting
        print(f"\n{'='*70}")
        print(f"Phase 3: Generating {level} plot...")
        print(f"{'='*70}\n")

        data_by_tile = organize_by_tile_and_date(results)
        print(f"Organized data for {len(data_by_tile)} tiles")

        # Print per-tile summary
        print(f"\n{level} Summary by Tile:")
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
        print_best_days(best_days, data_by_tile, level)

        # Generate plot
        plot_filename = f"{output_prefix}_{level}.png"
        plot_cloud_cover(data_by_tile, best_days, level, plot_filename, plot_average=plot_average)

    if failures:
        print(f"\n{level} Failed products ({len(failures)}):")
        print("-" * 90)
        for pid, error in failures[:10]:
            print(f"{pid}: {error[:70]}...")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more failures")

    return len(products), len(results), len(failures)


# ------------------------------------------------------------
# Restore-from-CSV mode
# ------------------------------------------------------------
def process_restore_csv(csv_file: str,
                        filter_start: Optional[date],
                        filter_end: Optional[date],
                        filter_tiles: Optional[List[str]],
                        output_prefix: str,
                        plot_average: bool = False,
                        utm_zone: Optional[str] = None):
    """
    Load results from a previously written CSV file, apply optional filters,
    then regenerate the plot (and print summary statistics).
    """
    print(f"\n{'='*70}")
    print(f"Restore-from-CSV mode")
    print(f"{'='*70}")
    print(f"  CSV file   : {csv_file}")
    print(f"  Date filter: {filter_start} → {filter_end if filter_end else '(all)'}")
    print(f"  Tile filter: {', '.join(filter_tiles) if filter_tiles else '(all)'}")
    print(f"  UTM zone   : {utm_zone if utm_zone else '(all)'}")

    level = infer_level_from_csv_filename(csv_file)
    print(f"  Inferred level: {level}")

    results = load_results_from_csv(csv_file,
                                    filter_start=filter_start,
                                    filter_end=filter_end,
                                    filter_tiles=filter_tiles or None,
                                    utm_zone=utm_zone or None)

    if not results:
        print("  No matching records found after filtering. Nothing to plot.")
        return

    # Stats
    cloud_values = [c for _, c in results]
    print(f"\n{level} Cloud cover statistics (filtered):")
    print(f"  Records  : {len(results)}")
    print(f"  Min      : {min(cloud_values):.2f}%")
    print(f"  Max      : {max(cloud_values):.2f}%")
    print(f"  Avg      : {sum(cloud_values)/len(cloud_values):.2f}%")

    data_by_tile = organize_by_tile_and_date(results)
    print(f"\nOrganized data for {len(data_by_tile)} tiles")

    # Per-tile summary
    print(f"\n{level} Summary by Tile (filtered):")
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
        date_range = (f"{min(tile_dates).strftime('%Y-%m-%d')} to "
                      f"{max(tile_dates).strftime('%Y-%m-%d')}")

        print(f"{tile_id:<10} {len(tile_data):>6} {tile_min:>7.2f}% {tile_max:>7.2f}% {tile_avg:>7.2f}% {date_range}")

    # Best days
    daily_totals = compute_daily_totals(data_by_tile)
    best_days = find_best_days(daily_totals, n=5)
    print_best_days(best_days, data_by_tile, level)

    # Plot
    plot_filename = f"{output_prefix}_{level}.png"
    print(f"\nGenerating plot → {plot_filename}")
    plot_cloud_cover(data_by_tile, best_days, level, plot_filename, plot_average=plot_average)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    global N_WORKERS, SINGLE_THREAD

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Parse arguments
    use_cache = True
    output_prefix = "cloud_cover_by_tile"
    query_l1 = False
    query_l2 = False
    plot_average = False
    restore_csv = None          # path to CSV file for --restore_csv
    utm_zone: Optional[str] = None
    args = []                   # positional args: dates + tile IDs

    for arg in sys.argv[1:]:
        if arg == "--no-cache":
            use_cache = False
        elif arg == "--single-thread":
            SINGLE_THREAD = True
        elif arg == "--L1":
            query_l1 = True
        elif arg == "--L2":
            query_l2 = True
        elif arg.startswith("--workers="):
            N_WORKERS = int(arg.split("=")[1])
        elif arg.startswith("--output="):
            output_prefix = arg.split("=")[1]
        elif arg == "--average":
            plot_average = True
        elif arg.startswith("--restore_csv="):
            restore_csv = arg.split("=", 1)[1]
        elif arg.startswith("--utm_zone="):
            utm_zone = arg.split("=", 1)[1].strip()
        else:
            args.append(arg)

    # ------------------------------------------------------------------
    # Restore-from-CSV mode
    # ------------------------------------------------------------------
    if restore_csv is not None:
        # Positional args in restore mode are all optional:
        #   [yyyymmdd_start] [yyyymmdd_end] [TILE_ID ...]
        # We try to parse the first two positional args as dates; anything
        # that doesn't look like a date is treated as a tile ID.
        filter_start: Optional[date] = None
        filter_end: Optional[date] = None
        filter_tiles: List[str] = []

        remaining = list(args)

        # Try to parse up to two leading dates
        parsed_dates: List[date] = []
        while remaining and len(parsed_dates) < 2:
            candidate = remaining[0]
            if re.match(r'^\d{8}$', candidate):
                try:
                    parsed_dates.append(parse_yyyymmdd(candidate))
                    remaining.pop(0)
                    continue
                except ValueError:
                    pass
            break

        if len(parsed_dates) == 2:
            filter_start, filter_end = parsed_dates[0], parsed_dates[1]
            if filter_start > filter_end:
                raise ValueError("Start date must be <= end date")
        elif len(parsed_dates) == 1:
            # Single date means start == end (one specific day)
            filter_start = filter_end = parsed_dates[0]

        # Anything left is tile IDs
        filter_tiles = remaining

        print(f"\n{'='*70}")
        print(f"Sentinel-2 Cloud Cover - Restore from CSV")
        print(f"{'='*70}")
        print(f"CSV file       : {restore_csv}")
        print(f"Date filter    : {filter_start} → {filter_end}")
        print(f"Tile filter    : {filter_tiles if filter_tiles else '(all)'}")
        print(f"UTM zone       : {utm_zone if utm_zone else '(all)'}")
        print(f"Plot average   : {plot_average}")
        print(f"Output prefix  : {output_prefix}")
        print(f"{'='*70}")

        process_restore_csv(
            csv_file=restore_csv,
            filter_start=filter_start,
            filter_end=filter_end if filter_end else None,
            filter_tiles=filter_tiles if filter_tiles else None,
            output_prefix=output_prefix,
            plot_average=plot_average,
            utm_zone=utm_zone or None,
        )
        return

    # ------------------------------------------------------------------
    # Normal (download) mode
    # ------------------------------------------------------------------
    if len(args) < 3:
        print("Error: Need at least start_date, end_date, and one tile ID")
        print(__doc__)
        sys.exit(1)

    # Default: query both if neither specified
    if not query_l1 and not query_l2:
        query_l1 = True
        query_l2 = True

    start_date = parse_yyyymmdd(args[0])
    end_date = parse_yyyymmdd(args[1])
    requested_tiles = args[2:]

    if start_date > end_date:
        raise ValueError("Start date must be <= end date")

    levels_to_query = []
    if query_l1:
        levels_to_query.append("L1C")
    if query_l2:
        levels_to_query.append("L2A")

    print(f"\n{'='*70}")
    print(f"Sentinel-2 Cloud Cover Extraction")
    print(f"{'='*70}")
    print(f"Requested tiles: {', '.join(requested_tiles)}")
    print(f"Date range: {start_date} → {end_date}")
    print(f"Levels: {', '.join(levels_to_query)}")
    print(f"GDAL available: {HAS_GDAL}")
    print(f"Matplotlib available: {HAS_MATPLOTLIB}")
    print(f"Workers: {N_WORKERS}")
    print(f"Single-thread mode: {SINGLE_THREAD}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print(f"Output prefix: {output_prefix}")
    print(f"Plot average: {plot_average}")
    print(f"UTM zone filter: {utm_zone if utm_zone else '(none)'}")
    print(f"{'='*70}")

    t_total_start = time.time()

    # Process each level
    total_stats = {}
    for level in levels_to_query:
        num_products, num_success, num_failed = process_level(
            level, start_date, end_date, requested_tiles, use_cache, output_prefix,
            plot_average=plot_average,
            utm_zone=utm_zone or None,
        )
        total_stats[level] = (num_products, num_success, num_failed)

    t_total_end = time.time()

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}\n")

    for level, (num_products, num_success, num_failed) in total_stats.items():
        print(f"{level}: {num_products} discovered, {num_success} successful, {num_failed} failed")

    print(f"\nTotal time: {t_total_end - t_total_start:.1f}s")


if __name__ == "__main__":
    main()
