#!/usr/bin/env python3
"""
CDSE Mosaic Downloader
======================
Automated tiled download of Sentinel-1 Monthly Mosaics and Sentinel-2 Quarterly
Cloudless Mosaics from Copernicus Data Space Ecosystem (CDSE).

Supports two download methods:
  1. Sentinel Hub Process API (tiled 2500x2500 px, merged into one GeoTIFF)
  2. STAC + COG direct access (reads COGs remotely, clips & merges to shapefile extent)

All intermediate tile files and the final merged product are written to the
current working directory and are preserved after the run completes.

Requirements:
    pip install sentinelhub geopandas rasterio shapely pystac-client numpy

Setup:
    1. Register at https://dataspace.copernicus.eu
    2. Create OAuth client at https://shapps.dataspace.copernicus.eu/dashboard/#/
    3. Set environment variables or pass credentials:
         CDSE_CLIENT_ID=<your_client_id>
         CDSE_CLIENT_SECRET=<your_client_secret>

Usage examples:
    # Sentinel-2 cloudless mosaic via Sentinel Hub (tiled approach)
    python cdse_mosaic_downloader.py \\
        --shapefile area.shp \\
        --product s2-mosaic \\
        --date 2023-07-01 \\
        --output output_mosaic.tif \\
        --method sentinelhub

    # Sentinel-1 monthly mosaic via STAC+COG
    python cdse_mosaic_downloader.py \\
        --shapefile area.shp \\
        --product s1-mosaic \\
        --date 2023-08-01 \\
        --output output_mosaic.tif \\
        --method stac

    # Custom bands and resolution
    python cdse_mosaic_downloader.py \\
        --shapefile area.shp \\
        --product s2-mosaic \\
        --date 2023-04-01 \\
        --bands B04 B03 B02 \\
        --resolution 10 \\
        --output output.tif
"""

import argparse
import logging
import math
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS as RioCRS
from shapely.geometry import box, mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Product Configuration ───────────────────────────────────────────────────

S2_MOSAIC_COLLECTION_ID = "5460de54-082e-473a-b6ea-d5cbe3c17cca"
S2_MOSAIC_STAC_COLLECTION = "sentinel-2-global-mosaics"
S2_MOSAIC_DEFAULT_BANDS = ["B04", "B03", "B02"]
S2_MOSAIC_RESOLUTION = 10

S1_MOSAIC_COLLECTION_ID = "3c662330-108b-4378-8899-525fd5a225cb"
S1_MOSAIC_STAC_COLLECTION = "sentinel-1-global-mosaics"
S1_MOSAIC_DEFAULT_BANDS = ["VV", "VH"]
S1_MOSAIC_RESOLUTION = 10

CDSE_SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

MAX_TILE_PX = 2500


# ─── Progress Tracking ───────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "--:--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m:02d}m {s:02d}s"


def _fmt_bytes(nbytes: float) -> str:
    if nbytes < 1024:
        return f"{nbytes:.0f} B"
    elif nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 ** 3:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    else:
        return f"{nbytes / 1024 ** 3:.2f} GB"


def _fmt_rate(bytes_per_sec: float) -> str:
    if bytes_per_sec <= 0:
        return "-- MB/s"
    elif bytes_per_sec < 1024:
        return f"{bytes_per_sec:.0f} B/s"
    elif bytes_per_sec < 1024 ** 2:
        return f"{bytes_per_sec / 1024:.1f} KB/s"
    else:
        return f"{bytes_per_sec / 1024 ** 2:.2f} MB/s"


def _read_rx_bytes() -> int:
    """
    Read total received bytes across all non-loopback network interfaces
    from /sys/class/net/*/statistics/rx_bytes.  Returns 0 on failure.
    """
    total = 0
    try:
        net_dir = Path("/sys/class/net")
        if not net_dir.exists():
            return 0
        for iface in net_dir.iterdir():
            if iface.name == "lo":
                continue
            rx_file = iface / "statistics" / "rx_bytes"
            if rx_file.exists():
                total += int(rx_file.read_text().strip())
    except (OSError, ValueError):
        pass
    return total


def _progress_bar(fraction: float, width: int = 30) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(width * fraction)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fraction * 100:5.1f}%"


class ProgressTracker:
    """
    Tracks download progress with live network throughput and global ETA.

    - Background thread refreshes every ~1.5 s
    - Live download rate: sampled from network interface rx_bytes counters
    - Global average rate: computed from actual completed tile file sizes
    - ETA: projected from rolling average of step durations
    """

    def __init__(self, total_steps: int, step_label: str = "tile"):
        self.total_steps = total_steps
        self.step_label = step_label
        self.completed_steps = 0
        self.start_time = time.time()
        self.step_durations: list[float] = []
        self.step_bytes: list[int] = []
        self.total_bytes: int = 0
        self._current_step_start: float = 0.0
        self._spinner_idx = 0
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        self._lock = threading.Lock()
        # Network sampling state
        self._last_rx_bytes: int = _read_rx_bytes()
        self._last_rx_time: float = time.time()
        self._live_rate: float = 0.0  # bytes/sec from NIC sampling

    # ── internal helpers ──

    def _avg_step_time(self) -> float:
        if not self.step_durations:
            return 0.0
        return sum(self.step_durations) / len(self.step_durations)

    def _global_rate(self) -> float:
        """Average download rate (bytes/sec) from completed tile sizes."""
        total_time = sum(self.step_durations)
        if total_time <= 0:
            return 0.0
        return self.total_bytes / total_time

    def _sample_network_rate(self):
        """Sample rx_bytes from NIC counters to get live throughput."""
        now = time.time()
        rx = _read_rx_bytes()
        dt = now - self._last_rx_time
        if dt > 0 and rx >= self._last_rx_bytes:
            self._live_rate = (rx - self._last_rx_bytes) / dt
        self._last_rx_bytes = rx
        self._last_rx_time = now

    def _eta_remaining(self) -> float:
        remaining = self.total_steps - self.completed_steps
        avg = self._avg_step_time()
        if avg <= 0 and self._current_step_start > 0:
            avg = time.time() - self._current_step_start
        return remaining * avg

    def _print_status(self):
        self._sample_network_rate()
        elapsed = time.time() - self.start_time
        spinner = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
        self._spinner_idx += 1

        # Per-tile progress (time-estimated from rolling avg)
        avg = self._avg_step_time()
        if self._current_step_start > 0 and avg > 0:
            tile_elapsed = time.time() - self._current_step_start
            tile_frac = min(tile_elapsed / avg, 0.99)
            tile_eta = max(0, avg - tile_elapsed)
            tile_str = f"{_progress_bar(tile_frac, 20)} ~{_fmt_duration(tile_eta)} left"
        elif self._current_step_start > 0:
            tile_elapsed = time.time() - self._current_step_start
            tile_str = f"[{'▓' * 3}{'░' * 17}] {_fmt_duration(tile_elapsed)} so far"
        else:
            tile_str = ""

        # Live NIC rate vs global average
        live_str = _fmt_rate(self._live_rate)
        global_r = self._global_rate()
        avg_str = _fmt_rate(global_r) if global_r > 0 else "---"

        # Global progress
        global_frac = (
            self.completed_steps / self.total_steps if self.total_steps > 0 else 0
        )
        global_bar = _progress_bar(global_frac, 25)
        eta = self._eta_remaining()
        eta_str = _fmt_duration(eta) if self.completed_steps > 0 else "estimating..."

        line = (
            f"\r{spinner} {self.step_label.capitalize()} "
            f"{self.completed_steps + 1}/{self.total_steps}  "
            f"{tile_str}  │  "
            f"Net: {live_str}  Avg: {avg_str}  │  "
            f"Overall: {global_bar}  "
            f"{_fmt_bytes(self.total_bytes)}  "
            f"Elapsed: {_fmt_duration(elapsed)}  "
            f"ETA: {eta_str}"
        )
        sys.stderr.write(f"{line:<180}")
        sys.stderr.flush()

    # ── public interface ──

    def start_step(self, label: str = ""):
        with self._lock:
            self._current_step_start = time.time()
            # Reset NIC baseline so first sample of new tile is clean
            self._last_rx_bytes = _read_rx_bytes()
            self._last_rx_time = time.time()
        self._start_monitor()

    def finish_step(self, nbytes: int = 0):
        """
        Call when a tile/band download completes.
        Pass nbytes = file size on disk for accurate global rate tracking.
        """
        self._stop_monitor()
        with self._lock:
            duration = 0.0
            if self._current_step_start > 0:
                duration = time.time() - self._current_step_start
                self.step_durations.append(duration)
            if nbytes > 0:
                self.step_bytes.append(nbytes)
                self.total_bytes += nbytes
            self.completed_steps += 1
            self._current_step_start = 0.0

        elapsed = time.time() - self.start_time
        eta = self._eta_remaining()
        global_frac = (
            self.completed_steps / self.total_steps if self.total_steps > 0 else 0
        )

        # Per-tile rate from actual bytes and duration
        tile_rate = nbytes / duration if duration > 0 and nbytes > 0 else 0
        tile_info = f" {_fmt_bytes(nbytes)} @ {_fmt_rate(tile_rate)}" if tile_rate > 0 else ""
        # Rolling global rate
        global_r = self._global_rate()
        avg_info = f"  Avg: {_fmt_rate(global_r)}" if global_r > 0 else ""

        sys.stderr.write(
            f"\r✓ {self.step_label.capitalize()} "
            f"{self.completed_steps}/{self.total_steps} done "
            f"({_fmt_duration(duration)}{tile_info})  │  "
            f"Overall: {_progress_bar(global_frac, 25)}  "
            f"{_fmt_bytes(self.total_bytes)}{avg_info}  "
            f"Elapsed: {_fmt_duration(elapsed)}  "
            f"ETA: {_fmt_duration(eta) if eta > 0 else 'done'}"
            f"{'':20}\n"
        )
        sys.stderr.flush()

    def skip_step(self, reason: str = "skipped"):
        self._stop_monitor()
        with self._lock:
            self.completed_steps += 1
            self._current_step_start = 0.0

        sys.stderr.write(
            f"\r⊘ {self.step_label.capitalize()} "
            f"{self.completed_steps}/{self.total_steps} {reason}"
            f"{'':100}\n"
        )
        sys.stderr.flush()

    def finish_all(self):
        self._stop_monitor()
        total_elapsed = time.time() - self.start_time
        ok = len(self.step_durations)
        skipped = self.completed_steps - ok
        global_r = self._global_rate()

        sys.stderr.write(f"\n{'─' * 78}\n")
        sys.stderr.write(f"  Download complete: {ok} succeeded")
        if skipped:
            sys.stderr.write(f", {skipped} skipped")
        sys.stderr.write(f"  —  {_fmt_bytes(self.total_bytes)}")
        sys.stderr.write(f" in {_fmt_duration(total_elapsed)}")
        if global_r > 0:
            sys.stderr.write(f"  ({_fmt_rate(global_r)} avg)")
        if self.step_durations:
            sys.stderr.write(
                f"  —  {_fmt_duration(self._avg_step_time())}/{self.step_label}"
            )
        sys.stderr.write(f"\n{'─' * 78}\n\n")
        sys.stderr.flush()

    # ── background refresh thread ──

    def _start_monitor(self):
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

    def _stop_monitor(self):
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3)
        self._monitor_thread = None

    def _monitor_loop(self):
        while self._monitoring:
            self._print_status()
            time.sleep(1.5)


# ─── Sentinel Hub Process API Method ─────────────────────────────────────────

def download_sentinelhub(
    shapefile: str,
    product: str,
    date: str,
    bands: list[str],
    resolution: float,
    output: str,
    client_id: str,
    client_secret: str,
    tile_size_px: int = 2400,
    overlap_px: int = 10,
):
    """
    Download mosaic data using the Sentinel Hub Process API.

    Splits the shapefile bounding box into tiles of at most `tile_size_px`
    (default 2400, safely under the 2500 limit), downloads each tile as a
    GeoTIFF, then merges and clips to the shapefile boundary.
    """
    from sentinelhub import (
        SHConfig,
        DataCollection,
        SentinelHubRequest,
        BBox,
        CRS,
        MimeType,
    )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = CDSE_TOKEN_URL
    config.sh_base_url = CDSE_SH_BASE_URL

    gdf = gpd.read_file(shapefile)
    gdf_4326 = gdf.to_crs(epsg=4326)
    total_bounds = gdf_4326.total_bounds
    minx, miny, maxx, maxy = total_bounds

    log.info(f"Shapefile bounding box (EPSG:4326): {total_bounds}")

    if product == "s2-mosaic":
        collection_id = S2_MOSAIC_COLLECTION_ID
        default_res = S2_MOSAIC_RESOLUTION
    elif product == "s1-mosaic":
        collection_id = S1_MOSAIC_COLLECTION_ID
        default_res = S1_MOSAIC_RESOLUTION
    else:
        raise ValueError(f"Unknown product: {product}")

    res = resolution or default_res
    data_collection = DataCollection.define_byoc(collection_id)

    band_input = ", ".join(f'"{b}"' for b in bands)
    band_return = ", ".join(f"sample.{b}" for b in bands)
    n_bands = len(bands)

    if product == "s2-mosaic":
        evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{ bands: [{band_input}] }}],
        output: {{ bands: {n_bands}, sampleType: "FLOAT32" }}
    }};
}}
function evaluatePixel(sample) {{
    return [{band_return}];
}}
"""
    else:
        evalscript = f"""
//VERSION=3
function setup() {{
    return {{
        input: [{{ bands: [{band_input}], units: "LINEAR_POWER" }}],
        output: {{ bands: {n_bands}, sampleType: "FLOAT32" }}
    }};
}}
function evaluatePixel(sample) {{
    return [{band_return}];
}}
"""

    centre_lat = (miny + maxy) / 2
    deg_per_m_lat = 1.0 / 111_320
    deg_per_m_lon = 1.0 / (111_320 * math.cos(math.radians(centre_lat)))

    tile_width_deg = tile_size_px * res * deg_per_m_lon
    tile_height_deg = tile_size_px * res * deg_per_m_lat
    overlap_deg_x = overlap_px * res * deg_per_m_lon
    overlap_deg_y = overlap_px * res * deg_per_m_lat

    step_x = tile_width_deg - overlap_deg_x
    step_y = tile_height_deg - overlap_deg_y

    n_tiles_x = max(1, math.ceil((maxx - minx) / step_x))
    n_tiles_y = max(1, math.ceil((maxy - miny) / step_y))
    total_tiles = n_tiles_x * n_tiles_y

    log.info(
        f"Grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles "
        f"({tile_size_px}px @ {res}m resolution)"
    )

    pixels_per_tile = tile_size_px * tile_size_px
    pu_per_tile = (pixels_per_tile / 262_144) * (n_bands / 3)
    total_pu = pu_per_tile * total_tiles
    log.info(
        f"Estimated cost: ~{pu_per_tile:.1f} PU/tile, "
        f"~{total_pu:.0f} PU total (free tier: 10,000 PU/month)"
    )

    if total_pu > 10_000:
        log.warning(
            "⚠  Estimated PU exceeds free monthly quota! "
            "Consider using --method stac instead, or reduce the area."
        )

    # ── Download tiles (saved to current working directory) ──
    tile_files = []

    progress = ProgressTracker(total_tiles, step_label="tile")
    log.info("")

    for iy in range(n_tiles_y):
        for ix in range(n_tiles_x):
            tile_minx = minx + ix * step_x
            tile_miny = miny + iy * step_y
            tile_maxx = min(tile_minx + tile_width_deg, maxx + overlap_deg_x)
            tile_maxy = min(tile_miny + tile_height_deg, maxy + overlap_deg_y)

            tile_bbox = BBox(
                bbox=[tile_minx, tile_miny, tile_maxx, tile_maxy],
                crs=CRS.WGS84,
            )

            tile_w_m = (tile_maxx - tile_minx) / deg_per_m_lon
            tile_h_m = (tile_maxy - tile_miny) / deg_per_m_lat
            px_w = min(int(round(tile_w_m / res)), MAX_TILE_PX)
            px_h = min(int(round(tile_h_m / res)), MAX_TILE_PX)

            if px_w < 1 or px_h < 1:
                progress.skip_step("empty tile")
                continue

            tile_idx = iy * n_tiles_x + ix + 1
            progress.start_step(f"tile {tile_idx}")

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=data_collection,
                        time_interval=(date, date),
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=tile_bbox,
                size=(px_w, px_h),
                config=config,
            )

            try:
                data = request.get_data()
            except Exception as e:
                progress.skip_step(f"error: {e}")
                continue

            if not data or data[0] is None:
                progress.skip_step("no data")
                continue

            img = data[0]

            tile_path = f"tile_{iy:03d}_{ix:03d}.tif"
            transform = from_bounds(
                tile_minx, tile_miny, tile_maxx, tile_maxy, px_w, px_h
            )

            if img.ndim == 2:
                count = 1
                write_data = img[np.newaxis, :, :]
            else:
                count = img.shape[2]
                write_data = np.moveaxis(img, -1, 0)

            with rasterio.open(
                tile_path, "w", driver="GTiff",
                height=px_h, width=px_w, count=count,
                dtype=write_data.dtype, crs="EPSG:4326",
                transform=transform, compress="deflate",
            ) as dst:
                dst.write(write_data)

            tile_bytes = os.path.getsize(tile_path)
            tile_files.append(tile_path)
            progress.finish_step(nbytes=tile_bytes)

    progress.finish_all()

    if not tile_files:
        log.error("No tiles downloaded successfully. Exiting.")
        sys.exit(1)

    log.info(f"Downloaded {len(tile_files)} tiles. Merging ...")
    log.info(f"Tile files kept in: {os.getcwd()}")
    _merge_and_clip(tile_files, gdf_4326, output, n_bands=len(bands), bands=bands)

    log.info(f"Done! Output saved to: {output}")


# ─── STAC + COG Method ───────────────────────────────────────────────────────

def download_stac(
    shapefile: str,
    product: str,
    date: str,
    bands: list[str],
    resolution: float,
    output: str,
    client_id: str,
    client_secret: str,
):
    """
    Download mosaic data using the STAC API to discover COG tiles,
    then read them remotely via rasterio (HTTPS) and clip to the
    shapefile extent.

    No per-tile pixel limit, no Processing Unit cost.
    Counts against 12 TB/month transfer quota.
    """
    import pystac_client

    gdf = gpd.read_file(shapefile)
    gdf_4326 = gdf.to_crs(epsg=4326)
    total_bounds = gdf_4326.total_bounds
    minx, miny, maxx, maxy = total_bounds

    log.info(f"Shapefile bounding box (EPSG:4326): {total_bounds}")

    if product == "s2-mosaic":
        stac_collection = S2_MOSAIC_STAC_COLLECTION
        default_res = S2_MOSAIC_RESOLUTION
    elif product == "s1-mosaic":
        stac_collection = S1_MOSAIC_STAC_COLLECTION
        default_res = S1_MOSAIC_RESOLUTION
    else:
        raise ValueError(f"Unknown product: {product}")

    res = resolution or default_res

    log.info(f"Searching STAC catalog for collection '{stac_collection}' ...")
    client = pystac_client.Client.open(CDSE_STAC_URL)

    search = client.search(
        collections=[stac_collection],
        bbox=[minx, miny, maxx, maxy],
        datetime=date,
    )
    items = list(search.items())

    if not items:
        log.info("No exact match; trying wider date window ...")
        from datetime import datetime, timedelta

        dt = datetime.strptime(date, "%Y-%m-%d")
        start = (dt - timedelta(days=92)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=92)).strftime("%Y-%m-%d")
        search = client.search(
            collections=[stac_collection],
            bbox=[minx, miny, maxx, maxy],
            datetime=f"{start}/{end}",
        )
        items = list(search.items())

    log.info(f"Found {len(items)} STAC item(s).")

    if not items:
        log.error(
            "No items found. Check your date and collection. "
            "Sentinel-2 quarterly mosaics use dates like 2023-01-01, "
            "2023-04-01, 2023-07-01, 2023-10-01. "
            "Sentinel-1 monthly mosaics use the first of each month."
        )
        sys.exit(1)

    token = _get_cdse_token(client_id, client_secret)

    total_work = sum(
        1 for item in items for band in bands if band in item.assets
    )
    log.info(f"Downloading {total_work} band file(s) across {len(items)} item(s) ...")

    tile_files = []  # list of (band_name, filepath)

    gdal_env = {
        "GDAL_HTTP_RETRY_COUNT": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "50000000",
        "GDAL_HTTP_MAX_RETRY": "5",
    }
    if token:
        gdal_env["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"

    progress = ProgressTracker(total_work, step_label="band")
    log.info("")

    for i, item in enumerate(items):
        available_assets = list(item.assets.keys())

        for band in bands:
            if band not in item.assets:
                continue

            progress.start_step(f"{item.id}/{band}")

            asset = item.assets[band]
            href = asset.href

            if href.startswith("s3://eodata/"):
                href = href.replace(
                    "s3://eodata/",
                    "https://eodata.dataspace.copernicus.eu/",
                )

            tile_path = f"item_{i:03d}_{band}.tif"

            try:
                with rasterio.Env(**gdal_env):
                    with rasterio.open(href) as src:
                        from rasterio.windows import from_bounds as window_from_bounds

                        src_crs = src.crs
                        if src_crs and str(src_crs) != "EPSG:4326":
                            from pyproj import Transformer

                            transformer = Transformer.from_crs(
                                "EPSG:4326", src_crs, always_xy=True
                            )
                            cog_minx, cog_miny = transformer.transform(minx, miny)
                            cog_maxx, cog_maxy = transformer.transform(maxx, maxy)
                        else:
                            cog_minx, cog_miny = minx, miny
                            cog_maxx, cog_maxy = maxx, maxy

                        window = window_from_bounds(
                            cog_minx, cog_miny, cog_maxx, cog_maxy, src.transform,
                        )
                        window = window.intersection(
                            rasterio.windows.Window(0, 0, src.width, src.height)
                        )

                        if window.width < 1 or window.height < 1:
                            progress.skip_step("no overlap")
                            continue

                        data = src.read(1, window=window)
                        win_transform = src.window_transform(window)

                        with rasterio.open(
                            tile_path, "w", driver="GTiff",
                            height=data.shape[0], width=data.shape[1],
                            count=1, dtype=data.dtype, crs=src_crs,
                            transform=win_transform, compress="deflate",
                        ) as dst:
                            dst.write(data, 1)

                        tile_bytes = os.path.getsize(tile_path)
                        tile_files.append((band, tile_path))
                        progress.finish_step(nbytes=tile_bytes)

            except Exception as e:
                progress.skip_step(f"error: {e}")
                continue

    progress.finish_all()

    if not tile_files:
        log.error("No data downloaded. Exiting.")
        sys.exit(1)

    log.info(f"Tile files kept in: {os.getcwd()}")
    _merge_stac_tiles(tile_files, gdf_4326, bands, output)

    log.info(f"Done! Output saved to: {output}")


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _get_cdse_token(client_id: str, client_secret: str) -> Optional[str]:
    try:
        import requests

        response = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        response.raise_for_status()
        token = response.json().get("access_token")
        log.info("Successfully obtained CDSE access token.")
        return token
    except Exception as e:
        log.warning(
            f"Could not obtain CDSE token: {e}. "
            "COG access may fail if authentication is required."
        )
        return None


def _merge_stac_tiles(
    tile_files: list[tuple[str, str]],
    gdf_4326: gpd.GeoDataFrame,
    bands: list[str],
    output: str,
):
    """Merge per-band tile files, stack bands, clip to shapefile."""
    from collections import defaultdict

    log.info("Merging downloaded bands ...")

    band_files = defaultdict(list)
    for band, path in tile_files:
        band_files[band].append(path)

    merged_band_files = []

    for band in bands:
        if band not in band_files:
            log.warning(f"No data for band {band}")
            continue

        files = band_files[band]
        if len(files) == 1:
            merged_band_files.append((band, files[0]))
            continue

        datasets = [rasterio.open(f) for f in files]
        merged, merged_transform = merge(datasets)
        for ds in datasets:
            ds.close()

        merged_path = f"merged_{band}.tif"
        with rasterio.open(
            merged_path, "w", driver="GTiff",
            height=merged.shape[1], width=merged.shape[2], count=1,
            dtype=merged.dtype,
            crs=datasets[0].crs if datasets else "EPSG:4326",
            transform=merged_transform, compress="deflate",
        ) as dst:
            dst.write(merged[0], 1)

        merged_band_files.append((band, merged_path))

    if not merged_band_files:
        log.error("No merged bands available.")
        sys.exit(1)

    with rasterio.open(merged_band_files[0][1]) as ref:
        ref_profile = ref.profile.copy()
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    n_bands = len(merged_band_files)
    stacked = np.zeros((n_bands, ref_shape[0], ref_shape[1]), dtype=np.float32)

    for idx, (band, path) in enumerate(merged_band_files):
        with rasterio.open(path) as src:
            stacked[idx] = src.read(1)

    stacked_path = "stacked.tif"
    ref_profile.update(count=n_bands, dtype="float32", compress="deflate")

    with rasterio.open(stacked_path, "w", **ref_profile) as dst:
        dst.write(stacked)
        for idx, (band, _) in enumerate(merged_band_files):
            dst.set_band_description(idx + 1, band)

    _clip_to_shapefile(stacked_path, gdf_4326, output, ref_crs)


def _merge_and_clip(
    tile_files: list[str],
    gdf_4326: gpd.GeoDataFrame,
    output: str,
    n_bands: int,
    bands: list[str],
):
    """Merge downloaded tile GeoTIFFs and clip to shapefile boundary."""
    log.info("Merging tiles ...")
    datasets = [rasterio.open(f) for f in tile_files]
    merged, merged_transform = merge(datasets)

    src_crs = datasets[0].crs
    for ds in datasets:
        ds.close()

    merged_path = "merged.tif"

    with rasterio.open(
        merged_path, "w", driver="GTiff",
        height=merged.shape[1], width=merged.shape[2], count=merged.shape[0],
        dtype=merged.dtype, crs=src_crs,
        transform=merged_transform, compress="deflate",
    ) as dst:
        dst.write(merged)
        for idx, band in enumerate(bands):
            dst.set_band_description(idx + 1, band)

    log.info("Clipping to shapefile boundary ...")
    _clip_to_shapefile(merged_path, gdf_4326, output, src_crs)


def _clip_to_shapefile(
    raster_path: str,
    gdf_4326: gpd.GeoDataFrame,
    output: str,
    raster_crs,
):
    """Clip a raster to the shapefile geometry and save."""
    if raster_crs and str(raster_crs) != "EPSG:4326":
        gdf_clip = gdf_4326.to_crs(raster_crs)
    else:
        gdf_clip = gdf_4326

    geometries = [mapping(geom) for geom in gdf_clip.geometry]

    with rasterio.open(raster_path) as src:
        clipped, clipped_transform = rio_mask(
            src, geometries, crop=True, nodata=0
        )
        profile = src.profile.copy()

    profile.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=clipped_transform,
        compress="deflate",
        nodata=0,
    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(clipped)

    log.info(
        f"Clipped output: {clipped.shape[2]}x{clipped.shape[1]} px, "
        f"{clipped.shape[0]} band(s)"
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1/2 mosaics from CDSE, clipped to a shapefile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Product choices:
  s2-mosaic   Sentinel-2 Quarterly Cloudless Mosaics (10m, bands: B02-B12, SCL)
  s1-mosaic   Sentinel-1 Monthly Mosaics (10m, bands: VV, VH)

Date formats:
  s2-mosaic   Use start of quarter: 2023-01-01, 2023-04-01, 2023-07-01, 2023-10-01
  s1-mosaic   Use first of month:   2023-01-01, 2023-02-01, etc.

Methods:
  sentinelhub  Tiled download via Process API (2500px limit, costs PU)
  stac         Direct COG access via STAC catalog (no pixel limit, uses transfer quota)

Examples:
  %(prog)s --shapefile roi.shp --product s2-mosaic --date 2023-07-01 -o out.tif
  %(prog)s --shapefile roi.shp --product s1-mosaic --date 2023-08-01 -o out.tif --method stac
        """,
    )

    parser.add_argument(
        "--shapefile", "-s", required=True, help="Path to input shapefile (.shp)"
    )
    parser.add_argument(
        "--product", "-p", required=True,
        choices=["s2-mosaic", "s1-mosaic"], help="Product to download",
    )
    parser.add_argument(
        "--date", "-d", required=True, help="Date of the mosaic (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bands", "-b", nargs="+", default=None,
        help="Band names to download (default: product-specific)",
    )
    parser.add_argument(
        "--resolution", "-r", type=float, default=None,
        help="Pixel resolution in metres (default: native resolution)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output GeoTIFF path",
    )
    parser.add_argument(
        "--method", "-m", choices=["sentinelhub", "stac"], default="sentinelhub",
        help="Download method (default: sentinelhub)",
    )
    parser.add_argument(
        "--client-id", default=None,
        help="CDSE OAuth client ID (or set CDSE_CLIENT_ID env var)",
    )
    parser.add_argument(
        "--client-secret", default=None,
        help="CDSE OAuth client secret (or set CDSE_CLIENT_SECRET env var)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=2400,
        help="Tile size in pixels for sentinelhub method (default: 2400, max: 2500)",
    )

    args = parser.parse_args()

    cid = args.client_id or os.environ.get("CDSE_CLIENT_ID", "")
    csecret = args.client_secret or os.environ.get("CDSE_CLIENT_SECRET", "")

    if not cid or not csecret:
        log.error(
            "CDSE credentials required. Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET "
            "environment variables, or use --client-id and --client-secret flags.\n"
            "Get credentials at: https://shapps.dataspace.copernicus.eu/dashboard/#/"
        )
        sys.exit(1)

    if args.bands is None:
        if args.product == "s2-mosaic":
            args.bands = S2_MOSAIC_DEFAULT_BANDS
        else:
            args.bands = S1_MOSAIC_DEFAULT_BANDS

    log.info(f"Product:    {args.product}")
    log.info(f"Date:       {args.date}")
    log.info(f"Bands:      {args.bands}")
    log.info(f"Method:     {args.method}")
    log.info(f"Resolution: {args.resolution or 'native'}m")
    log.info(f"Output:     {args.output}")

    if args.method == "sentinelhub":
        download_sentinelhub(
            shapefile=args.shapefile,
            product=args.product,
            date=args.date,
            bands=args.bands,
            resolution=args.resolution,
            output=args.output,
            client_id=cid,
            client_secret=csecret,
            tile_size_px=min(args.tile_size, MAX_TILE_PX),
        )
    else:
        download_stac(
            shapefile=args.shapefile,
            product=args.product,
            date=args.date,
            bands=args.bands,
            resolution=args.resolution,
            output=args.output,
            client_id=cid,
            client_secret=csecret,
        )


if __name__ == "__main__":
    main()

