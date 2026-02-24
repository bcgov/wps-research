#!/usr/bin/env python3
"""
e.g. 
python cdse_mosaic_downloader.py -s sefc.shp -p s2-mosaic -d 2025-07-01 -b B02 B03 B04 B08 -r 10 -o sefc_s2_mosaic_2025Q3.tif # download 2025 mosaic over SEFC shapefile

CDSE Mosaic Downloader
======================
Automated tiled download of Sentinel-1 Monthly Mosaics and Sentinel-2 Quarterly
Cloudless Mosaics from Copernicus Data Space Ecosystem (CDSE).

Supports two download methods:
  1. Sentinel Hub Process API (tiled 2500x2500 px, merged into one GeoTIFF)
  2. STAC + COG direct access (reads COGs remotely, clips & merges to shapefile extent)

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
import tempfile
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

# Sentinel-2 Level 3 Quarterly Cloudless Mosaics (10m resolution, BYOC collection)
S2_MOSAIC_COLLECTION_ID = "5460de54-082e-473a-b6ea-d5cbe3c17cca"
S2_MOSAIC_STAC_COLLECTION = "sentinel-2-global-mosaics"
S2_MOSAIC_DEFAULT_BANDS = ["B04", "B03", "B02"]
S2_MOSAIC_RESOLUTION = 10  # metres

# Sentinel-1 IW Monthly Mosaics (BYOC collection)
S1_MOSAIC_COLLECTION_ID = "3c662330-108b-4378-8899-525fd5a225cb"
S1_MOSAIC_STAC_COLLECTION = "sentinel-1-global-mosaics"
S1_MOSAIC_DEFAULT_BANDS = ["VV", "VH"]
S1_MOSAIC_RESOLUTION = 10  # metres

# CDSE endpoints
CDSE_SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

# Sentinel Hub limits
MAX_TILE_PX = 2500  # max pixels per dimension for Process API


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

    # ── Configure credentials ──
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = CDSE_TOKEN_URL
    config.sh_base_url = CDSE_SH_BASE_URL

    # ── Read shapefile and get bounding box in EPSG:4326 ──
    gdf = gpd.read_file(shapefile)
    gdf_4326 = gdf.to_crs(epsg=4326)
    total_bounds = gdf_4326.total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = total_bounds

    log.info(f"Shapefile bounding box (EPSG:4326): {total_bounds}")

    # ── Select product collection ──
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

    # ── Build evalscript ──
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
    else:  # s1-mosaic
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

    # ── Calculate tile grid ──
    # Convert resolution from metres to approximate degrees at the centre latitude
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

    # ── Estimate PU cost ──
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

    # ── Download tiles ──
    tmpdir = tempfile.mkdtemp(prefix="cdse_tiles_")
    tile_files = []

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

            # Calculate pixel dimensions for this tile
            tile_w_m = (tile_maxx - tile_minx) / deg_per_m_lon
            tile_h_m = (tile_maxy - tile_miny) / deg_per_m_lat
            px_w = min(int(round(tile_w_m / res)), MAX_TILE_PX)
            px_h = min(int(round(tile_h_m / res)), MAX_TILE_PX)

            if px_w < 1 or px_h < 1:
                continue

            tile_idx = iy * n_tiles_x + ix + 1
            log.info(
                f"  Downloading tile {tile_idx}/{total_tiles} "
                f"({px_w}x{px_h} px) ..."
            )

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
                log.error(f"  Failed to download tile {tile_idx}: {e}")
                continue

            if not data or data[0] is None:
                log.warning(f"  Tile {tile_idx}: no data returned, skipping.")
                continue

            img = data[0]  # numpy array (height, width, bands)

            # Save as GeoTIFF
            tile_path = os.path.join(tmpdir, f"tile_{iy:03d}_{ix:03d}.tif")
            transform = from_bounds(
                tile_minx, tile_miny, tile_maxx, tile_maxy, px_w, px_h
            )

            # Handle shape: sentinelhub returns (H, W) or (H, W, B)
            if img.ndim == 2:
                count = 1
                write_data = img[np.newaxis, :, :]
            else:
                count = img.shape[2]
                write_data = np.moveaxis(img, -1, 0)  # (B, H, W)

            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                height=px_h,
                width=px_w,
                count=count,
                dtype=write_data.dtype,
                crs="EPSG:4326",
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(write_data)

            tile_files.append(tile_path)

    if not tile_files:
        log.error("No tiles downloaded successfully. Exiting.")
        sys.exit(1)

    log.info(f"Downloaded {len(tile_files)} tiles. Merging ...")

    # ── Merge tiles ──
    _merge_and_clip(tile_files, gdf_4326, output, n_bands=len(bands), bands=bands)

    # ── Cleanup ──
    for f in tile_files:
        os.remove(f)
    os.rmdir(tmpdir)

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
    then read them remotely via rasterio (HTTPS, no S3 auth needed for
    COG overview reads) and clip to the shapefile extent.

    This approach reads directly from the CDSE COGs using HTTP range
    requests — no per-tile pixel limit, and no Processing Unit cost.
    It does count against your monthly transfer quota (12 TB).
    """
    import pystac_client

    # ── Read shapefile ──
    gdf = gpd.read_file(shapefile)
    gdf_4326 = gdf.to_crs(epsg=4326)
    total_bounds = gdf_4326.total_bounds
    minx, miny, maxx, maxy = total_bounds

    log.info(f"Shapefile bounding box (EPSG:4326): {total_bounds}")

    # ── Select product ──
    if product == "s2-mosaic":
        stac_collection = S2_MOSAIC_STAC_COLLECTION
        default_res = S2_MOSAIC_RESOLUTION
    elif product == "s1-mosaic":
        stac_collection = S1_MOSAIC_STAC_COLLECTION
        default_res = S1_MOSAIC_RESOLUTION
    else:
        raise ValueError(f"Unknown product: {product}")

    res = resolution or default_res

    # ── Search STAC catalog ──
    log.info(f"Searching STAC catalog for collection '{stac_collection}' ...")
    client = pystac_client.Client.open(CDSE_STAC_URL)

    # Use the date as a point query — mosaics have a single date per composite
    search = client.search(
        collections=[stac_collection],
        bbox=[minx, miny, maxx, maxy],
        datetime=date,
    )
    items = list(search.items())

    if not items:
        # Try a wider date range (quarterly mosaics use start-of-quarter dates)
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

    # ── Get CDSE access token for authenticated COG access ──
    token = _get_cdse_token(client_id, client_secret)

    # ── Collect band URLs from items ──
    tmpdir = tempfile.mkdtemp(prefix="cdse_stac_")
    tile_files = []

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

    for i, item in enumerate(items):
        available_assets = list(item.assets.keys())
        log.info(f"Item {i}: {item.id} — assets: {available_assets}")

        for band in bands:
            if band not in item.assets:
                log.warning(
                    f"  Band '{band}' not found in item {item.id}. "
                    f"Available: {available_assets}"
                )
                continue

            asset = item.assets[band]
            href = asset.href

            # Convert s3:// URLs to HTTPS
            if href.startswith("s3://eodata/"):
                href = href.replace(
                    "s3://eodata/",
                    "https://eodata.dataspace.copernicus.eu/",
                )

            log.info(f"  Reading {band} from: {href}")

            tile_path = os.path.join(
                tmpdir, f"item_{i:03d}_{band}.tif"
            )

            try:
                with rasterio.Env(**gdal_env):
                    with rasterio.open(href) as src:
                        # Compute the window for our bounding box
                        from rasterio.windows import from_bounds as window_from_bounds

                        # Reproject bbox to the COG's native CRS if needed
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
                            cog_minx,
                            cog_miny,
                            cog_maxx,
                            cog_maxy,
                            src.transform,
                        )

                        # Clip window to valid range
                        window = window.intersection(
                            rasterio.windows.Window(
                                0, 0, src.width, src.height
                            )
                        )

                        if window.width < 1 or window.height < 1:
                            log.warning(
                                f"  No overlap for {band} in item {item.id}"
                            )
                            continue

                        data = src.read(1, window=window)
                        win_transform = src.window_transform(window)

                        with rasterio.open(
                            tile_path,
                            "w",
                            driver="GTiff",
                            height=data.shape[0],
                            width=data.shape[1],
                            count=1,
                            dtype=data.dtype,
                            crs=src_crs,
                            transform=win_transform,
                            compress="deflate",
                        ) as dst:
                            dst.write(data, 1)

                        tile_files.append((band, tile_path))

            except Exception as e:
                log.error(f"  Error reading {band} from {item.id}: {e}")
                continue

    if not tile_files:
        log.error("No data downloaded. Exiting.")
        sys.exit(1)

    # ── Merge bands per tile, then merge tiles ──
    _merge_stac_tiles(tile_files, gdf_4326, bands, output)

    # ── Cleanup ──
    for _, f in tile_files:
        if os.path.exists(f):
            os.remove(f)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    log.info(f"Done! Output saved to: {output}")


def _get_cdse_token(client_id: str, client_secret: str) -> Optional[str]:
    """Get an OAuth2 access token from CDSE."""
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
    """
    Merge per-band tile files:
    1. For each band, merge all tile files (from different STAC items) into one
    2. Stack bands into a single multi-band GeoTIFF
    3. Clip to shapefile geometry
    """
    from collections import defaultdict

    band_files = defaultdict(list)
    for band, path in tile_files:
        band_files[band].append(path)

    tmpdir = tempfile.mkdtemp(prefix="cdse_merge_")
    merged_band_files = []

    for band in bands:
        if band not in band_files:
            log.warning(f"No data for band {band}")
            continue

        files = band_files[band]
        if len(files) == 1:
            merged_band_files.append((band, files[0]))
            continue

        # Merge multiple tiles for this band
        datasets = [rasterio.open(f) for f in files]
        merged, merged_transform = merge(datasets)
        for ds in datasets:
            ds.close()

        merged_path = os.path.join(tmpdir, f"merged_{band}.tif")
        with rasterio.open(
            merged_path,
            "w",
            driver="GTiff",
            height=merged.shape[1],
            width=merged.shape[2],
            count=1,
            dtype=merged.dtype,
            crs=datasets[0].crs if datasets else "EPSG:4326",
            transform=merged_transform,
            compress="deflate",
        ) as dst:
            dst.write(merged[0], 1)

        merged_band_files.append((band, merged_path))

    if not merged_band_files:
        log.error("No merged bands available.")
        sys.exit(1)

    # Stack bands and clip
    # Read first band to get dimensions
    with rasterio.open(merged_band_files[0][1]) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)

    n_bands = len(merged_band_files)
    stacked = np.zeros((n_bands, ref_shape[0], ref_shape[1]), dtype=np.float32)

    for idx, (band, path) in enumerate(merged_band_files):
        with rasterio.open(path) as src:
            stacked[idx] = src.read(1)

    # Write stacked
    stacked_path = os.path.join(tmpdir, "stacked.tif")
    ref_profile.update(count=n_bands, dtype="float32", compress="deflate")

    with rasterio.open(stacked_path, "w", **ref_profile) as dst:
        dst.write(stacked)
        for idx, (band, _) in enumerate(merged_band_files):
            dst.set_band_description(idx + 1, band)

    # Clip to shapefile
    _clip_to_shapefile(stacked_path, gdf_4326, output, ref_crs)

    # Cleanup temp merged files
    for band, path in merged_band_files:
        if path.startswith(tmpdir) and os.path.exists(path):
            os.remove(path)
    if os.path.exists(stacked_path):
        os.remove(stacked_path)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


def _merge_and_clip(
    tile_files: list[str],
    gdf_4326: gpd.GeoDataFrame,
    output: str,
    n_bands: int,
    bands: list[str],
):
    """Merge downloaded tile GeoTIFFs and clip to shapefile boundary."""
    datasets = [rasterio.open(f) for f in tile_files]
    merged, merged_transform = merge(datasets)

    # Get CRS from first dataset
    src_crs = datasets[0].crs
    for ds in datasets:
        ds.close()

    # Write merged raster
    tmpdir = tempfile.mkdtemp(prefix="cdse_merged_")
    merged_path = os.path.join(tmpdir, "merged.tif")

    with rasterio.open(
        merged_path,
        "w",
        driver="GTiff",
        height=merged.shape[1],
        width=merged.shape[2],
        count=merged.shape[0],
        dtype=merged.dtype,
        crs=src_crs,
        transform=merged_transform,
        compress="deflate",
    ) as dst:
        dst.write(merged)
        for idx, band in enumerate(bands):
            dst.set_band_description(idx + 1, band)

    # Clip to shapefile
    _clip_to_shapefile(merged_path, gdf_4326, output, src_crs)

    os.remove(merged_path)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


def _clip_to_shapefile(
    raster_path: str,
    gdf_4326: gpd.GeoDataFrame,
    output: str,
    raster_crs,
):
    """Clip a raster to the shapefile geometry and save."""
    # Reproject shapefile to raster CRS if needed
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
        "--product",
        "-p",
        required=True,
        choices=["s2-mosaic", "s1-mosaic"],
        help="Product to download",
    )
    parser.add_argument(
        "--date",
        "-d",
        required=True,
        help="Date of the mosaic (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bands",
        "-b",
        nargs="+",
        default=None,
        help="Band names to download (default: product-specific)",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=None,
        help="Pixel resolution in metres (default: native resolution)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output GeoTIFF path",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["sentinelhub", "stac"],
        default="sentinelhub",
        help="Download method (default: sentinelhub)",
    )
    parser.add_argument(
        "--client-id",
        default=None,
        help="CDSE OAuth client ID (or set CDSE_CLIENT_ID env var)",
    )
    parser.add_argument(
        "--client-secret",
        default=None,
        help="CDSE OAuth client secret (or set CDSE_CLIENT_SECRET env var)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2400,
        help="Tile size in pixels for sentinelhub method (default: 2400, max: 2500)",
    )

    args = parser.parse_args()

    # Resolve credentials
    cid = args.client_id or os.environ.get("CDSE_CLIENT_ID", "")
    csecret = args.client_secret or os.environ.get("CDSE_CLIENT_SECRET", "")

    if not cid or not csecret:
        log.error(
            "CDSE credentials required. Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET "
            "environment variables, or use --client-id and --client-secret flags.\n"
            "Get credentials at: https://shapps.dataspace.copernicus.eu/dashboard/#/"
        )
        sys.exit(1)

    # Resolve default bands
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


