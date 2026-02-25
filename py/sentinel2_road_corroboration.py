#!/usr/bin/env python3
"""
Sentinel Road Corroboration Tool
=================================
Replicates the ArcGIS-based workflow described in
"BC PODS: Scoping – road classification using Sentinel imagery"
by Nick Walsworth, PFC.

Given a Sentinel-2 image mosaic (TIF) and a DGTL roads geodatabase (.gdb),
this tool:
  1. Applies a high-pass filter to detect bright edges (road signatures).
  2. Buffers each road segment from the DGTL layer.
  3. Computes mean edge intensity within each buffer zone.
  4. Classifies road segments into confidence tiers:
       Very Good / Good / Marginal / Likely No Road
  5. Exports classified roads as GeoPackage and an edge raster as GeoTIFF.

Requirements:
    pip install rasterio fiona geopandas numpy scipy shapely pyproj

Usage:
    python sentinel_road_corroboration.py \\
        --image   mosaic.tif \\
        --roads   dgtl_roads.gdb \\
        --layer   DGTL_ROAD_LAYER_NAME \\
        --output  output_dir/

Run with --help for full option list.
"""

import argparse
import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np

# Suppress noisy warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger("sentinel_road_corr")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def list_gdb_layers(gdb_path: str):
    """Print available layers in a .gdb and exit."""
    import fiona
    layers = fiona.listlayers(gdb_path)
    print(f"\nLayers in {gdb_path}:")
    for i, name in enumerate(layers, 1):
        print(f"  {i}. {name}")
    print()


# ---------------------------------------------------------------------------
# Step 1 – Load & prepare the image
# ---------------------------------------------------------------------------

def load_image(image_path: str, band_index: int = 1):
    """
    Load a single band from the image mosaic.
    Returns the 2-D numpy array, profile (metadata), and transform.
    """
    import rasterio

    logger.info("Loading image: %s (band %d)", image_path, band_index)
    with rasterio.open(image_path) as src:
        data = src.read(band_index).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Mask nodata
    if nodata is not None:
        data[data == nodata] = np.nan

    logger.info("  Image shape: %s, CRS: %s", data.shape, crs)
    return data, profile, transform, crs


# ---------------------------------------------------------------------------
# Step 2 – High-pass filter to extract edges
# ---------------------------------------------------------------------------

def highpass_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 high-pass (Laplacian-style) filter identical to
    ArcGIS Spatial Analyst > Neighborhood > Filter (HIGH PASS).

    The ArcGIS high-pass kernel is:
        -1 -1 -1
        -1  8 -1
        -1 -1 -1
    which emphasises edges / rapid intensity transitions.
    """
    from scipy.ndimage import convolve

    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ], dtype=np.float32)

    logger.info("Applying 3×3 high-pass filter …")

    # Replace NaN with 0 for convolution, then restore
    mask = np.isnan(image)
    img = np.where(mask, 0.0, image)

    edges = convolve(img, kernel, mode="constant", cval=0.0)
    edges = np.abs(edges)  # We want magnitude of edge response
    edges[mask] = np.nan

    logger.info("  Edge range: %.1f – %.1f", np.nanmin(edges), np.nanmax(edges))
    return edges


# ---------------------------------------------------------------------------
# Step 3 – Threshold edges (SetNull equivalent)
# ---------------------------------------------------------------------------

def threshold_edges(edges: np.ndarray, percentile: float = 70.0,
                    absolute_threshold: float | None = None) -> np.ndarray:
    """
    Null out weak edges below a threshold.
    - If absolute_threshold is given, use it directly.
    - Otherwise, compute the threshold as a percentile of non-zero edge values.

    This mirrors the ArcGIS step:
        SetNull(edges < threshold, edges)
    """
    if absolute_threshold is not None:
        thresh = absolute_threshold
    else:
        valid = edges[~np.isnan(edges) & (edges > 0)]
        if valid.size == 0:
            logger.warning("No valid edge pixels – returning all NaN")
            return np.full_like(edges, np.nan)
        thresh = float(np.percentile(valid, percentile))

    logger.info("Edge threshold: %.2f  (percentile=%.0f)", thresh, percentile)

    result = edges.copy()
    result[result < thresh] = np.nan
    n_valid = int(np.count_nonzero(~np.isnan(result)))
    logger.info("  Retained %s edge pixels (%.1f%%)",
                f"{n_valid:,}", 100.0 * n_valid / result.size)
    return result


# ---------------------------------------------------------------------------
# Step 4 – Load & buffer road segments
# ---------------------------------------------------------------------------

def load_and_buffer_roads(gdb_path: str, layer: str, crs,
                          buffer_m: float, clip_bounds=None,
                          id_field: str | None = None):
    """
    Load roads from .gdb, reproject to target CRS, clip to bounds,
    add a unique ID, and create buffered polygons.

    Returns (roads_gdf, buffered_gdf) both in target CRS.
    """
    import geopandas as gpd

    logger.info("Loading roads: %s [%s]", gdb_path, layer)
    roads = gpd.read_file(gdb_path, layer=layer)
    logger.info("  %d features, CRS: %s", len(roads), roads.crs)

    # Reproject if needed
    if roads.crs is None:
        logger.warning("  Roads have no CRS – assuming same as image")
    elif not roads.crs.equals(crs):
        logger.info("  Reprojecting roads to %s", crs)
        roads = roads.to_crs(crs)

    # Add unique ID if needed
    if id_field and id_field in roads.columns:
        roads["_uid"] = roads[id_field]
    elif "OBJECTID" in roads.columns:
        roads["_uid"] = roads["OBJECTID"]
    else:
        roads["_uid"] = range(1, len(roads) + 1)

    # Clip to image bounds if provided
    if clip_bounds is not None:
        from shapely.geometry import box
        bbox = box(*clip_bounds)
        pre = len(roads)
        roads = roads[roads.intersects(bbox)].copy()
        roads["geometry"] = roads.geometry.intersection(bbox)
        roads = roads[~roads.is_empty]
        logger.info("  Clipped to image extent: %d → %d features", pre, len(roads))

    if len(roads) == 0:
        logger.error("No road features remain after clipping – check extents / CRS")
        sys.exit(1)

    # Buffer
    logger.info("  Buffering roads by %.1f m …", buffer_m)
    buffered = roads.copy()
    buffered["geometry"] = roads.geometry.buffer(buffer_m)
    buffered = buffered[~buffered.is_empty]

    return roads, buffered


# ---------------------------------------------------------------------------
# Step 5 – Zonal statistics: mean edge per buffered road
# ---------------------------------------------------------------------------

def zonal_edge_stats(buffered_gdf, edge_raster: np.ndarray,
                     transform, chunk_size: int = 5000):
    """
    Compute mean edge intensity within each buffered road polygon.
    Equivalent to ArcGIS Spatial Analyst > Zonal Statistics (MEAN).

    Works in chunks for memory efficiency on large datasets.
    """
    from rasterio.features import geometry_mask
    import geopandas as gpd

    logger.info("Computing zonal statistics for %d buffered segments …",
                len(buffered_gdf))

    means = []
    counts = []
    uids = buffered_gdf["_uid"].values
    geoms = buffered_gdf.geometry.values
    h, w = edge_raster.shape

    total = len(geoms)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        logger.debug("  Processing segments %d–%d / %d", start + 1, end, total)

        for idx in range(start, end):
            geom = geoms[idx]
            if geom is None or geom.is_empty:
                means.append(np.nan)
                counts.append(0)
                continue

            # Create a mask for this single geometry
            try:
                mask = geometry_mask(
                    [geom],
                    out_shape=(h, w),
                    transform=transform,
                    invert=True,   # True inside polygon
                    all_touched=True,
                )
            except Exception:
                means.append(np.nan)
                counts.append(0)
                continue

            vals = edge_raster[mask]
            valid = vals[~np.isnan(vals)]
            if valid.size > 0:
                means.append(float(np.mean(valid)))
                counts.append(int(valid.size))
            else:
                means.append(np.nan)
                counts.append(0)

        pct = 100.0 * end / total
        logger.info("  … %.0f%% done (%d / %d)", pct, end, total)

    buffered_gdf = buffered_gdf.copy()
    buffered_gdf["edge_mean"] = means
    buffered_gdf["edge_count"] = counts
    return buffered_gdf


def zonal_edge_stats_fast(buffered_gdf, edge_raster: np.ndarray,
                          transform):
    """
    Faster alternative using rasterstats if available.
    Falls back to the manual approach otherwise.
    """
    try:
        from rasterstats import zonal_stats
        logger.info("Using rasterstats for fast zonal computation …")

        stats = zonal_stats(
            buffered_gdf.geometry,
            edge_raster,
            affine=transform,
            stats=["mean", "count"],
            nodata=np.nan,
            all_touched=True,
        )
        buffered_gdf = buffered_gdf.copy()
        buffered_gdf["edge_mean"] = [s["mean"] for s in stats]
        buffered_gdf["edge_count"] = [s["count"] for s in stats]
        return buffered_gdf

    except ImportError:
        logger.info("rasterstats not installed – using built-in method")
        return zonal_edge_stats(buffered_gdf, edge_raster, transform)


# ---------------------------------------------------------------------------
# Step 6 – Classify road segments
# ---------------------------------------------------------------------------

def classify_roads(roads_gdf, buffered_gdf, thresholds: tuple[float, ...] | None = None):
    """
    Join mean-edge values back to original road lines and classify:
        4 = Very Good  (strong edge evidence)
        3 = Good
        2 = Marginal
        1 = Likely No Road

    If thresholds is None, automatic quartile-based breaks are used.
    thresholds should be (t_low, t_mid, t_high) ascending.
    """
    import geopandas as gpd

    logger.info("Classifying road segments …")

    # Merge stats onto original road lines
    stats = buffered_gdf[["_uid", "edge_mean", "edge_count"]].copy()
    stats = stats.drop(columns=["geometry"], errors="ignore")
    roads = roads_gdf.merge(stats, on="_uid", how="left")

    valid_means = roads["edge_mean"].dropna()
    if valid_means.empty:
        logger.warning("All edge means are NaN – cannot classify")
        roads["road_class"] = 1
        roads["road_label"] = "Likely No Road"
        return roads

    if thresholds is None:
        q25, q50, q75 = np.nanpercentile(valid_means, [25, 50, 75])
        thresholds = (q25, q50, q75)
        logger.info("  Auto thresholds (25/50/75%%): %.2f / %.2f / %.2f",
                     *thresholds)
    else:
        logger.info("  User thresholds: %.2f / %.2f / %.2f", *thresholds)

    t_low, t_mid, t_high = thresholds

    def _classify(val):
        if np.isnan(val):
            return 1
        if val >= t_high:
            return 4
        if val >= t_mid:
            return 3
        if val >= t_low:
            return 2
        return 1

    roads["road_class"] = roads["edge_mean"].apply(_classify)
    label_map = {4: "Very Good", 3: "Good", 2: "Marginal", 1: "Likely No Road"}
    roads["road_label"] = roads["road_class"].map(label_map)

    # Summary
    for cls in sorted(label_map):
        n = int((roads["road_class"] == cls).sum())
        logger.info("    %s : %d segments", label_map[cls], n)

    return roads


# ---------------------------------------------------------------------------
# Step 7 – Save outputs
# ---------------------------------------------------------------------------

def save_edge_raster(edge_raster: np.ndarray, profile: dict,
                     output_path: str):
    """Save the thresholded edge image as a GeoTIFF."""
    import rasterio

    logger.info("Saving edge raster: %s", output_path)
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(np.where(np.isnan(edge_raster), np.nan, edge_raster), 1)


def save_classified_roads(roads_gdf, output_path: str):
    """Save classified roads as GeoPackage."""
    logger.info("Saving classified roads: %s", output_path)

    # Drop internal helper columns
    out = roads_gdf.copy()
    cols_to_drop = [c for c in out.columns if c.startswith("_")]
    out = out.drop(columns=cols_to_drop, errors="ignore")
    out.to_file(output_path, driver="GPKG")


def save_summary_csv(roads_gdf, output_path: str):
    """Save a lightweight CSV summary (no geometry)."""
    logger.info("Saving summary CSV: %s", output_path)
    cols = [c for c in roads_gdf.columns if c != "geometry"]
    roads_gdf[cols].to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Execute the full road corroboration pipeline."""
    import rasterio

    os.makedirs(args.output, exist_ok=True)
    prefix = args.prefix or Path(args.image).stem

    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    image, profile, transform, crs = load_image(args.image, args.band)

    # ------------------------------------------------------------------
    # 2. High-pass filter
    # ------------------------------------------------------------------
    edges = highpass_filter(image)

    # ------------------------------------------------------------------
    # 3. Threshold edges
    # ------------------------------------------------------------------
    edge_thresh = threshold_edges(
        edges,
        percentile=args.edge_percentile,
        absolute_threshold=args.edge_threshold,
    )

    # Save intermediate edge raster
    edge_path = os.path.join(args.output, f"{prefix}_edges_highpass.tif")
    save_edge_raster(edge_thresh, profile, edge_path)

    # ------------------------------------------------------------------
    # 4. Load & buffer roads
    # ------------------------------------------------------------------
    with rasterio.open(args.image) as src:
        img_bounds = src.bounds  # (left, bottom, right, top)

    roads, buffered = load_and_buffer_roads(
        gdb_path=args.roads,
        layer=args.layer,
        crs=crs,
        buffer_m=args.buffer,
        clip_bounds=(img_bounds.left, img_bounds.bottom,
                     img_bounds.right, img_bounds.top),
        id_field=args.id_field,
    )

    # ------------------------------------------------------------------
    # 5. Zonal statistics
    # ------------------------------------------------------------------
    buffered = zonal_edge_stats_fast(buffered, edge_thresh, transform)

    # ------------------------------------------------------------------
    # 6. Classify
    # ------------------------------------------------------------------
    if args.thresholds:
        t = tuple(float(x) for x in args.thresholds.split(","))
        if len(t) != 3:
            logger.error("--thresholds must be 3 comma-separated values")
            sys.exit(1)
    else:
        t = None

    classified = classify_roads(roads, buffered, thresholds=t)

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    roads_path = os.path.join(args.output, f"{prefix}_classified_roads.gpkg")
    save_classified_roads(classified, roads_path)

    csv_path = os.path.join(args.output, f"{prefix}_road_summary.csv")
    save_summary_csv(classified, csv_path)

    logger.info("=" * 60)
    logger.info("Pipeline complete.  Outputs in: %s", args.output)
    logger.info("  Edge raster  : %s", edge_path)
    logger.info("  Roads (GPKG) : %s", roads_path)
    logger.info("  Summary CSV  : %s", csv_path)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Sentinel Road Corroboration – classify DGTL roads by "
                    "edge evidence in Sentinel-2 imagery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
--------
  # Basic run with defaults:
  python sentinel_road_corroboration.py \\
      --image  KamloopsRed_BCAlbers_full.tif \\
      --roads  DGTL_roads.gdb \\
      --layer  TRANSPORT_LINE \\
      --output results/

  # Custom buffer, threshold, and classification breaks:
  python sentinel_road_corroboration.py \\
      --image  mosaic.tif \\
      --roads  roads.gdb \\
      --layer  TRANSPORT_LINE \\
      --buffer 40 \\
      --edge-percentile 65 \\
      --thresholds 5.0,15.0,30.0 \\
      --output results/

  # List layers in a .gdb:
  python sentinel_road_corroboration.py \\
      --roads  roads.gdb --list-layers

WORKFLOW
--------
  This program replicates the following ArcGIS Pro workflow using
  open-source Python libraries:

  1. Load a Sentinel-2 red-band mosaic (already projected, 8-bit).
  2. High-pass filter (3×3 Laplacian) → bright edge detection.
  3. Threshold edges (SetNull equivalent) to retain only strong edges.
  4. Load DGTL roads from .gdb, clip to image extent, buffer segments.
  5. Zonal Statistics – compute mean edge intensity per buffer zone.
  6. Classify: Very Good / Good / Marginal / Likely No Road.
  7. Export classified road vectors (.gpkg) + edge raster (.tif).

OUTPUT CLASSES
--------------
  4 = Very Good    – Strong edge evidence; road clearly visible.
  3 = Good         – Moderate edge evidence.
  2 = Marginal     – Weak evidence; road may exist.
  1 = Likely No Road – No meaningful edge evidence.
""",
    )

    p.add_argument("--image", "-i",
                    help="Path to Sentinel-2 mosaic GeoTIFF (projected, single or multi-band).")
    p.add_argument("--roads", "-r", required=True,
                    help="Path to DGTL roads geodatabase (.gdb) or GeoPackage.")
    p.add_argument("--layer", "-l",
                    help="Layer name within the .gdb (use --list-layers to inspect).")
    p.add_argument("--output", "-o", default="output",
                    help="Output directory (default: ./output).")

    p.add_argument("--band", type=int, default=1,
                    help="Band index to use from the image (default: 1 = red band).")
    p.add_argument("--buffer", type=float, default=30.0,
                    help="Buffer distance in metres around each road segment (default: 30).")
    p.add_argument("--edge-percentile", type=float, default=70.0,
                    help="Percentile for automatic edge threshold (default: 70).")
    p.add_argument("--edge-threshold", type=float, default=None,
                    help="Absolute edge threshold (overrides --edge-percentile).")
    p.add_argument("--thresholds",
                    help="Classification break values as 3 comma-separated numbers "
                         "(low,mid,high).  E.g. '5.0,15.0,30.0'.  "
                         "Default: automatic quartile breaks.")
    p.add_argument("--id-field",
                    help="Unique ID field in the roads layer (default: OBJECTID or auto).")
    p.add_argument("--prefix",
                    help="Output filename prefix (default: derived from image name).")

    p.add_argument("--list-layers", action="store_true",
                    help="List available layers in the .gdb and exit.")
    p.add_argument("--verbose", "-v", action="store_true",
                    help="Enable debug logging.")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Quick mode: just list layers
    if args.list_layers:
        list_gdb_layers(args.roads)
        return

    # Validate required args
    if not args.image:
        parser.error("--image is required (unless using --list-layers)")
    if not args.layer:
        parser.error("--layer is required (use --list-layers to see options)")
    if not os.path.isfile(args.image):
        parser.error(f"Image not found: {args.image}")
    if not os.path.exists(args.roads):
        parser.error(f"Roads database not found: {args.roads}")

    run_pipeline(args)


if __name__ == "__main__":
    main()


