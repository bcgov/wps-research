#!/usr/bin/env python
"""
Batch rasterize VIIRS fire shapefiles onto a reference Sentinel-2 ENVI image.

Walks the VNP14IMG directory tree (organized by year/julian_day), finds all
shapefiles produced by vnp14_to_shp.py, buffers each fire point, and rasterizes
onto the reference image grid. Output .bin/.hdr files are written to a single
flat output directory with names matching the source shapefile.

Usage:
    python rasterize_batch.py <viirs_base_dir> <reference_image> <output_dir> [--buffer METERS]

Examples:
    python rasterize_batch.py /data/bill/viirs/VNP14IMG /path/to/sentinel2.bin /data/bill/viirs_rasters
    python rasterize_batch.py /data/bill/viirs/VNP14IMG /path/to/sentinel2.bin /data/bill/viirs_rasters --buffer 200
    python rasterize_batch.py /data/bill/viirs/VNP14IMG /path/to/sentinel2.bin /data/bill/viirs_rasters --workers 4

Directory structure expected:
    <viirs_base_dir>/
        2025/
            122/
                VIIRS_VNP14IMG_20250502T1024.shp
                ...
            123/
                VIIRS_VNP14IMG_20250503T0812.shp
                ...

Output:
    <output_dir>/
        VIIRS_VNP14IMG_20250502T1024.bin
        VIIRS_VNP14IMG_20250502T1024.hdr
        VIIRS_VNP14IMG_20250503T0812.bin
        VIIRS_VNP14IMG_20250503T0812.hdr
        ...
"""

import argparse
import sys
import os
import glob
from osgeo import ogr, gdal, osr
from multiprocessing import Pool
from functools import partial


def rasterize_shapefile(shp_path, ref_image, output_dir, buffer_m=375.0):
    """Rasterize a single shapefile onto the reference image grid.
    Returns output path or None."""

    basename = os.path.splitext(os.path.basename(shp_path))[0]
    out_bin = os.path.join(output_dir, f'{basename}.bin')

    if os.path.exists(out_bin):
        print(f"  Skipping (exists): {out_bin}")
        return out_bin

    print(f"  Rasterizing: {shp_path}")

    # Open reference image
    Image = gdal.Open(ref_image, gdal.GA_ReadOnly)
    if Image is None:
        print(f"  Error: Cannot open reference image: {ref_image}")
        return None

    # Open shapefile
    Shapefile = ogr.Open(shp_path)
    if Shapefile is None:
        print(f"  Error: Cannot open shapefile: {shp_path}")
        Image = None
        return None

    layer = Shapefile.GetLayer()

    # Check if layer has any features
    if layer.GetFeatureCount() == 0:
        print(f"  No features in {shp_path}. Skipping.")
        Shapefile = None
        Image = None
        return None

    # Create buffered version in memory
    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('')
    mem_layer = mem_ds.CreateLayer('buffered', layer.GetSpatialRef(), ogr.wkbPolygon)

    # Copy field definitions
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        mem_layer.CreateField(layer_defn.GetFieldDefn(i))

    # Buffer each point
    for feature in layer:
        geom = feature.GetGeometryRef()
        buffered = geom.Buffer(buffer_m)
        out_feat = ogr.Feature(mem_layer.GetLayerDefn())
        out_feat.SetGeometry(buffered)
        mem_layer.CreateFeature(out_feat)

    layer.ResetReading()

    # Create output raster
    Output = gdal.GetDriverByName('ENVI').Create(
        out_bin,
        Image.RasterXSize,
        Image.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform())

    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)

    # Rasterize
    gdal.RasterizeLayer(Output, [1], mem_layer, burn_values=[1.0], options=["ALL_TOUCHED=TRUE"])

    # Cleanup
    Band = None
    Output = None
    Image = None
    Shapefile = None
    mem_ds = None

    print(f"  +w {out_bin}")
    return out_bin


def _worker(shp_path, ref_image, output_dir, buffer_m):
    """Wrapper for multiprocessing."""
    try:
        return rasterize_shapefile(shp_path, ref_image, output_dir, buffer_m)
    except Exception as e:
        print(f"  Error processing {shp_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Batch rasterize VIIRS fire shapefiles onto a Sentinel-2 reference image.'
    )
    parser.add_argument('viirs_dir',
                        help='Base VIIRS directory (e.g. /data/bill/viirs/VNP14IMG)')
    parser.add_argument('ref_image',
                        help='Reference Sentinel-2 ENVI image (.bin)')
    parser.add_argument('output_dir',
                        help='Output directory for rasterized .bin/.hdr files (flat, no subdirs)')
    parser.add_argument('--buffer', type=float, default=375.0,
                        help='Buffer radius in meters around each fire point (default: 375m)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.viirs_dir):
        sys.exit(f"Error: Directory not found: {args.viirs_dir}")
    if not os.path.exists(args.ref_image):
        sys.exit(f"Error: Reference image not found: {args.ref_image}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all shapefiles under viirs_dir/YYYY/DDD/
    pattern = os.path.join(args.viirs_dir, '*', '*', 'VIIRS_VNP14IMG_*.shp')
    shp_files = sorted(glob.glob(pattern))

    if not shp_files:
        print(f"No shapefiles found matching: {pattern}")
        sys.exit(0)

    print(f"Found {len(shp_files)} shapefile(s) under {args.viirs_dir}")
    print(f"Reference image: {args.ref_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Buffer: {args.buffer}m")
    print(f"Workers: {args.workers}\n")

    if args.workers <= 1:
        results = []
        for shp in shp_files:
            r = rasterize_shapefile(shp, args.ref_image, args.output_dir, args.buffer)
            results.append(r)
    else:
        worker_fn = partial(_worker,
                            ref_image=args.ref_image,
                            output_dir=args.output_dir,
                            buffer_m=args.buffer)
        with Pool(processes=min(args.workers, len(shp_files))) as pool:
            results = pool.map(worker_fn, shp_files)

    written = [r for r in results if r is not None]
    print(f"\nDone. {len(written)}/{len(shp_files)} rasters written to {args.output_dir}")


if __name__ == '__main__':
    main()