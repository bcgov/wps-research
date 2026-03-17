#!/usr/bin/env python
"""
viirs/utils/rasterize_batch.py

Batch rasterize shapefiles onto a reference Sentinel-2 ENVI image.

Walks a directory tree recursively to find ALL .shp files, buffers each
fire point, and rasterizes onto the reference image grid. Output .bin/.hdr
files are written to a single flat output directory.

Also accepts a single .shp file instead of a directory.

Usage:
    # Single shapefile
    python rasterize_batch.py fire_pixels.shp /path/to/sentinel2.bin /output/dir

    # Directory (recursively finds all .shp files)
    python rasterize_batch.py /data/bill/viirs/VNP14IMG /path/to/sentinel2.bin /output/dir

    # With options
    python rasterize_batch.py /data/viirs /path/to/sentinel2.bin /output/dir --buffer 200 -w 4

Output:
    <output_dir>/
        <shapefile_basename>.bin
        <shapefile_basename>.hdr
        ...
"""

import argparse
import sys
import os
import glob
from osgeo import ogr, gdal, osr
from multiprocessing import Pool
from functools import partial


def find_shapefiles(input_path):
    """
    Resolve input to a list of .shp file paths.

    - If input_path is a .shp file, return it as a single-element list.
    - If input_path is a directory, recursively find ALL .shp files inside.
    """
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.shp'):
            return [input_path]
        else:
            sys.exit(f"Error: {input_path} is not a .shp file.")

    elif os.path.isdir(input_path):
        shp_files = sorted(
            glob.glob(os.path.join(input_path, '**', '*.shp'), recursive=True)
        )
        return shp_files

    else:
        sys.exit(f"Error: {input_path} is not a file or directory.")


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
    mem_driver = ogr.GetDriverByName('MEM')
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
        description='Batch rasterize shapefiles onto a Sentinel-2 reference image.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single shapefile
  python rasterize_batch.py fire.shp ref.bin output/

  # Directory (finds all .shp recursively)
  python rasterize_batch.py /data/viirs ref.bin output/

  # With buffer and workers
  python rasterize_batch.py /data/viirs ref.bin output/ --buffer 200 -w 8
        """,
    )
    parser.add_argument('input',
                        help='Single .shp file OR directory to recursively scan for .shp files')
    parser.add_argument('ref_image',
                        help='Reference Sentinel-2 ENVI image (.bin)')
    parser.add_argument('output_dir',
                        help='Output directory for rasterized .bin/.hdr files')
    parser.add_argument('--buffer', type=float, default=375.0,
                        help='Buffer radius in meters around each fire point (default: 375m)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    # Validate reference image
    if not os.path.exists(args.ref_image):
        sys.exit(f"Error: Reference image not found: {args.ref_image}")

    # Find shapefiles — works for a single .shp or a directory
    shp_files = find_shapefiles(args.input)

    if not shp_files:
        print(f"No .shp files found in: {args.input}")
        sys.exit(0)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Found {len(shp_files)} shapefile(s)")
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