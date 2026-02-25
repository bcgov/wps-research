'''20260224 clip a GDB layer, to an extent given by a raster file.
'''
#!/usr/bin/env python3

import sys
import os
import argparse
from osgeo import gdal, ogr, osr


def clip_gdb_to_raster_extent(
    gdb_path,
    raster_path,
    layer_name,
    output_path,
    output_driver="GPKG",
    overwrite=False
):

    # ---- Open raster ----
    raster = gdal.Open(raster_path)
    if raster is None:
        raise RuntimeError(f"Could not open raster: {raster_path}")

    gt = raster.GetGeoTransform()

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * raster.RasterXSize
    ymin = ymax + gt[5] * raster.RasterYSize

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjection())

    # ---- Open GDB ----
    gdb = ogr.Open(gdb_path)
    if gdb is None:
        raise RuntimeError(f"Could not open GDB: {gdb_path}")

    layer = gdb.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"Layer '{layer_name}' not found in {gdb_path}")

    layer_srs = layer.GetSpatialRef()

    # ---- Reproject extent if CRS differ ----
    if not raster_srs.IsSame(layer_srs):
        transform = osr.CoordinateTransformation(raster_srs, layer_srs)

        # Transform all four corners to be safe
        corners = [
            transform.TransformPoint(xmin, ymin),
            transform.TransformPoint(xmin, ymax),
            transform.TransformPoint(xmax, ymin),
            transform.TransformPoint(xmax, ymax),
        ]

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]

        xmin_t = min(xs)
        xmax_t = max(xs)
        ymin_t = min(ys)
        ymax_t = max(ys)
    else:
        xmin_t, ymin_t, xmax_t, ymax_t = xmin, ymin, xmax, ymax

    # ---- Overwrite handling ----
    if overwrite and os.path.exists(output_path):
        driver = ogr.GetDriverByName(output_driver)
        driver.DeleteDataSource(output_path)

    # ---- Clip ----
    options = gdal.VectorTranslateOptions(
        format=output_driver,
        spatFilter=[xmin_t, ymin_t, xmax_t, ymax_t],
        layers=[layer_name]
    )

    result = gdal.VectorTranslate(
        destNameOrDestDS=output_path,
        srcDS=gdb_path,
        options=options
    )

    if result is None:
        raise RuntimeError("VectorTranslate failed.")

    print(f"Clipping complete: {output_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Clip a File Geodatabase layer to a raster extent."
    )

    parser.add_argument("gdb_path", help="Path to .gdb directory")
    parser.add_argument("raster_path", help="Path to raster file (any GDAL-supported format)")
    parser.add_argument("layer_name", help="Layer inside the geodatabase")
    parser.add_argument("output_path", help="Output vector file")
    parser.add_argument(
        "--driver",
        default="GPKG",
        help="Output driver (default: GPKG)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output"
    )

    args = parser.parse_args()

    try:
        clip_gdb_to_raster_extent(
            gdb_path=args.gdb_path,
            raster_path=args.raster_path,
            layer_name=args.layer_name,
            output_path=args.output_path,
            output_driver=args.driver,
            overwrite=args.overwrite
        )
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
