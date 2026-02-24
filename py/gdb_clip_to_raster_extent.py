'''20260224 clip a GDB layer, to an extent given by a raster file.
'''
from osgeo import gdal, ogr, osr
import os


def clip_gdb_to_raster_extent(
    gdb_path,
    raster_path,
    layer_name,
    output_path="clipped.gpkg",
    output_driver="GPKG"
):
    """
    Clip a File Geodatabase layer to the spatial extent of a raster.

    Parameters
    ----------
    gdb_path : str
        Path to .gdb directory
    raster_path : str
        Path to raster (any GDAL-supported format)
    layer_name : str
        Layer inside the geodatabase to clip
    output_path : str
        Output file path
    output_driver : str
        Output format (default GeoPackage)
    """

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
        raise RuntimeError(f"Layer '{layer_name}' not found in GDB")

    layer_srs = layer.GetSpatialRef()

    # ---- Reproject extent if CRS differ ----
    if not raster_srs.IsSame(layer_srs):
        transform = osr.CoordinateTransformation(raster_srs, layer_srs)

        # Transform bounding box corners
        ll = transform.TransformPoint(xmin, ymin)
        ur = transform.TransformPoint(xmax, ymax)

        xmin_t, ymin_t = ll[0], ll[1]
        xmax_t, ymax_t = ur[0], ur[1]
    else:
        xmin_t, ymin_t, xmax_t, ymax_t = xmin, ymin, xmax, ymax

    # ---- Perform clip using GDAL VectorTranslate ----
    options = gdal.VectorTranslateOptions(
        format=output_driver,
        spatFilter=[xmin_t, ymin_t, xmax_t, ymax_t],
        layerName=layer_name
    )

    gdal.VectorTranslate(
        destNameOrDestDS=output_path,
        srcDS=gdb_path,
        options=options
    )

    print("Clipping complete:", output_path)


