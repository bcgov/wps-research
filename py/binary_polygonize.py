'''
Not finished.

Polygonize a raster mask (values 0 or 1, data type and format otherwise not assumed)

Thanks to Sybrand Strauss for developing this module:
    https://github.com/bcgov/wps/blob/story/classify_hfi/api/scripts/polygonize_hfi.py#L52
'''
import os
import sys
import json
import tempfile
import numpy as np
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer, Proj
from shapely.ops import transform
from shapely.geometry import shape, mapping

from misc import exist, err, args
if len(args) < 2:
    err('python3 binary_polygonize.py [input raster mask file 1/0 values]')


def create_in_memory_band(data: np.ndarray, cols, rows, projection, geotransform):
    """ Create an in memory data band to represent a single raster layer.
    See https://gdal.org/user/raster_data_model.html#raster-band for a complete
    description of what a raster band is.
    """
    mem_driver = gdal.GetDriverByName('MEM')
    dataset = mem_driver.Create('memory', cols, rows, 1, gdal.GDT_Byte)
    dataset.SetProjection(projection)
    dataset.SetGeoTransform(geotransform)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    return dataset, band


def polygonize(geotiff_filename, geojson_filename):
    raster = gdal.Open(geotiff_filename, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)

    projection = raster.GetProjection()
    geotransform = raster.GetGeoTransform()
    rows = band.YSize
    cols = band.XSize

    # generate mask data
    mask_data = np.where(classification_data == 0, False, True)
    mask_ds, mask_band = create_in_memory_band(
        mask_data, cols, rows, projection, geotransform)

    # Create a GeoJSON layer.
    geojson_driver = ogr.GetDriverByName('GeoJSON')
    dst_ds = geojson_driver.CreateDataSource(geojson_filename)
    dst_layer = dst_ds.CreateLayer('fire')
    field_name = ogr.FieldDefn("fire", ogr.OFTInteger)
    field_name.SetWidth(24)
    dst_layer.CreateField(field_name)

    # Turn the rasters into polygons.
    gdal.Polygonize(band, mask_band, dst_layer, 0, [], callback=None)

    # Ensure that all data in the target dataset is written to disk.
    dst_ds.FlushCache()
    # Explicitly clean up (is this needed?)

    del dst_ds, raster, mask_ds
    print(f'{geojson_filename} written')


polygonize(args[1], args[1] + '.json')
