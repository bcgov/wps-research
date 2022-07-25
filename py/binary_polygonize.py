'''Not finished.

Polygonize a raster mask (values 0 or 1, data type and format otherwise not assumed)

Based on a module by Sybrand Strauss: 
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
    mem_driver = gdal.GetDriverByName('MEM')
    dataset = mem_driver.Create('memory', cols, rows, 1, gdal.GDT_Byte)
    dataset.SetProjection(projection)
    dataset.SetGeoTransform(geotransform)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    return dataset, band


def polygonize(geotiff_filename, json_filename):
    raster = gdal.Open(geotiff_filename, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    src_projection, geotransform = raster.GetProjection(), raster.GetGeoTransform()
    rows, cols = band.YSize, band.XSize
    
    # source coordinate reference system
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjectionRef())
    # as in: https://trac.osgeo.org/gdal/browser/trunk/gdal/swig/python/scripts/gdal_polygonize.py#L237
    
    # generate mask data
    mask_data = np.where(band.ReadAsArray() == 0, False, True)
    mask_ds, mask_band = create_in_memory_band(mask_data, cols, rows, src_projection, geotransform)

    # Create a GeoJSON layer.
    geojson_driver = ogr.GetDriverByName('ESRI Shapefile') #GeoJSON')
    dst_ds = geojson_driver.CreateDataSource(json_filename)
    dst_layer = dst_ds.CreateLayer('fire', srs)
    field_name = ogr.FieldDefn("fire", ogr.OFTInteger)
    field_name.SetWidth(24)
    dst_layer.CreateField(field_name)
    gdal.Polygonize(band, mask_band, dst_layer, 0, [], callback=None)  # polygonize
    dst_ds.FlushCache()
    del dst_ds, raster, mask_ds # print(f'{json_filename} written')

polygonize(args[1], args[1] + '.shp')
