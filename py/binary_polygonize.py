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

raster_fn = args[1]
geojson_filename = raster_fn + '_poly.json'  # output filename
kml_filename = raster_fn + '_poly.kml'
ofn = raster_fn + '.shp' # kml_filename

def _create_in_memory_band(data: np.ndarray, cols, rows, original): # projection, geotransform):
    """ Create an in memory data band to represent a single raster layer.
    See https://gdal.org/user/raster_data_model.html#raster-band for a complete
    description of what a raster band is.
    """

    driver = gdal.GetDriverByName('ENVI') #MEM')
    dataset = mem_driver.Create('memory', cols, rows, 1, gdal.GDT_Byte)
    #dataset.SetProjection(projection)
    #dataset.SetGeoTransform(geotransform)
    dataset = driver.CreateCopy(dst_filename, src_ds, 0)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    return dataset, band

def polygonize(raster_fn, geojson_filename):
    classification = gdal.Open(raster_fn, gdal.GA_ReadOnly)
    band = classification.GetRasterBand(1)
    classification_data = band.ReadAsArray()

    sr = None
    try:
        sr = osr.SpatialReference(wkt=classification.GetProjection()) #wkt=d2.GetProjection())
    except:
        driver2 = ogr.GetDriverByName('ENVI')
        dataset2 = driver.Open(raster_fn)
        layer2 = dataset2.GetLayer()  # from layer
        sr = layer2.GetSpatialRef()

    #sr = classification.GetSpatialRef()
    print("sr", sr)

    # generate mask data
    driver = ogr.GetDriverByName('ENVI')
    mask_data = np.where(classification_data == 0, False, True)
    mask_ds, mask_band = _create_in_memory_band(
        mask_data, band.XSize, band.YSize, classification) # classification.GetProjection(),
        #classification.GetGeoTransform())


    # Create a vector layer.
    driver = ogr.GetDriverByName('ESRI Shapefile') #'GeoJSON')
    print('+w', ofn)    
    dst_ds = driver.CreateDataSource(ofn) 
    dst_layer = dst_ds.CreateLayer('polygonize', sr)
    field_name = ogr.FieldDefn("polygonize", ogr.OFTInteger)
    field_name.SetWidth(24)
    dst_layer.CreateField(field_name)
    # Turn the rasters into polygons.
    gdal.Polygonize(band, mask_band, dst_layer, 0, [], callback=None)
    # Ensure that all data in the target dataset is written to disk.
    dst_ds.FlushCache()

polygonize(raster_fn, geojson_filename)
# a = os.system('ogr2ogr -f "GeoJSON" ' + kml_filename + ' ' + geojson_filename) # + ' ' + kml_filename)
