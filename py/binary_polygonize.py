'''Polygonize a raster mask (values 0 or 1, data type and format otherwise not assumed)

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

from misc import exist, err, args
if len(args) < 2:
    err('python3 binary_polygonize.py [input raster mask file 1/0 values]')

raster_fn = args[1]
geojson_filename = raster_fn + '_poly.json'  # output filename

def polygonize(raster_fn, geojson_filename):
    classification = gdal.Open(raster_fn, gdal.GA_ReadOnly)
    band = classification.GetRasterBand(1)
    classification_data = band.ReadAsArray()

    # generate mask data
    mask_data = np.where(classification_data == 0, False, True)
    mask_ds, mask_band = _create_in_memory_band(
        mask_data, band.XSize, band.YSize, classification.GetProjection(),
        classification.GetGeoTransform())

    # Create a GeoJSON layer.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filename = os.path.join(temp_dir, 'temp.geojson')
        geojson_driver = ogr.GetDriverByName('GeoJSON')
        dst_ds = geojson_driver.CreateDataSource(temp_filename)
        dst_layer = dst_ds.CreateLayer('hfi')
        field_name = ogr.FieldDefn("hfi", ogr.OFTInteger)
        field_name.SetWidth(24)
        dst_layer.CreateField(field_name)

        # Turn the rasters into polygons.
        gdal.Polygonize(band, mask_band, dst_layer, 0, [], callback=None)

        # Ensure that all data in the target dataset is written to disk.
        dst_ds.FlushCache()
        source_projection = classification.GetProjection()
        # Explicitly clean up (is this needed?)
        del dst_ds, classification, mask_band, mask_ds

        data = _re_project_and_classify_geojson(temp_filename, source_projection)

        # Remove any existing target file.
        if os.path.exists(geojson_filename):
            os.remove(geojson_filename)
        with open(geojson_filename, 'w') as file_pointer:
            json.dump(data, file_pointer, indent=2)

polygonize(raster_fn, gfeojson_filename)
