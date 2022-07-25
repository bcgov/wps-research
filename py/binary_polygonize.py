'''Not finished.

Polygonize a raster mask (values 0 or 1, data type and format otherwise not assumed)

(*) Output in same coordinate reference system as source data 

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

from misc import exist, err, args
if len(args) < 2:
    err('python3 binary_polygonize.py [input raster mask file 1/0 values]')

def get_wkt(epsg, wkt_format="esriwkt"):
    default = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295],UNIT["Meter",1]]'
    spatial_ref = osr.SpatialReference()
    try:
        spatial_ref.ImportFromEPSG(epsg)
    except TypeError:
        print("ERROR: epsg must be integer. Returning default WKT(epsg=4326).")
        return default
    except Exception:
        print("ERROR: epsg number does not exist. Returning default WKT(epsg=4326).")
        return default
    if wkt_format=="esriwkt":
        spatial_ref.MorphToESRI()
    # return a nicely formatted WKT string (alternatives: ExportToPCI(), ExportToUSGS(), or ExportToXML())
    return spatial_ref.ExportToPrettyWkt()

def create_in_memory_band(data: np.ndarray, cols, rows, projection, geotransform):
    mem_driver = gdal.GetDriverByName('MEM')
    dataset = mem_driver.Create('memory', cols, rows, 1, gdal.GDT_Byte)
    dataset.SetProjection(projection)
    dataset.SetGeoTransform(geotransform)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    return dataset, band

def polygonize(geotiff_filename, filename):
    raster = gdal.Open(geotiff_filename, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    src_projection, geotransform = raster.GetProjection(), raster.GetGeoTransform()
    print(src_projection)
    print(geotransform)
    rows, cols = band.YSize, band.XSize
    
    # source coordinate reference system
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjectionRef()) # as in: https://trac.osgeo.org/gdal/browser/trunk/gdal/swig/python/scripts/gdal_polygonize.py#L237
    
    # generate mask data
    mask_data = np.where(band.ReadAsArray() == 0, False, True)
    mask_ds, mask_band = create_in_memory_band(mask_data, cols, rows, src_projection, geotransform)

    # Create output 
    geojson_driver = ogr.GetDriverByName('ESRI Shapefile') #GeoJSON')
    dst_ds = geojson_driver.CreateDataSource(filename)
    dst_layer = dst_ds.CreateLayer('fire', srs)   # not sure how to get the CRS info into the output
    field_name = ogr.FieldDefn("fire", ogr.OFTInteger)
    field_name.SetWidth(24)
    dst_layer.CreateField(field_name)
    gdal.Polygonize(band, mask_band, dst_layer, 0, [], callback=None)  # polygonize
    dst_ds.FlushCache()
    del dst_ds, raster, mask_ds # print(f'{filename} written')

    open(args[1] + '.prj', 'wb').write(str(src_projection).encode()) #.write(get_wkt(32609).encode()) # str(srs).encode())

polygonize(args[1],
           args[1] + '.shp')
