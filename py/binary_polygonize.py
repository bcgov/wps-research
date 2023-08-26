''' 202207 Polygonize a raster mask (values 0 or 1, data type and format otherwise not assumed)

(*) Output in same coordinate reference system as source data 

Based on a module by Sybrand Strauss: 
    https://github.com/bcgov/wps/blob/story/classify_hfi/api/scripts/polygonize_hfi.py#L52

20230825 '''
import os
import sys
import json
import tempfile
import numpy as np
from osgeo import ogr
from osgeo import gdal
from osgeo import osr
from misc import exist, err, args, run, sep

fire_number = None
image_date = None

pwd = os.path.abspath( os.getcwd())
w = pwd.split(sep)
if w[-3] == 'active':
    print("ACTIVE")
    print("Fire_number", w[-2])
    print("Image date", w[-1])
    fire_number = w[-2]
    image_date = w[-1]
print(w)

print(pwd)
# sys.exit(1)

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

def polygonize(geotiff_filename, filename):
    raster = gdal.Open(geotiff_filename, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    src_projection, geotransform = raster.GetProjection(), raster.GetGeoTransform()
    print(src_projection)
    #print(geotransform)
    rows, cols = band.YSize, band.XSize
    
    # source coordinate reference system
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster.GetProjectionRef()) # as in: https://trac.osgeo.org/gdal/browser/trunk/gdal/swig/python/scripts/gdal_polygonize.py#L237
    # generate mask data
    mask_data = np.where(band.ReadAsArray() == 0, False, True)
    mask_ds, mask_band = create_in_memory_band(mask_data, cols, rows, src_projection, geotransform)

    # Create output 
    driver = ogr.GetDriverByName('ESRI Shapefile') #GeoJSON')
    dst_ds = driver.CreateDataSource(filename)
    # dst_ds.SetProjection(src_projection)  # AttributeError: 'DataSource' object has no attribute 'SetProjection'
    # dst_ds.SetGeoTransform(geotransform)

    # add layer
    dst_layer = dst_ds.CreateLayer('fire')   # not sure how to get the CRS info into the output
    # dst_layer.SetProjection(src_projection)  # AttributeError: 'Layer' object has no attribute 'SetProjection'
    # dst_layer.SetGeoTransform(geotransform)

    field_name = ogr.FieldDefn("fire", ogr.OFTInteger)
    field_name.SetWidth(24)
    dst_layer.CreateField(field_name)
    gdal.Polygonize(band, mask_band, dst_layer, 0, [], callback=None)  # polygonize
    dst_ds.FlushCache()
    del dst_ds, raster, mask_ds # print(f'{filename} written')
    open(args[1] + '.prj', 'wb').write(str(src_projection).encode())

polygonize(args[1],
           args[1] + '.shp')

run(' '.join(['ogr2ogr -f "KML"',
              args[1] + '.kml',
              args[1] + '.shp']))

run('sentinel2_trace_active_alpha.py ' + args[1])


# assume were in an active/fire_number directory, rename the files accordingly for the product format specification.

if fire_number is not None and image_date is not None:
    print("ACTIVE")
    # have fire_number (fire ID) and image date. 

    # just need to get the image time right, now

    # find the symbolic links to S2 data files, in the present directory. And/or real files (legacy compatibility)
    # ls -1 S2*.bin
    # count the date-time stamps, use the most-ocurring observed pair (date, time)

'''
osgeo.ogr.GetDriverByName vs osgeo.gdal.GetDriverByName
Ok - so - ogr == vectors ; gdal == raster
https://pcjericks.github.io/py-gdalogr-cookbook/projection.html#get-projection
'''
