import os
import sys
from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst

def err(m):
    print("Error: " + m); sys.exit(1)

args = sys.argv
if len(args) < 4:
    err("Error: rasterize_onto.py: usage:" +
        "  python3 rasterize_onto.py [shapefile to rasterize] [image file: footprint to rasterize onto] [output filename]")
    sys.exit(1)

shp = args[1]
footprint_img = args[2] #  '/home/zeito/pyqgis_data/utah_demUTM2.tif'
output = args[3]
if os.path.exists(output):
    err("output file already exists")

# shp = '/home/zeito/pyqgis_data/polygon8.shp'
data = gdal.Open(footprint_img, gdalconst.GA_ReadOnly)
geo_transform = data.GetGeoTransform()
source_layer = data.GetLayer()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * data.RasterXSize
y_min = y_max + geo_transform[5] * data.RasterYSize
x_res = data.RasterXSize
y_res = data.RasterYSize
mb_v = ogr.Open(shp)
mb_l = mb_v.GetLayer()
pixel_width = geo_transform[1]
target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1,
        gdal.GDT_Float32) #target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
target_ds.SetGeoTransform(data.GetGeoTransform())
band = target_ds.GetRasterBand(1)
NoData_value = 0.
band.SetNoDataValue(NoData_value)
band.FlushCache()
gdal.RasterizeLayer(target_ds, [1], mb_l) #, options=["ATTRIBUTE=hedgerow"])

target_ds = None


