# this script isn't working yet, try:
#  gdallocationinfo x.tif -wgs84 -131.125631744031 57.9156512571252

# extract spectra on grid pattern, for each point
from osgeo import gdal
from osgeo import ogr
import struct
import sys

def err(m):
    printf("Error: " + str(m)); sys.exit(1)

def pt2fmt(pt):
    fmttypes = {gdal.GDT_Byte: 'B',
                gdal.GDT_Int16: 'h',
		gdal.GDT_UInt16: 'H',
		gdal.GDT_Int32: 'i',
		gdal.GDT_UInt32: 'I',
		gdal.GDT_Float32: 'f',
		gdal.GDT_Float64: 'f'}
    return fmttypes.get(pt, 'x')

lat = float(sys.argv[2]) # lat
lon = float(sys.argv[3]) # long
ds = gdal.Open(sys.argv[1], gdal.GA_ReadOnly) # input data file
print("ds", ds)

transf = ds.GetGeoTransform()
cols = ds.RasterXSize
rows = ds.RasterYSize
bands = ds.RasterCount #1
band = ds.GetRasterBand(1)
bandtype = gdal.GetDataTypeName(band.DataType) #Int16
driver = ds.GetDriver().LongName #'GeoTIFF'
# success, transfInv = 
transfInv = gdal.InvGeoTransform(transf)

px, py = gdal.ApplyGeoTransform(transfInv, lon, lat)
structval = band.ReadRaster(int(px), int(py), 1,1, buf_type = band.DataType)
fmt = pt2fmt(band.DataType)
print("fmt", fmt)
print("structval", structval)
intval = struct.unpack(fmt , structval)
print(round(intval[0], 2)) #intval is a tuple, length=1 as we only asked for 1 pixel value

