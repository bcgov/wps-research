'''20230518 determine area of raster

Fancier version would exclude NAN areas'''
from misc import err, args
from osgeo import gdal

if len(args) < 2:
		err("raster_area [raster filename]")

data = gdal.Open(args[1], gdal.GA_ReadOnly) 

geotr = data.GetGeoTransform()
pixel_area = abs(geotr[1] * geotr[5])

cols, rows = data.RasterXSize, data.RasterYSize

sq_m = pixel_area * rows * cols
sq_km = sq_m / (1000. * 1000.)  # 1000. * 1000. (square m) / (square km)?
print("Raster area: " + str(pixel_area * rows * cols) + ' m^2')
print("= "  + str(sq_km) + ' km^2')
