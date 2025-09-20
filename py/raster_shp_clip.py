'''
20250919: raster_shp_mask.py


raster_shp mask [raster_file] [shape_file (.shp)]
- [raster_file]: input data file
- [shape_file]: polygon shapefile, apply NAN to areas not under polygons
'''
from osgeo import gdal, ogr
from misc import err, args
import numpy as np

if len(args) < 3:
    err("raster_shp_mask [rasetr_file] [polygon shapefile] # create output with NAN outside of poly areas")

# File paths
raster_path = args[1] # "your_raster_file.bsq"        # ENVI binary file (with .hdr)
vector_path = args[2] # "your_shapefile.shp"          # Shapefile with polygon mask
output_path = args[1] + "_clipped.bin" # "masked_output.tif"           # Output file path

# 1. Open the raster
raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
num_bands = raster_ds.RasterCount
xsize = raster_ds.RasterXSize
ysize = raster_ds.RasterYSize
geotransform = raster_ds.GetGeoTransform()
projection = raster_ds.GetProjection()

# 2. Create in-memory raster mask
mem_driver = gdal.GetDriverByName('MEM')
mask_ds = mem_driver.Create('', xsize, ysize, 1, gdal.GDT_Byte)
mask_ds.SetGeoTransform(geotransform)
mask_ds.SetProjection(projection)

# 3. Rasterize shapefile into mask
vector_ds = ogr.Open(vector_path)
layer = vector_ds.GetLayer()
gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1])

# 4. Read the mask
mask_band = mask_ds.GetRasterBand(1)
mask_array = mask_band.ReadAsArray()

# 5. Create output dataset
gtiff_driver = gdal.GetDriverByName('ENVI')
out_ds = gtiff_driver.Create(output_path, xsize, ysize, num_bands, gdal.GDT_Float32)
out_ds.SetGeoTransform(geotransform)
out_ds.SetProjection(projection)

# 6. Process each band
for b in range(1, num_bands + 1):
    band = raster_ds.GetRasterBand(b)
    data = band.ReadAsArray().astype(np.float32)
    
    # Mask: keep values inside polygons, set others to NaN
    masked_data = np.where(mask_array == 1, data, np.nan)

    out_band = out_ds.GetRasterBand(b)
    out_band.WriteArray(masked_data)
    out_band.SetNoDataValue(np.nan)

# 7. Clean up
raster_ds = None
mask_ds = None
vector_ds = None
out_ds = None

print(f"Masked raster with {num_bands} bands saved to: {output_path}")

