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

InputVector = args[1] # shapefile to rasterize
RefImage = args[2] # footprint to rasterize onto
OutputImage = args[3]
if os.path.exists(OutputImage):
    err("output file already exists")

gdalformat = 'ENVI'
datatype = gdal.GDT_Float32 # Byte
burnVal = 1. #value for the output image pixels
##########################################################
# Get projection info from reference image
Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

# Open Shapefile
Shapefile = ogr.Open(InputVector)
Shapefile_layer = Shapefile.GetLayer()

# Rasterise
print("Rasterising shapefile...")
Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype) #, options=['COMPRESS=DEFLATE'])
Output.SetProjection(Image.GetProjectionRef())
Output.SetGeoTransform(Image.GetGeoTransform())

# Write data to band 1
Band = Output.GetRasterBand(1)
Band.SetNoDataValue(0)
gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

# Close datasets
Band = None
Output = None
Image = None
Shapefile = None
