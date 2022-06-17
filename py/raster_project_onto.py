#!/usr/bin/env python3
'''  projectOnto.py [A] [B] [C] [D]

Project data from file A to match the coordinate system of file B

The output to be stored in file C

Optional last parameter [D]: use nearest-neighbour resampling, not
bilinear which is the default

Note, probably need to compile GDAL from source to get this to work, with:
    ./configure --with-python
    make
    sudo make install

Also, sometimes the last step of that process breaks. May need to go into
the swig/python folder and manually do the:
    python setup.py build
    etc.
'''

import os
import sys
import multiprocessing as mp
N_THREAD = mp.cpu_count() # number of cpu thread
args = sys.argv
sep = os.path.sep
pd = sep.join(__file__.split(sep)[:-1]) + sep

def err(m):
    print("Error: " + m); sys.exit(1)

if len(args) < 4:
    err("project_onto [src image to reproject] " +
          "[target img to project onto] " +
          "[output filename] " + 
          "[optional parameter: override bilinear and use nearest-neighbour]")

use_nearest = len(sys.argv) >= 5
if use_nearest:
    print("using nearest neighbour resampling..")

src_filename = os.path.abspath(sys.argv[1])
match_filename = os.path.abspath(sys.argv[2])
dst_filename = os.path.abspath(sys.argv[3])

if(src_filename == match_filename or
   src_filename == dst_filename or
   match_filename == dst_filename):
    err("parameters must all be different.")

if(not(os.path.exists(src_filename)) or
   not(os.path.exists(match_filename))):
    err("input parameters must exist.")

# if(os.path.exists(dst_filename)):
#    err("destination filename already exists. Aborting.")

try:
    from osgeo import gdal
    from osgeo import gdalconst
    from osgeo import ogr
    from osgeo import osr
except:
    err("Error: gdal python API not available.\n" + 
        "Please install with e.g., ./configure --with-python" +
        "Or, google: pip install gdal python")

# use multithread option
gdal.SetConfigOption('GDAL_NUM_THREADS', str(N_THREAD)) # '4')  # 4 limit of SSD write?

# source image
src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
src_proj, src_geotrans = src.GetProjection(), src.GetGeoTransform()
src_sr = osr.SpatialReference(wkt=src.GetProjection())  # spatial reference

# want section of source that matches:
match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
match_proj, match_geotrans = match_ds.GetProjection(), match_ds.GetGeoTransform()
wide, high = match_ds.RasterXSize, match_ds.RasterYSize

# Output / destination
file_type = 'GTiff'
if dst_filename.split(".")[-1] == 'bin': file_type = 'ENVI'
print("driver_type", file_type)
dst = gdal.GetDriverByName(file_type)
dst = dst.Create(dst_filename, wide, high, src.RasterCount, gdalconst.GDT_Float32)
dst.SetGeoTransform(match_geotrans)
dst.SetProjection(match_proj)

# Do the work
resamp = gdalconst.GRA_NearestNeighbour if use_nearest else gdalconst.GRA_Bilinear
gdal.ReprojectImage(src, dst, src_proj, match_proj, resamp)

del dst

# copy band name info over if it got lost!
dst_hfn = dst_filename[:-4] + '.hdr'
src_hfn = src_filename[:-4] + '.hdr'
# print(src_hfn)
# a = os.system("cat " + src_hfn)
# print(dst_hfn)
# a = os.system("cat " + dst_hfn)

a = os.system('python3 ' + pd + 'envi_update_band_names.py ' + src_hfn + ' ' + dst_hfn)


'''
from https://gdal.org/python/osgeo.gdalconst-module.html, data types:

 	GDT_Byte = _gdalconst.GDT_Byte
  	GDT_UInt16 = _gdalconst.GDT_UInt16
  	GDT_Int16 = _gdalconst.GDT_Int16
  	GDT_UInt32 = _gdalconst.GDT_UInt32
  	GDT_Int32 = _gdalconst.GDT_Int32
  	GDT_Float32 = _gdalconst.GDT_Float32
  	GDT_Float64 = _gdalconst.GDT_Float64
  	GDT_CInt16 = _gdalconst.GDT_CInt16
  	GDT_CInt32 = _gdalconst.GDT_CInt32
  	GDT_CFloat32 = _gdalconst.GDT_CFloat32
  	GDT_CFloat64 = _gdalconst.GDT_CFloat64

sampling methods:
      	GRA_NearestNeighbour = _gdalconst.GRA_NearestNeighbour
  	GRA_Bilinear = _gdalconst.GRA_Bilinear
  	GRA_Cubic = _gdalconst.GRA_Cubic
  	GRA_CubicSpline = _gdalconst.GRA_CubicSpline
  	GRA_Lanczos = _gdalconst.GRA_Lanczos
  	GRA_Average = _gdalconst.GRA_Average
  	GRA_Mode = _gdalconst.GRA_Mode
  	GRA_Max = _gdalconst.GRA_Max
  	GRA_Min = _gdalconst.GRA_Min
  	GRA_Med = _gdalconst.GRA_Med
  	GRA_Q1 = _gdalconst.GRA_Q1
  	GRA_Q3 = _gdalconst.GRA_Q3
'''
