# take input raster file. change CRS to one specified in EPSG format
import os
import sys
args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

if len(args) < 3:
    err("raster_enforce_crs [input image file] [crs in EPSG]")

dst_EPSG = 32609 # default CRS: EPSG 32609
try:
    dst_EPSG = int(args[2])
except:
    err("target crs in EPSG must be an integer")

fn = args[1]
if not os.path.exists(fn):
    err("input file not found: " + str(fn))

ofn = fn + "_epsg_" + str(dst_EPSG)
try:
    ofn = fn[:-4] + "epsg_" + str(dst_EPSG) + fn[-4:]
except Exception:
    err("please check input filename format, needs extension .xxx")

cmd = ['gdalwarp',
        '-t_srs',
        'EPSG:' + str(dst_EPSG),
        input_raster,
        output_raster]


