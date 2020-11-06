# take input raster file. change CRS to one specified in EPSG format
import os
import sys
args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

if len(args) < 2:
    err("raster_enforce_crs [input image file] [crs in EPSG] # default target crs: EPSG 32609")

dst_EPSG = 32609 # default CRS: EPSG 32609
if len(args) > 2:
    try:
        dst_EPSG = int(args[2])
    except Exception:
        err("target crs in EPSG must be an integer")

fn = args[1]
if not os.path.exists(fn):
    err("input file not found: " + str(fn))

ofn = fn + "_epsg_" + str(dst_EPSG)
try:
    ofn = fn[:-4] + "_epsg_" + str(dst_EPSG) + fn[-4:]
except Exception:
    err("please check input filename format, needs extension .xxx")

input_raster, output_raster = fn, ofn
cmd = ['gdalwarp -of ENVI -ot Float32 ',
        '-t_srs',
        'EPSG:' + str(dst_EPSG),
        input_raster,
        output_raster]

cmd = ' '.join(cmd)
print(cmd)

a = os.system(cmd)
