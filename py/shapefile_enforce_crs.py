# 20230604 reproject shapefile to specified CRS, where a CRS is indicated in EPSG format
import os
import sys
from misc import args, err, exist, run

dst_EPSG = 3347 # 3005? # 32609 # default CRS: EPSG 32609 

if len(args) < 2:
    err("python3 shapefile_enforce_crs.py [input shapefile] [optional argument: destination crs EPSG number] # default EPSG 32609")

fn = args[1]
try:
    if fn[-4:] != '.shp':
        err("shapefile input req'd")
except Exception:
    err("please check input file")

if not exist(fn):
    err("could not find input file: " + fn)

if len(args) > 2:
    try:
        dst_EPSG = int(args[2]) # override default EPSG
    except Exception:
        err("EPSG parameter must be an integer")

ofn = fn[:-4] + '_EPSG' + str(dst_EPSG) + '.shp'
cmd = ' '.join['ogr2ogr',
               '-t_srs',
               'EPSG:' + str(dst_EPSG),
               ofn,
               fn,
               "-lco ENCODING=UTF-8"] # source data goes second, ogr is weird!
run(cmd)
