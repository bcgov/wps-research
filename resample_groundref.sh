#!/usr/bin/env python
import os
import sys

def err(m):
    print "Error: " + str(m)
    sys.exit(1)

def run(c):
    print "run(\"" + c + "\")"
    a = os.system(c)
    if a != 0:
        err("command failed")
    return a


t = 'S2A.bin'
fn =['BROADLEAF_SP.tif',
     'CCUTBL_SP.tif',
     'CONIFER_SP.tif',
     'EXPOSED_SP.tif',
     'HERB_GRAS_SP.tif',
     'MIXED_SP.tif',
     'RiversSP.tif',
     'RoadsSP.tif',
     'SHRUB_SP.tif',
     'WATERSP.tif']

for f in fn:
    p_f = f + '_project.tif'
    p_b = f + '_project.bin'
    p_f4 = '_project_4x.bin'

    if not os.path.exists(p_f4):
        print "processing file: " + f + ".."
        run('rm -f ' + p_f)
        run('project_onto ' + f + ' ' + t + ' ' + p_f + ' 1')
        run('gdal_translate -of ENVI -ot Float32 ' + p_f + ' ' + p_b)
        run('gdal_translate -r nearest -of ENVI -outsize 25% 25% ' + p_f + ' ' + p_f4)

t4 = t + '_4x.bin'
if not os.path.exists(t4):
    run('gdal_translate -r average -of ENVI -outsize 25% 25% ' + t + ' ' + t4)

t2 = "L8.bin"
t2_4 = t2 + '_4x.bin'
if not os.path.exists(t2_4):
    run('gdal_translate -r average -of ENVI -outsize 25% 25% ' + t2 + ' ' + t2_4)
