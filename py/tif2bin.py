'''convert all .tif files in directory, to .bin (ENVI type 4, Float32) 20220504'''
import os
from misc import parfor, run

tifs = [x.strip()
        for x in os.popen('ls -1 *.tif').readlines()]

cmds = []
for t in tifs:
    cmds.append(' '.join(['gdal_translate',
                          '-of ENVI',
                          '-ot Float32',
													'-co INTERLEAVE=BSQ',
                          t,  # input tif format raster file
                          t[:-3] + 'bin']))  # output ENVI file
for c in cmds:
    print(c)

parfor(run, cmds, 4)

