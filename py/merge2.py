'''20230516 mosaic all .bin files in present folder 
Handles partial overlap. No-data value NAN assumed https://github.com/OSGeo/gdal/issues/3098

EPSG 
'''
EPSG = 3005  # BC Albers
# EPSG = 3347 # Canada LCC
import os
import sys
from misc import run, parfor, exists, sep
lines = [x.strip() for x in os.popen('ls -1 *.bin').readlines()]

if not exists('resample'):
	os.mkdir('resample')

cmds = []
for L in lines:
	if not exists('resample' + sep + L):
		cmds += [' '.join(['gdalwarp',
                           '-wo NUM_THREADS=8',
                           '-multi',
                           '-r bilinear',
                           '-srcnodata nan',
                           '-dstnodata nan',
                           '-of ENVI',
                           '-ot Float32',
                           '-t_srs EPSG:' + str(EPSG),
                           L,
                           'resample' + sep + L])]
parfor(run, cmds, 8)

run(' '.join(['gdalbuildvrt',
              '-srcnodata nan',
              '-vrtnodata nan',
              '-resolution highest',
              '-overwrite',
              'merge.vrt',
              'resample' + sep + '*.bin']))

if not exists('merge.bin'):
	run(' '.join(['gdalwarp',
                  '-wo NUM_THREADS=16',
                  '-multi',
                  '-overwrite',
                  '-r bilinear',
                  '-of ENVI',
                  '-ot Float32',
                  '-srcnodata nan',
                  '-dstnodata nan',
                  'merge.vrt',
                  'merge.bin']))  

run('fh merge.hdr')
run('envi_header_copy_bandnames.py ' + lines[0][:-4] + '.hdr merge.hdr')

