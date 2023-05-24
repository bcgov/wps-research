'''20230516 run gdal_merge.py on all .bin files in present folder https://github.com/OSGeo/gdal/issues/3098
'''
import os
import sys
from misc import run, parfor, exists
lines = [x.strip() for x in os.popen('ls -1 *.bin').readlines()]

if not exists('resample'):
	os.mkdir('resample')

cmds = []
for L in lines:
	if not exists('resample/' + L):
		cmds += [' '.join(['gdalwarp',
						   '-wm 1024',
						   '-wo NUM_THREADS=4',
						   '-multi',
						   '-r bilinear',
						   '-srcnodata nan',
						   '-dstnodata nan',
						   '-of ENVI',
						   '-ot Float32',
						   '-t_srs EPSG:4326',
						   L,
						   'resample/' + L])]
parfor(run, cmds, 2)

run(' '.join(['gdalbuildvrt',
			  '-srcnodata nan',
		      '-vrtnodata nan',
			  '-resolution highest',
			  '-overwrite',
			  'merge.vrt',
			  'resample/*.bin']))

if not exists('merge.bin'):
	run('gdalwarp -wm 4096 -wo NUM_THREADS=8 -multi -overwrite -r bilinear -of ENVI -ot Float32 -srcnodata nan -dstnodata nan merge.vrt merge.bin')  # -t_srs EPSG:4326
open('.file_list', 'wb').write(('\n'.join(lines)).encode())

run('fh merge.hdr')
run('envi_header_copy_bandnames.py ' + lines[0][:-4] + '.hdr merge.hdr')

