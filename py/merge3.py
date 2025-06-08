'''20250607: mosaic all .bin files specified on command line. 

NB last file specified is output file!!!!

adapted from:
20230516 mosaic all .bin files in present folder 
Handles partial overlap. No-data value NAN assumed https://github.com/OSGeo/gdal/issues/3098

Default EPSG for resampling: 3005 (BC Albers)

*** add an extra arg to use 3347: Canada LCC'''
EPSG = 3005  # BC Albers
# EPSG = 3347 # Canada LCC
import os
import sys
import multiprocessing as mp
from misc import run, parfor, exists, sep, err, hdr_fn
'''assume we are in BC
if len(sys.argv) > 1:
    EPSG = 3347
'''
lines = sys.argv[1:-1] #  [x.strip() for x in os.popen('ls -1 *.bin').readlines()]
output_file = sys.argv[-1]  # output file

if not exists('resample'):
    os.mkdir('resample')
else:
    pass #     err('directory resample/ exists. Remove and run again')

run('rm -f tmp_subset.*')
run('rm -f merge.*') 

cmds = []
resampled = []
for L in lines:
    ofn = L + '_resample.bin'
    resampled += [ofn]
    if not exists(ofn):
        cmds += [' '.join(['gdalwarp',
                           '-wo NUM_THREADS=16',
                           '-multi',
                           '-r bilinear',
                           '-srcnodata nan',
                           '-dstnodata nan',
                           '-of ENVI',
                           '-ot Float32',
                           '-t_srs EPSG:' + str(EPSG),
                           L,
                           ofn])]
    else:
        print("Skipping:", ofn)
parfor(run, cmds, int(mp.cpu_count()))

run(' '.join(['gdalbuildvrt',
              '-srcnodata nan',
              '-vrtnodata nan',
              '-resolution highest',
              '-overwrite',
              'merge.vrt'] + resampled))

if not exists(output_file):
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
                  output_file]))  

run('fh ' + hdr_fn(output_file))
run('envi_header_copy_bandnames.py ' + lines[0][:-4] + '.hdr ' + hdr_fn(output_file))
