'''20230516 run gdal_merge.py on all .bin files in present folder'''
import os
from misc import run
lines = [x.strip() for x in os.popen('ls -1 *.bin').readlines()]

open('.file_list', 'wb').write(('\n'.join(lines)).encode())
run(' '.join(['gdalbuildvrt',
        	  '-input_file_list .file_list',
			  'tmp.vrt']))
run(' '.join(['gdal_translate',
			  'tmp.vrt', 
			  '-of ENVI -ot Float32',
			  "-a_nodata '-nan'",
			  'merge.bin']))
run('fh merge.hdr')
run('envi_header_copy_bandnames.py ' + lines[0][:-4] + '.hdr merge.hdr')
