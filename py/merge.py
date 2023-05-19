'''20230516 run gdal_merge.py on all .bin files in present folder'''
import os
from misc import run
lines = [x.strip() for x in os.popen('ls -1 *.bin').readlines()]
cmd = 'gdal_merge.py ' + ' '.join(lines) + ' -of ENVI -ot Float32 -o merge.bin'
run(cmd)