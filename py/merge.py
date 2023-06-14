'''20230516 run gdal_merge.py on all .bin files in present folder
NB can use this one if the data are in the same UTM zone'''
import os
import sys
from misc import run
lines = []

ohn = 'merge.hdr'
ofn = 'merge.bin'
if len(sys.argv) < 4:
    lines = [x.strip() for x in os.popen('ls -1 *.bin').readlines()]

if len(sys.argv) >= 4:
    ofn = sys.argv[-1]
    if ofn[-3:] != 'bin':
        print("Error: .bin expected")
    ohn = ofn[:-4] + '.hdr'
    lines = sys.argv[1:-1]

if len(lines) >= 2:
    cmd = 'gdal_merge.py -of ENVI -ot Float32 -n nan ' + ' '.join(lines) + ' -o ' + ofn
    run(cmd)
    run('fh ' + ohn)
    run('envi_header_copy_bandnames.py ' + lines[0][:-4] + '.hdr ' + ohn)
