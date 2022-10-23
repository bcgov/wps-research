'''20221022 run raster_burn_zeros on all .bin files in subfolders (except the reference file, which is used to burn the zeros from)'''
import os
from misc import run, err, args

select = os.path.abspath(args[1])

lines = os.popen('find ./ -name "*.bin"').readlines()
files = [os.path.abspath(x.strip()) for x in lines]

if select not in files:
    err("selected file should exist in present directory")

for f in files:
    if f != select:
        cmd = ' '.join(['raster_burn_zeros', select, f])
        print(cmd)

