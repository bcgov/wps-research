import os
import sys
args = sys.argv

def err(m):
    print('Error', m)
    sys.exit(1)

if len(args) < 3:
    err('raster_burn_zeros_all.py [file to select NAN from] [directory with .bin files to burn onto]')

files = [x.strip() for x in os.popen('ls -1 ' + os.path.abspath(args[2]) + os.path.sep + '*.bin').readlines()]
for f in files:
    cmd = 'raster_burn_zeros ' + args[1] + ' ' + f
    print(cmd)
    a = os.system(cmd)

