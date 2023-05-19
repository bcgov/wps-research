import os
import sys

args = sys.argv

def err(m):
    print('Error:', m)
    sys.exit(1)

print(args)

if len(args) < 4:
    print(args)
    err('raster_project_onto_all.py [path to .bin files to reproject] [.bin file (footprint) to project onto] [output path]')

if not os.path.exists(args[2]):
    err('file to project onto not found:' + args[2])

in_dir = os.path.abspath(args[1])
to_reproject = [x.strip() for x in os.popen('find ' + in_dir + os.path.sep + ' -name "*.bin"').readlines()]

for f in to_reproject:
    f = os.path.abspath(f)
    fn = f.split(os.path.sep)[-1]
    print(fn)

    ofn = os.path.abspath(args[3]) + os.path.sep + fn
    print(ofn)

    footp = os.path.abspath(args[2])

    cmd = 'raster_project_onto.py ' + f + ' ' + footp + ' ' + ofn
    print(cmd)

    a = os.system(cmd)

