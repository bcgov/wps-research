import os
import sys

def run(c):
    print(c)
    a = os.system(c)

sep = os.path.sep
pd = sep.join(__file__.split(sep)[:-1]) + sep
set_band_desc = pd + 'raster_set_band_desc.py'

files_list = [f.strip() for f in os.popen("ls -1 *.TIF *.tif").readlines()]

files = []
for f in files_list:
    if f[0] == 'L':
        files.append(f) # landsat files begin with L

w = files[0].split('_')[:7]
vrt = ("_".join(w)) + '_stack.vrt'
out = ("_".join(w)) + '_stack.bin'

cmd = ['gdalbuildvrt', '-r', 'bilinear', '-resolution', 'highest', '-separate', vrt] + files
cmd = ' '.join(cmd)

if not os.path.exists(vrt):
    run(cmd)

# ENVI format was selected as GEOTIFF format does not support band names..In SNAP at least!
if not os.path.exists(out):
    run(' '.join(['gdal_translate', '-of ENVI', '-ot Float32', vrt, out]))

print("updating band names for " + out + "..")
bi = 1
cmd = ['python3', set_band_desc, out]
for f in files:
    cmd += [str(bi), f]
    bi += 1
cmd = ' '.join(cmd)
run(cmd)
