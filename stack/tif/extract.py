# project_onto [src image to reproject] [target img to project onto] [output filename] [optional parameter: override bilinear and use nearest-neighbour]
import os
import sys

if not os.path.exists('bin'):
    os.mkdir('bin')

files = os.popen("ls -1 *.tif").readlines()

for f in files:
    f = f.strip()
    ofn = 'bin' + os.path.sep + f[:-3] + 'bin'
    #ofn2 = 'bin' + os.path.sep + f[:-4] + '2.bin'
    #if not os.path.exists(ofn):
    #    cmd = (' '.join(['gdal_translate -of ENVI -ot Float32', f, ofn]))
    #    print(cmd)
    if not os.path.exists(ofn):
        cmd = (' '.join(['project_onto', f, '../stack.bin', ofn, '1']))
        print(cmd)
        a = os.system(cmd)

    ofm = 'bin' + os.path.sep + f[:-3] + 'bin_binary.bin'
    if not os.path.exists(ofm):
        # python3 ../../py/class_merge.py bin/WATERSP.bin
        cmd = 'python3 ../../py/class_merge.py ' + ofn
        print("not executed:", cmd)

    # merging classes after scaling up is inefficient. We should fix this by revising the above, commented-out step
