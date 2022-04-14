import os
from misc import err, run, args, pd, exist

X = os.popen('find ./ -name "S*swir.bin_20m.bin_dominant.bin" | s2s | grep S2').readlines()
X, Y = [x.strip() for x in X], []

for f in X:
    print(f)

cmd = ['python3', pd + 'raster_stack.py'] + X + ['dom.bin']
cmd = ' '.join(cmd)
run(cmd)
