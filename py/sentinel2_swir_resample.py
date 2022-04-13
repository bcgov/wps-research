import os
from misc import err, run, args, pd, parfor

print("pd", pd)
X = os.popen('find ./ -name "S*swir.bin" | s2s | grep S2').readlines()
X, Y = [x.strip() for x in X], []

for f in X:
    Y.append(' '.join(['python3',
                       pd + 'raster_warp_20m.py',
                       f,
                       '1']))  # override output file check
    print(Y[-1])
parfor(run, Y, 4)
