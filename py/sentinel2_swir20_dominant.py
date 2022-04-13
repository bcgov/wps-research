import os
from misc import err, run, args, cd, parfor

X = os.popen('find ./ -name "S*swir.bin_20m.bin" | s2s | grep S2').readlines()
X, Y = [x.strip() for x in X], []

for f in X:
    Y.append(' '.join([cd + 'raster_dominant.exe',
                       f]))
    print(Y[-1])
# parfor(run, Y, 4)

for y in Y:
    run(y)
