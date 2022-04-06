'''open .bin files listed in bins.txt, with imv'''
import os
import sys
sep = os.path.sep
sys.path.append(sep.join(os.path.abspath(__file__).split(sep)[:-2]) + sep)
from misc import run, args

print('python3 sentinel2_imv.py [optional arg: just list sorted dates]')

cmd = 'find ./ -name "*_10m.bin"'
X = [x.strip().split('_') for x in os.popen(cmd).readlines()]
X = [[x[2], x] for x in X]
X.sort()

for x in X:
    fn = '_'.join(x[1])
    cmd = 'imv ' + fn
    if len(args) < 2:
        run(cmd)
    else:
        print(fn)
