'''open .bin files in subfolders, with imv'''
import os
import sys
sep = os.path.sep
from misc import run, args

print('python3 bin.py')

cmd = 'find ./ -name "*.bin"'
X = [x.strip() for x in os.popen(cmd).readlines()]
X.sort()

for x in X:
    fn = x
    cmd = 'imv ' + fn
    run(cmd)
