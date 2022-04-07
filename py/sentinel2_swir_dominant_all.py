''' dominant classifier on SWIR subselection, for all S2 dates in folder 20220407'''
import os
import sys
from misc import run, exists
lines = [x.strip() for x in os.popen('find ./ -name "S*10m.bin_swir.bin"').readlines()]

x = []
for line in lines:
    cmd = '~/GitHub/bcws-psu-research/cpp/raster_dominant.exe ' + line
    df = line + '_dominant.bin'
    if not exists(df):
        run(cmd)
    x.append(df)

X = []
for i in x:
    print(i)
    w = i.split('_')
    X.append([w[2], i])
X.sort()

print()
for i in X:
    x = i[1]
    run('imv ' + x)

