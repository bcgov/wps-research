''' dominant classifier on SWIR subselection, for all S2 dates in folder 20220407'''
import os
import sys
import multiprocessing as mp
from misc import run, exists, parfor, args

if False:
    try:
        run('find ./ -name "*norm*bin" | xargs rm -f')
    except:
        pass
    try:
        run('find ./ -name "*dominant*bin" | xargs rm -f')
    except:
        pass
lines = [x.strip() for x in os.popen('find ./ -name "S*10m.bin_swir.bin"').readlines()]

x = []
cmd1 = []
cmd2 = []
for line in lines:
    df = line
    cmd = '~/GitHub/bcws-psu-research/cpp/raster_normalize.exe ' + df
    df += '_norm.bin'
    if not exists(df):
        cmd1.append(cmd) #run(cmd)
    cmd = '~/GitHub/bcws-psu-research/cpp/raster_dominant.exe ' + df
    df = line + '_dominant.bin'
    if not exists(df):
        cmd2.append(cmd) # run(cmd)
    x.append(df)

print(cmd1)
print(cmd2)
if len(cmd1) > 0 and len(args) < 2:
    print("normalizing..")
    parfor(run, cmd1, int(mp.cpu_count()/2))
if len(cmd2) > 0 and len(args) < 2:
    print("classifying..")
    parfor(run, cmd2, int(mp.cpu_count()/2))

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

