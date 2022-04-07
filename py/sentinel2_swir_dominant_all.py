''' dominant classifier on SWIR subselection, for all S2 dates in folder 20220407'''
import os
import sys
import multiprocessing as mp
from misc import run, exists, parfor

if False:
    try:
        run('find ./ -name "*norm*" | xargs rm')
    except:
        pass
    try:
        run('find ./ -name "*dominant*" | xargs rm')
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

if len(cmd1) > 0:
    print("normalizing..")
    parfor(run, cmd1, int(mp.cpu_count()/2))
if len(cmd2) > 0:
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

