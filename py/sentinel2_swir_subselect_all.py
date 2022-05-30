''' SWIR subselection for all bands 20220407
Assume 2 threads? '''
import os
import sys
import multiprocessing as mp
from misc import parfor, run
lines = [x.strip() for x in os.popen('find ./ -name "S*10m.bin"').readlines()]

cmds = []
for line in lines:
    cmd = '~/GitHub/wps-research/cpp/sentinel2_swir_subselect.exe ' + line
    cmds.append(cmd)
parfor(run, cmds, 2) # int(mp.cpu_count() / 2))
