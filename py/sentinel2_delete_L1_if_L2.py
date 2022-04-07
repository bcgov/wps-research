'''20220406 delete Sentinel2 Level-1 folders, in present directory,
for which the corresponding Level-2 folders are available

  E.g, if 
S2A_MSIL2A_20210621T185921_N0300_R013_T10UFB_20210621T224855.SAFE
  and 
S2A_MSIL1C_20210621T185921_N0300_R013_T10UFB_20210621T224855.SAFE
  are both in the folder, 

S2A_MSIL1C_20210621T185921_N0300_R013_T10UFB_20210621T224855.SAFE
will get deleted
'''
import os
import sys
from misc import args, run, err
msg = 'python3 delete_L1_if_L2.py [optional arg: execute]'

L1 = os.popen('ls -1 | grep .SAFE | grep L1C').readlines()
L2 = os.popen('ls -1 | grep .SAFE | grep L2A').readlines()
L1 = [x.strip() for x in L1]
L2 = [x.strip() for x in L2]

cmds = []
for x in L1:
    x = x.strip()
    x_L2 =  x.replace('MSIL1C', 'MSIL2A')
    print(x,'has_L2=' + str(x_L2 in L2))
    if x_L2 in L2:
        c = 'rm -rf ' + x
        cmds.append(c)

if len(args) > 1:
    for c in cmds:
        run(c)
else:
    for c in cmds:
        print(c)
    err('\n  ' + msg)
