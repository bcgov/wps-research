'''
multitemporal accumulation of area detection result. Possibly mutate result using land cover filter before combining that
'''
from misc import sep, pd, exists, parfor
import os
# list L2 folders in the present directory. Will sort those in time! 

lines = [x.strip() for x in os.popen('ls -1').readlines()]
L = []
for x in lines:
    if x.strip()[-4:] == 'SAFE':
        w = x.split('_')
        if w[1] == 'MSIL2A':
            # L2A data set
            L.append([w[2], x])
L.sort()

cmds = []
dets = []
for x in L:
    # print(x)
    df = 'SENTINEL2_L2A_EPSG*10m.bin'
    cmd = 'find ' + x[1] + ' -name ' + df
    y = os.popen(cmd).read().strip()
    if y == '':
        print('x warning: no data ' + x[1])
    else:
        # run detector
        fn = y + '_active.bin'
        print('* ' + fn)
        dets.append([x[0], x[1], y])
        if not exists(fn):
            cmd = pd + '../cpp/sentinel2_active.exe ' + y
            print(cmd)
            cmds.append(cmd)

def r(x):
    return os.popen(x).read()

parfor(r, cmds, 4)  # limited by disk I/O

for d in dets:
    print(d)
