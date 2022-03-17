'''
multitemporal accumulation of area detection result. Possibly mutate result using land cover filter before combining that
'''
from misc import sep, pd, exists, parfor, run, read_hdr, write_hdr
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
        dets.append([x[0], x[1], y, fn])
        if not exists(fn):
            cmd = pd + '../cpp/sentinel2_active.exe ' + y
            print(cmd)
            cmds.append(cmd)

def r(x):
    return os.popen(x).read()
parfor(r, cmds, 1)  # limited by disk I/O

nrow, ncol, nband = 0, 0, 0
cmd = 'cat '
for d in dets:
    print(d)
    ncol, nrow, nband = [int(x) for x in read_hdr(d[3][:-3] + 'hdr')]
    cmd += d[3] + ' '
cmd += " > multi.bin"

if not exists('multi.bin'):
    run(cmd)

if not exists('multi.hdr'):
    write_hdr('multi.hdr', ncol, nrow, nband)

