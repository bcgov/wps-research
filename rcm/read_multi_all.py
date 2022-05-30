#!/usr/bin/env python3
# create png from rgb.bin files within subdirectories
import os
import sys

cmds = []
files = os.popen('find ./ -name "rgb.bin"').readlines()

py = '/home/' + os.popen('whoami').read().strip() + '/GitHub/wps-research/py/read_multi.py'
if not os.path.exists(py):
    print("Error: can't find file:", py)
    sys.exit(1)

for f in files:
    f = f.strip()
    w = f.split(os.path.sep)

    # write title string to file
    tsf = os.path.sep.join(w[:-1]) + os.path.sep + 'title_string.txt'
    print(tsf)
    open(tsf, 'wb').write((w[2][:-4]).encode())

    # write copyright string to file
    cf = os.path.sep.join(w[:-1]) + os.path.sep + 'copyright_string.txt'
    open(cf, 'wb').write('RCM data Copyright © 2020 Canadian Space Agency'.encode())

    # command to plot PNG with title and copyright strings
    cmd = py + ' ' + f + ' 1'
    cmds.append(cmd)

# write out command to file
cmds = [x.strip() for x in cmds]
print(cmds)
open('./read_multi_all.sh', 'wb').write(('\n'.join(cmds)).encode())

# process commands in parallel
a = os.system('multicore ./read_multi_all.sh')
a = os.system('rm -f read_multi_all.sh')  # clean up
