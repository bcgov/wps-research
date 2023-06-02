#!/usr/bin/env python3
import os
import sys
exist = os.path.exists

ignore = '.imv_ignore'
if not exist(ignore):
    open(ignore, 'wb').write("tmp_subset.bin".encode())


lines = os.popen("ls -1 *.bin").readlines()
lines = [x.strip() for x in lines]

for line in lines:
    ignore_files = set([x.strip() for x in open(ignore).read().strip().split('\n')])
    if line != "tmp_subset.bin" and line not in ignore_files:
        a = os.system("imv " + line)
        ignore_files.add(line)
        open(ignore,'wb').write(('\n'.join(list(ignore_files))).encode())
        sys.exit(0)
# need to plot filename title on imv
