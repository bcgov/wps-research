'''rename subsetted vri files to compact filename'''
import os
import sys
files = os.popen("ls -1 *_sub.*")
for f in files:
    f = f.strip()
    fn = f.replace(".bin_sub", "")
    if not os.path.exists(fn)
    cmd = "mv -v " + f + " " + fn
    a = os.system(cmd)
