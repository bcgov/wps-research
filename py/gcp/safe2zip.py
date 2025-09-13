'''20210718 take folders ending in .SAFE in a directory, create zip files from them
with same filename except replacing .SAFE with .zip.

If we do want to use this, presumably we'd apply it before running sen2cor by run_sen2cor.py
and save the archive somewhere else?'''
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from misc import err

to_zip = []
files = os.popen("ls -1").readlines()
for f in files:
    f = f.strip()
    if os.path.isdir(f):
        w = f.strip().split('.')[-1]
        if w == 'SAFE':
            to_zip.append(f)

for f in to_zip:
    target = '.'.join(f.strip().split('.')[:-1]) # replace .SAFE with .zip for zipfile..
    print(f, '->', target + '.zip')
    shutil.make_archive(target, 'zip', f)
