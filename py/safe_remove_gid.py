'''20230526 delete zip files from other Jurisdictions'''
import os
import sys
import datetime
from misc import parfor, sep, run

from gid import bc
select = bc() # unpack selected tiles. bc() is all tiles for BC. Tiles in .tiles_select are ones that intersect with known fires

zip_gid, zips = {}, [x.strip() for x in os.popen('ls -1 *.zip').readlines()]
for f in zips:
    w = f.split('_')
    gid = w[5]
    if gid not in zip_gid:
        zip_gid[gid] = []
    zip_gid[gid] += [f]

for gid in zip_gid:
    if gid not in select:
        for f in zip_gid[gid]:
            run('rm ' + f)

