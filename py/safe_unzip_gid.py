'''20230520 unzip sentinel2 files. Assume we are in folder active/yyyymmdd for a given date.
Used to be just for tiles that are "on fire" according to bcws data
20230525 changed to any tiles over BC 
20230515 unzip sentinel2 files that are not already unzipped'''
import os
import sys
import datetime
from misc import parfor, sep

from gid import bc
select = bc() # unpack selected tiles. bc() is all tiles for BC. Tiles in .tiles_select are ones that intersect with known fires

'''
now = datetime.date.today()
year, month, day = str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2)
print([year, month, day])
L2_F = 'L2_' + year + month + day + '/'
'''

zip_gid, zips = {}, [x.strip() for x in os.popen('ls -1 *.zip').readlines()]
for f in zips:
    w = f.split('_')
    gid = w[5]
    if gid not in zip_gid:
        zip_gid[gid] = []
    zip_gid[gid] += [f]

cmds = []
observed = []
not_observed = []
for gid in select:
    print(gid)
    if gid not in zip_gid:
        not_observed += [gid]
        continue
    else:
        observed += [gid]
    files = zip_gid[gid]
    for f in files:
        d = f[:-4] + '.SAFE'
        if not os.path.exists(d):
            cmds += ['unzip ' + f] #  + ' -d ~/tmp/' + L2_F]  # would probably be a lot faster to unzip here!

def run(c):
    return os.system(c)

for c in cmds:
    print(c)

parfor(run, cmds, 4)

print("observed: ("  + str(len(observed)) + '/' + str(len(select)) + ') ' + ' '.join(observed))
print("not obs.: ("  + str(len(not_observed)) + '/' + str(len(select)) + ') ' +  ' '.join(not_observed))
