'''20240604 run this in a folder containing multiple:

    L2_{S2-tile-id}

It will bring the latest MRAP files into the present directory, and merge them here.

'''

import os
import shutil
lines = os.popen('find ./ -name "*_cloudfree.bin_MRAP.bin"').readlines()
lines = [x.strip() for x in lines]

files = {}
for line in lines:
    line = os.path.normpath(line)
    x = line.split(os.path.sep)
    tile_ID = x[0].split('_')[1]
    w = x[1].split('_')
    tile_ID2 = w[5]
    
    if tile_ID != tile_ID2:
        err('data not properly grouped by sentinel-2 tileID')

    if tile_ID not in files:
        files[tile_ID] = []
    
    time_stamp = w[2]
    files[tile_ID] += [ [time_stamp, tile_ID, line]] 

latest = []

for tile_ID in files:
    # for f in files[tile_ID]: print(f)
    files[tile_ID].sort(reverse=True)
    # for f in files[tile_ID]: print(f)

    latest += [files[tile_ID][0]] 

for late in latest:
    bin_f = late[2]
    print(bin_f)
    shutil.copy(bin_f, './')
    print(bin_f[:-3] + 'hdr')
    shutil.copy(bin_f[:-3] + 'hdr', './')

print('now run\n\tmerge2.py')

