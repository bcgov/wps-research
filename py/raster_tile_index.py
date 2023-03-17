import os
import sys
from misc import timestamp
ts = timestamp()

rasters = [x.strip() for x in os.popen('find ./ -name "*.bin"').read().strip().split('\n')]

open(ts + '_files.txt', 'wb').write(('\n'.join(rasters)).encode())

shp = ts + '_tileindex.shp'

cmd = 'gdaltindex -t_srs EPSG:4326 ' + shp + ' --optfile ' + ts + '_files.txt'
print(cmd)

