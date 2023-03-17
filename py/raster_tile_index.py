import os
import sys
from misc import timestamp
ts = timestamp()

# list available rasters
rasters = [x.strip() for x in os.popen('find ./ -name "*.bin"').read().strip().split('\n')]

# write list of rasters
open(ts + '_files.txt', 'wb').write(('\n'.join(rasters)).encode())

# specify output file
shp = ts + '_tileindex.shp'

# print out command to generate tile index file
cmd = 'gdaltindex -t_srs EPSG:4326 ' + shp + ' --optfile ' + ts + '_files.txt'
print(cmd)
