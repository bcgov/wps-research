'''20230317 create a tile-index shapefile for rasters with .bin extension, in present worrking directory structure
'''
import os
import sys
from misc import timestamp, run
ts = timestamp()

# list available rasters
rasters = [x.strip() for x in os.popen('ls -1 *.bin').read().strip().split('\n')]

# write list of rasters
open(ts + '_files.txt', 'wb').write(('\n'.join(rasters)).encode())

# specify output file
shp = ts + '_tileindex.shp'

# print out command to generate tile index file
run('gdaltindex -t_srs EPSG:4326 ' + shp + ' --optfile ' + ts + '_files.txt')
