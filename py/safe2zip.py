'''20230726: create zip files (e.g. for use by sentinel2_extract_swir.py) from .SAFE folders, e.g. downloaded from gcp'''

from misc import err, run, exists, parfor
import sys
import os

lines = [x.strip() for x in os.popen('ls -1').readlines()]

for line in lines:
    if line[-5:] == '.SAFE':
        b = line[:-5] 
        zfn = b + '.zip'
        
        cmd = 'zip -r ' + zfn + ' ' + line
        if not exists(zfn):
            run(cmd)        






