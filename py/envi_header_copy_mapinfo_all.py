'''20221022: copy mapinfo from selected envi .hdr file, to all other
envi .hdr files in present directory (and subfolders)'''

import os
from misc import run, err, args

select = os.path.abspath(args[1])

lines = os.popen('find ./ -name "*.hdr"').readlines()
files = [os.path.abspath(x.strip()) for x in lines]

if select not in files:
    err("selected file should exist in present directory")

for f in files:
    if f != select:
        cmd = ' '.join(['envi_header_copy_mapinfo.py', select, f])
        run(cmd)

