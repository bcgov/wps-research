'''
 python3 py/sgd.py data_img/S2A.bin_4x.bin_sub.bin data_bcgw/merged/WATERSP.tif_project_4x.bin_sub.bin_binary.bin out
'''
from misc import *
ref = os.popen("ls -1 data_bcgw/merged/*binary.bin")

if not exist('out'):
    os.mkdir('out')

if not os.path.isdir('out'):
    err("out not directory")

jobs = []
for r in ref:
    r = r.strip()
    cmd = 'python3 py/sgd.py data_img/S2A.bin_4x.bin_sub.bin ' + r + ' out'
    jobs.append(cmd)

use_parallel = True
if use_parallel:
    parfor(run, jobs)
else:
    for job in jobs:
        run(job)
