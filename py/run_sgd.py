''' 20191213
run_sgd.py: run sgd classifier on
  - multispectral imagery and
  - a number of ground-reference layers (in data_bcgw/ and data_vri/ folders)

producing
  - an inference map for each ground-reference layer.

The data fraction excluded from modelling (withheld for model
validation) determined in sgd.py

(*) example of one run of sgd.py:
  "python3 py/sgd.py data_img/S2A.bin_4x.bin_sub.bin
    data_bcgw/merged/WATERSP.tif_project_4x.bin_sub.bin_binary.bin out"

(*) example of running sgd.py on multiple ground-reference classes:
    python3 py/run_sgd.py
    
todo: put accuracy on charts. Sort charts by accuracy'''
from misc import *
img_f = 'data_img/S2A.bin_4x.bin_sub.bin'

# sentinel 2: data_img/S2A.bin_4x.bin_sub.bin
# sentinel 2, landsat 8 fused: data_img/S2A_L8.bin_4x.bin_sub.bin
# landsat 8: data_img/L8.bin_4x.bin_sub.bin

# load bc geographic warehouse layers
ref = os.popen("ls -1 data_bcgw/merged/*binary.bin").readlines()

# load vri layers. Todo: extract leading species
vri_path = "data_vri/binary/"
vri = os.popen("ls -1 " + vri_path + "*.bin")
for v in vri:
    samples, lines, bands, data = read_binary(v.strip())
    count = hist(data.ravel())  # histogram counts of data

    # specify min # of points, positive or negative
    enough_points = min(count.values()) > 1000

    # todo: experiment with params or turn this off!
    if len(count) > 1 and enough_points:
        ref.append(v)

# set up output directory
if not exist('out'):
    os.mkdir('out')

# sanity check
if not os.path.isdir('out'):
    err("out not directory")

# make a list of all (model + predict) operations
jobs = []
for r in ref:
    r = r.strip()
    cmd = 'python3 py/sgd.py ' + img_f.strip() + ' ' + r + ' out'
    jobs.append(cmd)

# run all (model + predict) operations
use_parallel = True  # to turn off parallel processing: use_parallel = False
if use_parallel:
    parfor(run, jobs)
else:
    for job in jobs:
        run(job)
