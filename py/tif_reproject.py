# take a number of TIF files in present directory,
#   change the map projection before warping onto an image stack

# future: should autodetect the EPSG system for src and dst

# to build gdal from source:
# ./configure --with-proj=/usr/local --with-python

import os; exist = os.path.exists
import sys; args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c)
    if os.system(c) != 0:
        err("command failed: " + c)

def parfor(my_function, my_inputs): # evaluate a function in parallel, collect results
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    return pool.map(my_function, my_inputs)

# check input file exists
if len(args) < 2:
    err("tif_reproject.py [input image map to stack tifs onto]")

tgt = args[1]
if not exist(tgt):
    err("input file not found" + tgt)

# look for tifs 
tifs = [t.strip() for t in os.popen("ls -1 *.tif").readlines()]

# look for intermediary folder
for f in ["reprj", "type4", "merge", "out"]:
    if not exist(f):
        os.mkdir(f)

# warp the tifs
cmds = []
for tif in tifs:
    warped = "reprj" + os.path.sep + tif
    if not exist(warped):
        cmds.append("gdalwarp -s_srs EPSG:3005 -t_srs EPSG:32610 " + tif + " " + warped)
        print(cmds[-1])
parfor(run, cmds)

# convert dt
cmds = []
for tif in tifs:
    warped = "reprj" + os.path.sep + tif
    type4 = "type4" + os.path.sep + tif + '.bin'
    if not exist(type4):
        cmds.append("gdal_translate -of ENVI -ot Float32 " + warped + " " + type4)
        print(cmds[-1])
parfor(run, cmds)

# class_merge
cmds = []
for tif in tifs:
    type4 = "type4" + os.path.sep + tif + '.bin'
    out = type4 + '_binary.bin'
    if not exist(out):
        cmds.append(" ".join(["python3 ~/GitHub/wps-research/py/class_merge.py", type4, "1"]))
        print(cmds[-1])
parfor(run, cmds)

# project_onto
cmds = []
for tif in tifs:
    type4 = "type4" + os.path.sep + tif + '.bin'
    merge = type4 + '_binary.bin'
    out = "out" + os.path.sep + tif + '.bin'
    if not exist(out):
        cmds.append(" ".join(["project_onto", merge, tgt, out, "1"]))
        print(cmds[-1])
parfor(run, cmds)
