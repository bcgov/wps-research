# convert class maps to binary class maps, in parallel 20200602
import os
import sys

def parfor(my_function, my_inputs):
    # evaluate a function in parallel, and collect the results
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_function, my_inputs)
    return(result)

def run(c):
    print(c)
    a = os.system(c)

merge = "~/GitHub/bcws-psu-research/py/class_merge.py"
files = ["BROADLEAF_SP.bin",
         "CCUTBL_SP.bin",
         "CONIFER_SP.bin",
         "EXPOSED_SP.bin",
         "HERB_GRAS_SP.bin",
         "MIXED_SP.bin",
         "RiversSP.bin",
         "RoadsSP.bin",
         "SHRUB_SP.bin",
         "WATERSP.bin"]

cmds = []

for f in files:
    cmd = "python3 " + merge + " " + f
    print(cmd)
    if not os.path.exists(f + "_binary.bin"):
        cmds.append(cmd)

parfor(run, cmds)

