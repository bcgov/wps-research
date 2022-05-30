# convert class maps to binary class maps, in parallel 20200602
import os; exist = os.path.exists
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

merge = "~/GitHub/wps-research/py/class_merge.py"
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

search_files = False
for f in files:
    if not exist(f):
        search_files = True

if search_files:
    files = [f.strip() for f in os.popen("ls -1 b*.bin").readlines()]
print("files", files)

cmds = []
for f in files:
    cmd = "python3 " + merge + " " + f + " 1" # last arg skips generating binaries for all labels
    print(cmd)
    if not os.path.exists(f + "_binary.bin"):
        cmds.append(cmd)

parfor(run, cmds)

print("N.B., if class merging didn't work for a band, " +
      "try re-running the class_merge script for that band, " +
      "without the last argument, and examine the label maps " + 
      "using py/read_multi.py")
