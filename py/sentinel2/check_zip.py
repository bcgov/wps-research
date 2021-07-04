'''input:  run from folder containing zipfiles.
   output: script to delete broken zipfiles.

1) list zipfiles that fail the test extract (prefixed with rm)..
2) then, you can run that as a script to delete the broken zipfiles'''

import os
import sys

def parfor(my_function, inputs, cpu_use=None):
    import multiprocessing as mp
    n_cpu = mp.cpu_count()
    if cpu_use is not None:
        n_cpu = cpu_use
    pool = mp.Pool(n_cpu)
    return pool.map(my_function, inputs)

lines = os.popen("ls -1 *.zip").read().strip().split("\n")
lines = [x.strip() for x in lines]

def chkzp(f):
    print(f)
    return [f, os.popen("zip -T " + f).read().strip()]

results = parfor(chkzp, lines, 2)  # use two threads to check the zipfiles

print("+w rm_broken.sh # run this to delete files indicated..")
broken = open("rm_broken.sh", "wb")
good = open("good.txt", "wb")
for i in range(0, len(results)):
    f, r = results[i]

    if len(r.split("zip error")) > 1:
        broken.write(('rm ' + f + '\n').encode())
    else:
        good.write((f + '\n').encode())
broken.close()
good.close()
