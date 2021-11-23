'''parallel unzip of everything in the file sub-hierarchy'''
import os
import sys
import multiprocessing as mp
sep = os.path.sep

def parfor(my_function, my_inputs, n_thread=mp.cpu_count()): # eval fxn in parallel, collect
    pool = mp.Pool(n_thread)
    result = pool.map(my_function, my_inputs)
    return(result)

files = os.popen('find ./ -name "*.zip"').readlines()
files = [x.strip() for x in files]
for f in files:
    print(f)

def unzip(f):
    dst  = f.strip().split(sep)[:-1]
    dst = sep.join(dst) + sep
    cmd = ('unzip ' + f + ' -d ' + dst)
    return os.system(cmd)

parfor(unzip, files, 4)
