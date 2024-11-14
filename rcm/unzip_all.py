'''parallel unzip of everything in the file sub-hierarchy..
..needs to do a hierarchical search due to rcm_group_by_beam.py


20241114 Note: for ALOS case, ALOS-2 data from EODMS have a different file structure than ALOS-2 data from JAXA. 

../py/shallow.py may be need to run ( for the EODMS case ) to fix the file structure e.g. before running ./alos2_dualpol_processing.py 

20241114: update: ( for resuming partial completion ): only perform unzipping for cases where the destination folder doesn't already exist
'''
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
print(files)
for f in files:
    print(f)

def unzip(f):
    dst  = '.'.join(f.strip().split('.')[:-1])
    dst = dst + sep
    if not os.path.exists(dst):
        os.mkdir(dst)
        cmd = ('unzip -o ' + f + ' -d ' + dst)
        return os.system(cmd)
    else:
        print(f, 'SKIPPING')
        return ''

parfor(unzip, files, 2)
