''' run sen2cor: on Sentinel2 folders that don't already have L2 folders!
This converts Level-1 (C) data / L1C to Level-2 (A) / L2A..
..by calling the Linux binary for Sen2Cor v2.5. Sen2cor installation 
not done by this script. revised 20220228'''
import os
import sys
import multiprocessing as mp
sep = os.path.sep

def parfor(my_function, my_inputs, n_thread=mp.cpu_count()): # evaluate a function in parallel, collect the results
    pool = mp.Pool(n_thread)
    result = pool.map(my_function, my_inputs)
    return(result)

''' fix_s2 S2A_MSIL1C_20180728T200851_N0206_R028_T09VUE_20180728T233752.SAFE/
    L2A_Process S2A_MSIL1C_20180728T200851_N0206_R028_T09VUE_20180728T233752.SAFE/'''
n_l1, n_l2, do = 0, 0, []
files = [x.strip() for x in os.popen('ls -1').readlines()]
for f in files:
    if f[-5:] == '.SAFE':
        w = f.split('_')
        print(w)
        if w[1] == 'MSIL1C':
            if f.replace('MSIL1C', 'MSIL2A') not in files:
                do.append(f)
            n_l1 += 1
        elif w[1] == 'MSIL2A':
            n_l2 += 1
        else:
            pass
print("number of L1:", n_l1)
print("number of l2:", n_l2)
print("number to do:", len(do))

def fix_s2(f):  # make sure expected folders are there for sen2cor
    def mk(f):  # google omits empty dirs
        if not os.path.exists(f):
            print('mkdir', f)
            os.mkdir(f)
    mk(f + sep + 'AUX_DATA')
    mk(f + sep + 'HTML')

def run_sen2cor(f):
    fix_s2(f)
    L2A = 'L2A_Process' # '/home/' + os.popen('whoami').read().strip() + '/sen2cor/2.5/Sen2Cor-02.05.05-Linux64/bin/L2A_Process'
    cmd = (L2A + ' ' + f)
    print([cmd])
    a = os.system(cmd)
    return(a)

ret = parfor(run_sen2cor, do, 16)  # run with 8 threads! 
print('retcodes', ret)
