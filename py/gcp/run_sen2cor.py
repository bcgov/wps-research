''' run sen2cor:
    on Sentinel-2 folders in this directory, without matching L2 folders!

E.g. in the case of S2A_MSIL1C_20180728T200851_N0206_R028_T09VUE_20180728T233752.SAFE
this script would run:

    fix_s2 S2A_MSIL1C_20180728T200851_N0206_R028_T09VUE_20180728T233752.SAFE/
    L2A_Process S2A_MSIL1C_20180728T200851_N0206_R028_T09VUE_20180728T233752.SAFE/

Should have a path variable for adjusting this:
/sen2cor/2.5/Sen2Cor-02.05.05-Linux64/bin/L2A_Process
'''
import os
import sys
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + "..")
from misc import sep, parfor, exists, args, run, err
N_THREADS = None # default to number of CPU threads (could enter number to override here)

n_l1 = 0 # number of L1 folders
n_l2 = 0 # number of L2 folders
do = [] # L1 folders without L2 folder i.e. folders to process

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

if len(args) > 1:
    err('usage: python3 run_sen2cor.py [optional arg for info-only mode]' +
        'info-only mode does not run sen2cor')

def fix_s2(f):  # add folders expected by Sen2Cor! Gcp omits empty dirs
    def md(f):
        if not exists(f):
            print('mkdir', f)
            os.mkdir(f)
    md(f + sep + 'AUX_DATA')
    md(f + sep + 'HTML')

def run_sen2cor(f):
    fix_s2(f)
    L2A = '/home/' + os.popen('whoami').read().strip() + '/sen2cor/Sen2Cor-02.05.05-Linux64/bin/L2A_Process'
    if not exists(L2A):
        err('please install sen2cor and try again')
    L2A = os.path.abspath(L2A)
    cmd = (L2A + ' ' + f)
    return run(cmd)

ret = parfor(run_sen2cor,  # function to run in parallel
             do,  # to do list
             N_THREADS)  # run with 8 threads! 
print('retcodes', ret)
