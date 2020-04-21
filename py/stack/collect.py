import os
import sys
files = os.popen('find kamloops-data-ufb-pansharpen/ -name "*4.bin"').readlines()

def nfn(f):
    nf = '_'.join(f.split(os.path.sep)[-2:])
    w = nf.split('_')
    x = w[1:7]
    x.append(w[0])
    x.append(w[-1])
    nf = '_'.join(x)
    return nf


for f in files:
    f = f.strip()
    cmd = 'cp ' + f + ' ' + nfn(f)
    print(cmd)
    a = os.system(cmd)
    if a != 0:
        print("error")
        sys.exit(1)