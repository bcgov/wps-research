import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
sep = os.path.sep

def read_float(fn):  # read raw float data file. 4byte / float, byte-order 0
    return np.fromfile(fn, dtype = np.float32)

# load supervised result
supervised = {}
bins = [x.strip() for x in os.popen("ls -1 brad/*.bin").readlines()]

for b in bins:
    print("+r " + b)
    pfn = b.split(sep)[-1][:-9] + '.png'
    X = pickle.load(open(b, 'rb'))
    supervised[pfn[:-4]] = X
    if not os.path.exists(pfn):
        plt.imshow(X)
        plt.title(pfn[:-4])
        plt.tight_layout()
        print("+w " + pfn)
        plt.savefig(pfn)

print(supervised.keys())
print(len(supervised.keys()))
s = supervised[list(supervised.keys())[0]].shape

# load unsupervised result
bins = [x.strip() for x in os.popen("ls -1 gagan/*.png").readlines()]
unsupervised = {}
for b in bins:
    bf =  b[:-4] + ".bin"
    c = "gdal_translate -of ENVI -ot Float32 -b 1 " + b + " " + bf
    if not os.path.exists(bf):
        print(c)
        a = os.system(c)

    X = read_float(bf).reshape(s) / 255.
    pfn = b.split(sep)[-1].split('.')[0]
    unsupervised[pfn] = X
    pfn += '.png'
    if not os.path.exists(pfn):
        plt.imshow(X)
        plt.title(pfn)
        plt.tight_layout()
        print("+w " + pfn)
        plt.savefig(pfn)

if len(supervised.keys()) != len(unsupervised.keys()):
    print("Error: different number of files to match."); sys.exit(1)


'''
c = {}
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        d = X[i,j]
        if not d in c:
            c[d] = 0
        c[d] += 1
print(c)
'''
