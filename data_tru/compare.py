import os
import sys
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
sep = os.path.sep

def err(m):
    print("Error: " + str(m)); sys.exit(1)

def read_float(fn):  # read raw float data file. 4byte / float, byte-order 0
    return np.fromfile(fn, dtype = np.float32)

# load supervised result
supervised = {}
bins = [x.strip() for x in os.popen("ls -1 brad/*.bin").readlines()]

for b in bins:
    print("+r " + b)
    pfn = b.split(sep)[-1][:-9] + '.png'
    X = pickle.load(open(b, 'rb'))
    supervised[pfn[:-4]] = X.astype(np.float32)
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


values = []
s_used = {s:False for s in supervised}
u_used = {u:False for u in unsupervised}
for s in supervised:
    ds = supervised[s]
    for u in unsupervised:
        du = unsupervised[u]
        Z = np.sum(ds * du)
        values.append([Z, s, u])

values.sort()

pairs = []


def set(s, u):
    if(s_used[s] == True):
        err("already set:", s)
    if(u_used[u] == True):
        err("already set:", u)
    pairs.append([s,u])
    s_used[s] = True
    u_used[u] = True

set('pineburneddeciduous', '0')
set('fireweed', '1')
set('exposed', '2')
set('windthrowgreenherbs', '3')
set('grass', '4')
set('deciduous', '5')
set('lake', '6')
set('herb', '7')
set('conifer', '8')
set('blowdownfireweed', '9')
set('blowdownlichen', '10')
set('fireweeddeciduous', '11')
set('pineburnedfireweed', '12')
set('pineburned', '13')

'''
for (Z, s, u) in values:
    if s_used[s] != True and u_used[u] != True:
        s_used[s], u_used[u] = True, True
        pairs.append([s, u])
        print(Z, pairs[-1])
'''

for (s, u) in pairs:
    gif_file = s + "_" + u + ".gif"
    if not os.path.exists(gif_file):
        c = "convert -delay 111 " + s + ".png " + u + ".png " + gif_file
        print(c)
        a = os.system(c)

u_order = []
for (s, u) in pairs:
    u_order.append(u)

print(','.join([u_order[j] for j in range(len(pairs))]))
for i in range(len(pairs)):
    s = pairs[i][0]
    sX = supervised[s]
    print(s, end='')
    for j in range(len(pairs)):
        u = u_order[j]
        uX = unsupervised[u]
        print(",", end="")
        print(str(np.sum(  sX * uX)), end="")

        
    

