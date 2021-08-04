import os
import sys
import numpy as np
import matplotlib.pyplot as plt
def exist(f): return os.path.exists(f)

def err(m):
    print("Error: " + m); sys.exit(1)

def hdr_fn(bin_fn): # return filename for header file, given name for bin file
    hfn = bin_fn[:-4] + '.hdr'
    if not exist(hfn):
        hfn2 = bin_fn + '.hdr'
        if not exist(hfn2): err("didn't find hdr file at: " + hfn + " or: " + hfn2)
        return hfn2
    return hfn

def read_hdr(hdr): # read an ENVI hdr file
    ns, nl, nb = 0, 0, 0
    for line in open(hdr).readlines():
        line = line.strip()
        words = line.split('=')
        if len(words) == 2:
            f, g = words[0].strip(), words[1].strip()
            if f == 'samples': ns = g
            if f == 'lines': nl = g
            if f == 'bands': nb = g
    return [int(x) for x in [ns, nl, nb]]

def read_float(fn): # read a raw binary file type 4 / ENVI IEEE floating point 32bit
    print("+r", fn)
    return np.fromfile(fn, dtype = np.float32)

def htrim(x, percent = 1.):
    import math
    x = x.tolist()
    y = sorted(x)
    p = math.floor(.01 * len(x) * percent)
    print("p", p)
    xmn, xmx = y[p], y[-p]
    x = [(xi - xmn) / (xmx-xmn) for xi in x]
    for i in range(len(x)):
        x[i] = 0. if x[i] < 0. else x[i]
        x[i] = 1. if x[i] > 1. else x[i]
    return x

'''
x ={'RCH_i': 'i_RCH_slv1_15Jul2020.img', # data files for our acquisition
    'RCV_i': 'i_RCV_slv1_15Jul2020.img',
    'RCH_q': 'q_RCH_slv1_15Jul2020.img',
    'RCV_q': 'q_RCV_slv1_15Jul2020.img'} # you would need to change filenames here!

nc, nr, nb = 0, 0, 0
for i in x:
    nc, nr, nb = read_hdr(hdr_fn(x[i])) # stuff in header should all be the same
    output = i + ".bin"
    if not exist(output):
        a = os.system("sbo " + x[i] + " " + output + " 4")
print("cols", nc, "rows", nr)
'''

nc, nr, nb = read_hdr(hdr_fn('RCH_i.bin'))
print(nr, nc, nb)

RCH_i = read_float('RCH_i.bin')
RCV_i = read_float('RCV_i.bin')
RCH_q = read_float('RCH_q.bin')
RCV_q = read_float('RCV_q.bin')

RH = RCH_i + 1j*RCH_q # dual pol data!
RV = RCV_i + 1j*RCV_q

c11, c12 = RH * np.conj(RH), RH * np.conj(RV) # form J
c21, c22 = RV * np.conj(RH), RV * np.conj(RV)

g0 = .5 * (c11 - c22) # calculate stokes params
g1 = g0 - c22
g3 = (1. / (2. * 1j)) * (c21 - c12)
g2 = c21 - (1j * g3)

x = np.array(htrim(np.absolute(c12 - c21)))
x = x.reshape((nc, nr))
plt.imshow(x)
plt.tight_layout()
plt.show()
sys.exit(1)


