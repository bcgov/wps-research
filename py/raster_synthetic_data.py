# produce fake data, square chip, with one-hot encoded groundref
import math
import copy
from misc import *
import numpy as np

if exists('stack.bin'): # don't overwrite a stack
    err('file: stack.bin already exists')

nwin = 50 # number of pixels per side of "small" square
nsq = 3 # number of squares/side of big square
nb = 3 # number of bands, fake multispec data

L = nwin * nsq # image length
npx = L * L # number of pixels
n_class = nsq * nsq # number of classes
print("n_class", n_class, "\n", "image dimensions: " + str(L) + ',' + str(L) + ',' + str(nb))

n = npx * (nb + n_class)
print("n_float", n)
d = np.zeros(n, dtype=np.float32)

sigma, ci = 1 / n_class, 0

# simulate the multi/hyperspectral data bands
for k in range(0, nb):
    for i in range(0, L):
        for j in range(0, L):
            bi = math.floor(i / nwin)
            if((k % nsq) == 1): bi = math.floor(j / nwin)
            if((k % nsq) == 2): bi = math.floor((L - i - 1) / nwin)
            d[ci] = np.random.normal((bi -.5 / n_class), sigma)
            ci += 1

for k in range(0, n_class):
    for i in range(0, L):
        for j in range(0, L):
            class_i = (math.floor(i / nwin) * nsq) + math.floor(j / nwin)
            d[ci] = (1. if (class_i == float(k)) else 0.)
            ci += 1

dp = copy.deepcopy(d) # make data dirty by making groundref classes slightly overlap!

for k in range(0, n_class):
    for i in range(0, L):
        for j in range(0, L):
            if d[npx * (nb + k) + ((i * L) + j)] == 1.:
                # grow
                wg = 3
                for di in range(- wg, wg + 1):
                    ii = i + di
                    if ii < 0 or ii >= L: continue
                    for dj in range(- wg, wg + 1):
                        jj = j + dj
                        if jj < 0 or jj >= L: continue 
                        # print("i", i, "j", j, "di", di, "ii", ii, "dj", dj, "jj", jj)
                        dp[npx * (nb + k) + ((ii * L) + jj)] = 1.

write_binary(dp, 'stack.bin')
write_hdr('stack.hdr', L, L, nb + n_class)
f = open('stack.hdr', 'ab')  # append band name info onto header
f.write('\nband names = {band 1'.encode())
for i in range(1, nb): f.write((',\nband ' + str(i + 1)).encode())
for i in range(0, n_class): f.write((',\n' + chr(97 + i)).encode())
f.write("}".encode())
f.close()
