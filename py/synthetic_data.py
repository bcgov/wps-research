# produce fake data, square chip, with one-hot encoded groundref
import math
from misc import *
import numpy as np

if exists('stack.bin'): # don't overwrite a stack
    err('file: stack.bin already exists')

nwin = 100 # number of pixels per side of "small" square
nsq = 5 # number of squares/side of big square
nb = nsq # 4 # number of bands, fake multispec data

L = nwin * nsq # image length
print("image dimensions: " + str(L) + ',' + str(L) + ',' + str(nb))
npx = L * L # number of pixels
n_class = nsq * nsq # number of classes
print("n_class", n_class)

n = (npx * (nb + n_class))
print("n_float", n)
d = np.zeros(n, dtype=np.float32)

sigma = 1. / n_class
ci = 0

# simulate the multi/hyperspectral data bands
for k in range(0, nb):
    for i in range(0, L):
        for j in range(0, L):
            bi = math.floor(i / nwin) if (k % 2 == 0) else math.floor(j/nwin)
            d[ci] = np.random.normal((bi -.5 / n_class), sigma)
            ci += 1

# for i in range(0, n_class)
for k in range(0, n_class):
    for i in range(0, L):
        for j in range(0, L):
            class_i = (math.floor(i / nwin) * nsq) + math.floor(j / nwin)
            d[ci] = (1. if (class_i == float(k)) else 0.)
            ci += 1

write_binary(d, 'stack.bin')
write_hdr('stack.hdr', L, L, nb + n_class)

f = open('stack.hdr', 'ab')  # append band name info onto header
f.write('\nband names = {band 1'.encode())

for i in range(1, nb):
    f.write((',\nband ' + str(i + 1)).encode())

for i in range(0, n_class):
    f.write((',\ngt' + chr(97 + i)).encode())
f.write("}".encode())

f.close()
