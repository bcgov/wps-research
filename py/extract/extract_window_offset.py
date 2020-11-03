''' calculate x, y pixel offset coordinates for points within a given radius R...
... i.e., points within R distance away, relative to a centre coordinate (0,0) 
   assuming the image resolution is ____ 

usage: python3 window.py [radius (m)] [image resolution (m)]


e.g. to calculate the x,y coordinate offsets for a round window of radius 30, 
for a sensor with resolution 10m:
    python3 window.py 30 10 

Offsets are stored in a file

'''

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
args = sys.argv

xmin = 0 # search grid start (don't change)
resolution = 10 # 10 m res for Sentinel2 (change using args)
dist_max = 30. # distance from centre allowed (change using args)

if len(args) > 1: dist_max = float(args[1])
if len(args) > 2: resolution = float(args[2])

xmax = 2. * (math.ceil(dist_max / resolution))
xmax = int(xmax) # print(xmax)

X, Y = [], []
for i in range(xmin, xmax + 1):
    for j in range(xmin, xmax + 1):
        x = (i - (xmax/2)) * resolution
        y = (j - (xmax/2)) * resolution
        d = math.sqrt(x*x + y*y)
    
        if d == 0.: # tackle centre point separately
            continue

        if d <= dist_max:
            # print(x,y)
            X.append(x)
            Y.append(y)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(-(xmax / 2.) * resolution, (xmax / 2) * resolution, resolution))
ax.set_yticks(np.arange(-(xmax / 2.) * resolution, (xmax / 2) * resolution, resolution))

# plot the offsets
plt.scatter(X,Y)
plt.grid()
d_string = str(dist_max)
if float(int(dist_max)) == dist_max:
    d_string = str(int(dist_max))

window_png = "window_" + d_string + ".png"
print("+w", window_png)
plt.savefig(window_png)
npts = len(X)

x = np.array(X) / resolution
y = np.array(Y) / resolution
x = [int(i) for i in x.tolist()]
y = [int(i) for i in y.tolist()]
print(x)
print(y)
print("number of points:", npts)

open(".x_off", "wb").write((str(x).strip('[').strip(']').strip().replace(' ','')).encode())
open(".y_off", "wb").write((str(y).strip('[').strip(']').strip().replace(' ','')).encode())
print("+w .x_off")
print("+w .y_off")
