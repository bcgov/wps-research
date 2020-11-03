import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

xmin = 0
xmax = 10
resolution = 10 # 10 m res for Sentinel2

dist_max = 30. # distance from centre allowed

args = sys.argv
if len(args) > 1:
    dist_max = float(args[1])

X, Y = [], []
for i in range(xmin, xmax + 1):
    for j in range(xmin, xmax + 1):
        x = (i - (xmax/2)) * resolution
        y = (j - (xmax/2)) * resolution
        
        d = math.sqrt(x*x + y*y)

        if d <= dist_max:
            print(x,y)
            X.append(x)
            Y.append(y)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(-(xmax / 2.) * resolution, (xmax / 2) * resolution, resolution))
ax.set_yticks(np.arange(-(xmax / 2.) * resolution, (xmax / 2) * resolution, resolution))

plt.scatter(X,Y)
plt.grid()
plt.savefig("window_" + str(dist_max) + ".png")



