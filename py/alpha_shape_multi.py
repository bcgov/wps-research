'''20220227 this program calculates a concave-hull for a set
of points in 2d'''
import os
import sys
import copy
import sha3
import pickle
import fileinput
import alphashape
import numpy as np
from misc import colors
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from misc import args, err

c = colors()
files = [x.strip() for x in os.popen('ls -1 *.pkl').readlines()]
print(files)

print('+w alpha_shape.png')
fig, ax = plt.subplots() # plot init

ci = 0
for fn in files:
    n = fn.split('.')[0]
    print(n, c[ci])
    [X_bak, points, alpha, alpha_shape, patch_alpha] =  pickle.load(open(fn, 'rb'))
    ax.scatter(*zip(*points), color=c[ci]) # plot inputs
    ax.add_patch(PolygonPatch(alpha_shape, alpha=patch_alpha, facecolor=c[ci], edgecolor=c[ci]) ) # plot alpha shape
    ci += 1
plt.savefig("alpha_shape.png")
