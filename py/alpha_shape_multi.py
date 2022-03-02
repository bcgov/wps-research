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
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from misc import args, err

files = [x.strip() for x in os.popen('ls -1 *.pkl').readlines()]
print(files)

for fn in files:
    n = fn.split('.')[0]

pickle.dump([X_bak, points, alpha, alpha_shape, patch_alpha], open(pfn, 'wb'))
sys.exit(1)

if True:
    print('+w alpha_shape.png')
    fig, ax = plt.subplots() # plot init
    ax.scatter(*zip(*points)) # plot inputs
    ax.add_patch(PolygonPatch(alpha_shape, alpha=patch_alpha)) # plot alpha shape
    plt.savefig("alpha_shape.png")
