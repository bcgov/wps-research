'''20220227 this program calculates a concave-hull for a set
of points in 2d'''
import copy
import pickle
import fileinput
import alphashape
import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from misc import args, err, run, pd

fn = open('alpha_shape_input_file.txt').read().strip()

pfn = fn + '_alpha_shape.pkl'

'''
if len(args) < 2:
    err('python3 alphashape.py ' +
        '# please check ../cpp/binary_hull.cpp for parameters')
'''
ci, x = 0, None
for line in fileinput.input():
    if ci == 0: 
        X = copy.deepcopy(line)
    ci += 1

X_bak = copy.deepcopy(X)
X = X.strip().split()
N = int(X[1])
X = X[2:]

if len(X) != 2*N:
    err("bad data count")
X = [float(x) for x in X]

ci, points = 0, []
for i in range(N):
    #points.append((X[ci + 1], -X[ci]))
    points.append((X[ci], X[ci+1]))
    ci += 2

points = np.array(points)

# print("optimizing alpha..")
alpha = 1. / 50. # 0.95 * alphashape.optimizealpha(points) # optimal alpha
patch_alpha = .2
print("finding alpha shape..")
alpha_shape = alphashape.alphashape(points, alpha); # create alpha shape
print("alpha_shape", alpha_shape)
print('+w', pfn) # write stuff to pickle file
pickle.dump([X_bak, points, alpha, alpha_shape, patch_alpha], open(pfn, 'wb'))

alpha_pts = str(alpha_shape).strip('P').strip('O').strip('L').strip('Y').strip('G').strip('O').strip('N').strip().strip('(').strip('(').strip(')').strip(')').replace(',', '')
apfn = fn + '_alpha_points.txt'
print('+w', apfn)
open(apfn, 'wb').write(alpha_pts.encode())
if True:
    pfn = fn + '_alpha_points.png'
    print('+w', pfn)
    fig, ax = plt.subplots() # plot init
    ax.scatter(*zip(*points)) # plot inputs
    ax.add_patch(PolygonPatch(alpha_shape, alpha=patch_alpha)) # plot alpha shape
    plt.savefig(pfn)

''' remember to look at:
[1] (2007) Concave hull: A k-nearest neighbours approach for
the computation of the region occupied by a set of points.
    [2] "On the shape of a set of points in the plane" Herbert Edelsbrunner, et al" 
''' 
