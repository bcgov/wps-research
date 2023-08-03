'''20220227 this program calculates a concave-hull for a set
of points in 2d'''
import copy
import pickle
import fileinput
import alphashape
import numpy as np
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from misc import args, err, run, pd, read_binary

if len(args) < 2:
    err("alpha_shape.py [input file name] # [optional arg: alpha value]")

fn = args[1] # fn = open('alpha_shape_input_file.txt').read().strip()
pfn = fn + '_alpha_shape.pkl'
samples, lines, bands, data = read_binary(fn)
samples, lines, bands = int(samples), int(lines), int(bands)

if bands != 1:
    err('expected 1-band image')

points = []
for i in range(lines):
    for j in range(samples):
        if data[i * samples + j] == 1.:
            points.append((i, j))
print(points)
points = np.array(points)

# print("optimizing alpha..")
alpha = 1. / 25 # 1./10.; #1./500.; # 1. / 50. # 0.95 * alphashape.optimizealpha(points) # optimal alpha
if len(args) > 2:
    alpha = float(args[2])

patch_alpha = .2
print("finding alpha shape..")
alpha_shape = alphashape.alphashape(points, alpha); # create alpha shape
print("alpha_shape", alpha_shape)
print('+w', pfn) # write stuff to pickle file
pickle.dump([points, alpha, alpha_shape, patch_alpha], open(pfn, 'wb'))

alpha_pts = str(alpha_shape)# .strip('P').strip('O').strip('L').strip('Y').strip('G').strip('O').strip('N').strip().strip('(').strip('(').strip(')').strip(')').replace(',', '')
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
