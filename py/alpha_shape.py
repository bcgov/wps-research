'''20220227 this program calculates a concave-hull for a set
of points in 2d'''
import copy
import fileinput
import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from misc import args, err

if len(args) < 2:
    err('python3 alphashape.py # please check ../cpp/binary_hull.cpp for parameters')

ci = 0
X = None
for line in fileinput.input():
    if ci == 0:
        X = copy.deepcopy(line)
    ci += 1

X = X.strip().split()
N = int(X[1])
X = X[2:]

if len(X) != 2*N:
    err("bad data count")
X = [float(x) for x in X]
ci = 0
points = []
for i in range(N):
    points.append((X[ci + 1], -X[ci]))
    ci += 2

print("optimizing alpha..")
alpha = .05 # 0.95 * alphashape.optimizealpha(points) # optimal alpha
print("finding alpha shape..")
alpha_shape = alphashape.alphashape(points, alpha); # create alpha shape
print(alpha_shape)
if True:
    print('+w alpha_shape.png')
    fig, ax = plt.subplots() # plot init
    ax.scatter(*zip(*points)) # plot inputs
    ax.add_patch(PolygonPatch(alpha_shape, alpha=.2)) # plot alpha shape
    plt.savefig("alpha_shape.png")
''' remember to look at:

[1] (2007) Concave hull: A k-nearest neighbours approach for
the computation of the region occupied by a set of points.
    [2] "On the shape of a set of points in the plane" Herbert Edelsbrunner, et al" 
'''
