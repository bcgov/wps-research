print("load deps..")
import copy
import fileinput
import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch

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
alpha = 0.95 * alphashape.optimizealpha(points) # optimal alpha
print("finding alpha shape..")
alpha_shape = alphashape.alphashape(points, alpha); # create alpha shape
print(alpha_shape)
if True:
    print('+w alpha_shape.png')
    fig, ax = plt.subplots() # plot init
    ax.scatter(*zip(*points)) # plot inputs
    ax.add_patch(PolygonPatch(alpha_shape, alpha=.2)) # plot alpha shape
    plt.savefig("alpha_shape.png")
''' don't forget to revisit:
(2007) Concave hull: A k-nearest neighbours approach for
the computation of the region occupied by a set of points.'''
