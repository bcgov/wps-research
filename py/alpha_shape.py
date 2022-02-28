import alphashape
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import copy
import fileinput

ci = 0
X = None
for line in fileinput.input():
    if ci == 0:
        X = copy.deepcopy(line)
    ci += 1

# print(" ****** X=", X)
# print(type(X))
X = X.strip()
# print(X)
X = X.split()
# print(X)
N = int(X[1])
# print(N)
X = X[2:]
# print(X)

if len(X) != 2*N:
    err("bad data count")
X = [float(x) for x in X]
ci = 0
points = []
for i in range(N):
    points.append((X[ci], X[ci+1]))
    ci += 2

print("optimizing alpha..")
alpha = 0.95 * alphashape.optimizealpha(points) # optimal alpha
print("finding alpha shape..")
alpha_shape = alphashape.alphashape(points, alpha); # create alpha shape
print(alpha_shape)
if False:
    fig, ax = plt.subplots() # plot init
    ax.scatter(*zip(*points)) # plot inputs
    ax.add_patch(PolygonPatch(alpha_shape, alpha=.2)) # plot alpha shape
    plt.savefig("alpha_shape.png")
