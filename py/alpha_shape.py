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

print(" ****** X=", X)
print(type(X))
X = X.strip()
print(X)
X = X.split()
print(X)
N = int(X[1])
print(N)
X = X[2:]
print(X)

if len(X) != 2*N:
    err("bad data count")





