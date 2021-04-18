import os
import sys
path = os.path.abspath(__file__).split(os.path.sep)
path = path[:-1] + ['centroids.csv']

lines = open(os.path.sep.join(path)).readlines()
lines = [x.strip() for x in lines]
lines = lines[1:]
for line in lines:
    print(line)




