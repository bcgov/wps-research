import os
import sys
path = os.path.abspath(__file__) + os.path.sep
print(path)

lines = open(path + "centroids.csv").readlines()
lines = [x.strip() for x in lines]
lines = lines[1:]
for line in lines:
    print(line)




