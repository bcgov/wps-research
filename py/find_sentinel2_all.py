'''centroid.csv has the list of TILE ID with centroid coords incl.

Write the centroid coordinates to "footprint" files and run find_sentinel2.py
on each footprint (actually a centroid but whatever)
'''
import os
import sys
path = os.path.abspath(__file__).split(os.path.sep)[:-1]
path = os.path.sep.join(path) + os.path.sep
print(path)

lines = [x.strip() for x in open(path + 'centroid.csv').readlines()][1:]
for i in range(0, len(lines)):
    w = lines[i].split(',')
    foot_print = 'Intersects(' + w[1] + ',' + w[0] + ')'
    print(foot_print)

    fpifn = '.foot_print_' + str(i)
    open(fpifn, 'wb').write(foot_print.encode())
    a = os.system("python3 " + path + "find_sentinel2.py " + fpifn) 
