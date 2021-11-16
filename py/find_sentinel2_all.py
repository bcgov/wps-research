'''centroid.csv has the list of TILE ID with centroid coords incl.

Write the centroid coordinates to "footprint" files and run find_sentinel2.py
on each footprint (actually a centroid but whatever)'''
import os
import sys
args = sys.argv
print("args", [args])
labels = args[1:]
print("labels", [labels])
path = os.path.abspath(__file__).split(os.path.sep)[:-1]
path = os.path.sep.join(path) + os.path.sep
print(path)

lines = [x.strip() for x in open(path + 'centroid.csv').readlines()][1:]
for i in range(0, len(lines)):
    w = lines[i].split(',')
    label = w[2].strip('"')
    
    if len(labels) > 0:  # selection mode
        if label not in labels:
            continue
    
    print(w)
    
    foot_print = 'Intersects(' + w[1] + ',' + w[0] + ')'
    print(foot_print)

    fpifn = '.foot_print_' + str(i)
    open(fpifn, 'wb').write(foot_print.encode())
    cmd = "python3 " + path + "find_sentinel2.py " + fpifn
    print(cmd)
    a = os.system(cmd)
