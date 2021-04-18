import os
import sys
path = os.path.abspath(__file__).split(os.path.sep)
path = path[:-1] + ['centroid.csv']

lines = open(os.path.sep.join(path)).readlines()
lines = [x.strip() for x in lines]
lines = lines[1:]
for line in lines:
    print(line)

    w = line.split(',')
    foot_print = 'Intersects(' + w[1] + ',' + w[0] + ')'
    print(foot_print)
    # foot_print = 'Intersects(51.0602686,-120.9083258)' # default location: Kamloops
    # VICTORIA: (48.4283334, -123.3647222)



