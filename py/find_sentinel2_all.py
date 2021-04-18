import os
import sys
path = os.path.abspath(__file__).split(os.path.sep)[-1]
path = os.path.sep.join(path) + os.path.sep

lines = open(path + 'centroid.csv').readlines()
lines = [x.strip() for x in lines]
lines = lines[1:]

for i in range(0, len(lines)):
    line = lines[i]
    w = line.split(',')
    foot_print = 'Intersects(' + w[1] + ',' + w[0] + ')'
    print(foot_print)

    fpifn = '.foot_print_' + str(i)

    open(fpifn, 'wb').write(foot_print.encode())

    a = os.system("python3 " + path + "find_sentinel2.py " + fpifn) 

    # foot_print = 'Intersects(51.0602686,-120.9083258)' # default location: Kamloops
    # VICTORIA: (48.4283334, -123.3647222)



