import os
import sys
lines = [x.strip() for x in open("spectral.csv").readlines()]

for line in lines:
    w = line.split()
    if len(w) == 5:
        pass
    elif len(w) == 3:
        w = [w[0], '', '', w[1], w[2]]
    else:
        print("error"); sys.exit(1)
    print(','.join(w))
