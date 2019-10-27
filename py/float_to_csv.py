# convert an image file to csv
from misc import * 

fn = args[1]
data = read_float(fn)

print fn
for d in data:
    if float(int(d)) == float(d):
        print int(d)
    else:
        print float(d)
