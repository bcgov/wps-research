# count the number of classes in a ENVI type-4 (i.e., ieee floating-point 32-bit "byte-order=0") image
from misc import * 

if len(args) < 2:
    args.append('20190926kamloops_data/WATERSP.tif_project_4x.bin_sub.bin')

fn = args[1]
data = read_float(fn)

count = {}
for d in data:
    if d not in count:
        count[d] = 0
    count[d] += 1

class_labels = count.keys()
print "number of distinct floats,", len(class_labels)
print "\tlabel,count"

for c in class_labels:
    print '\t' + str(c) + ',' + str(count[c])

print "count of counts,", str(len(count.keys()))
