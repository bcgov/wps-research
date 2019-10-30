# count the number of classes in a ENVI type-4 (i.e., ieee floating-point 32-bit "byte-order=0") image
from misc import * 

if len(args) < 2:
    err("class_count [input binary file]")
    #args.append('20190926kamloops_data/WATERSP.tif_project_4x.bin_sub.bin')

fn = args[1]
samples, lines, bands, data = read_binary(fn)

count = {}
for d in data:
    if d not in count:
        count[d] = 0
    count[d] += 1

class_labels = count.keys()
print "number of class labels,", len(class_labels)
print "\tlabel,count"

for c in class_labels:
    print '\t' + str(c) + ',' + str(count[c])
