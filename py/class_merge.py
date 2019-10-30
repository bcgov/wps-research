# guess at what the "null" / non-positive class is: output a new map with the other classes merged. Result: a binary class map
from misc import * 

if len(args) < 2:
    err("class_merge [input file name") # args.append('20190926kamloops_data/WATERSP.tif_project_4x.bin_sub.bin')

fn = args[1]
hfn = hdr_fn(fn)
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

print "class_labels", class_labels

print "assume most-frequent label is non-positive.."

max_label, max_count = None, None

# detect non-positive class
if 0. not in class_labels:
    max_label, max_count = class_labels[0], count[class_labels[0]]
    for c in class_labels:
        if count[c] > max_count:
            max_label, max_count = class_labels[c], count[c]

    print "\tmax_label,max_count"
    print "\t", str(max_label) + ',' + str(max_count)

# output a binary class map for merged class
output = copy.deepcopy(data)
for i in range(0, len(data)):
    output[i] = 0. if data[i] == max_label else 1.

# write output and copy header
ofn, ohfn = fn + '_binary.bin', fn + '_binary.hdr'
write_binary(output, ofn)
run('cp ' + hfn + ' ' + ohfn)

# create a binary map for each separate class
for c in class_labels:
    output = copy.deepcopy(data)
    for i in range(0, len(data)):
        output[i] = 0. if data[i] == c else 1. 
    
    # write output and copy header
    ofn, ohfn = fn + '_binary_' + str(c) + '.bin', fn + '_binary_' + str(c) + '.hdr'
    write_binary(output, ofn)
    run('cp ' + hfn + ' ' + ohfn)

