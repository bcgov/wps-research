# count the number of classes in a ENVI type-4 (i.e., ieee floating-point 32-bit "byte-order=0") image
from misc import * 

if len(args) < 2:
    err("class_count [input binary file]")
    #args.append('20190926kamloops_data/WATERSP.tif_project_4x.bin_sub.bin')

samples, lines, bands, data = read_binary(args[1])

count = {}
for d in data:
    if d not in count:
        count[d] = 0
    count[d] += 1

mean, count_n = 0, 0
class_labels = list(count.keys())  # update for python3
min_lab = max_lab = class_labels[0]
max_c = min_c = count[min_lab]
max_c_lab = min_c_lab = min_lab

print("\tlabel, count")
for c in class_labels:
    print('\t' + str(c) + ',' + str(count[c]))
    if c < min_lab:
        min_lab = c
    if c > max_lab:
        max_lab = c

    count_n += count[c]
    mean += count[c] * c

    if count[c] < min_c:
        min_c, min_c_lab = count[c], c
    if count[c] > max_c:
        max_c, max_c_lab = count[c], c

print("number of class labels,", len(class_labels))
print("min class label: ", min_lab)
print("max class label: ", max_lab)
print("avg class label: ", mean / count_n)
print("most  freq.label: ", max_c_lab, " x", max_c)
print("least freq.label: ", min_c_lab, " x", min_c)
