''' taking as input:
A)  multispectral data
b)  target (class) annotation file i.e. records are triples consisting of
  text label plus x,y pixel coordinates

perform

1. nearest-centre algorthm to determine seed layer:
   (a) extract data from under targets
   (b) average the data from each target to get initial centres
   (c) find closest centre to each data point, nearest-centre label becomes
         seed layer
   (d) pixel data with NAN or INF in any dimension, are assigned class NAN in
         seed layer

2. perform kmeans iteration (Lloyd's algorithm)

3. output a map with legend'''
from misc import *
from array import array
from sklearn.neighbors import KNeighborsClassifier # didn't use this yet but need to for step 4.

infile = "stack.bin" # default input file
if len(args) > 1:
    infile = args[1]

if len(args) < 2 and not os.path.exists(infile):
    err("kmeans_optimization.py [input image to run kmeans on]")

if not os.path.exists(infile):
    err("failed to find input file: " + infile)

tf = infile + "_targets.csv"
if not os.path.exists(tf):
    err("targets file not found: " + str(tf))

lines = open(tf).read().strip().split("\n")
lines = [line.strip().split(",") for line in lines]
hdr = lines[0] # 'row', 'lin', 'xoff', 'yoff'
i_row, i_lin, i_xof, i_yof, i_lab, sep = hdr.index('row'), hdr.index('lin'), hdr.index('xoff'), hdr.index('yoff'), hdr.index('feature_id'), os.path.sep
path = sep.join(__file__.split(sep)[:-1]) + sep  # path to this file
path = os.path.abspath(os.path.expanduser(os.path.expandvars(path))) + sep
p = path + "../cpp/" # path to c/c++ programs

run("rm -f " + p + "kmeans_iter.exe") # force recompile
if not exist(p + "kmeans_iter.exe"):
    run("g++ -w -O3 " + p + "kmeans_iter.cpp " + p + "misc.cpp -o " + p + "kmeans_iter.exe -lpthread")

run("rm -f " + p + "raster_nearest_centre.exe") # force recompile
if not exist(p + "raster_nearest_centre.exe"):
    run("g++ -w -O3 " + p + "raster_nearest_centre.cpp " + p + "misc.cpp -o " + p + "raster_nearest_centre.exe -lpthread")
cpp_path = p

ncol, nrow, bands = read_hdr(infile[:-3] + 'hdr') # read info from image file
ncol, nrow, bands = int(ncol), int(nrow), int(bands) # convert to int
ncol, nrow, bands, dat_img = read_binary(infile) # read raster data

target_ix = set() # class_label[ix] = label
c, class_label = {}, {} # start K at number of labels
for i in range(1, len(lines)): # iterate over the vector records in target file
    line = lines[i] # csv data record
    label = line[i_lab] # text label from csv.. target label
    x, y = int(line[i_row]), int(line[i_lin]) # pixel image coordinates
    ix = (y * ncol) + x  # linear array coordinates of the image pixl
    if ix < nrow * ncol:  # skip if out of bounds..naive bounds check
        class_label[ix] = label  # map pix/line coords to target label
        c[label] = (c[label] + 1) if label in c else 1 # count label occurrence
        target_ix.add(ix)
K = len(c) # starting number of classes: number of distinct labels that ocurred

# extract data under target points
for ix in target_ix:
    if ix < nrow * ncol:
        for k in range(0, bands):
            pass # print(dat_img[nrow*ncol*k + ix])
    else: print("warning: target out of bounds")

# print out linear array index, and multispectral data contents for each target
target_data = {ix: [dat_img[nrow * ncol * k + ix] for k in range(0, bands)] for ix in target_ix}
for ix in target_data:
    print(ix, target_data[ix])

# mean (by label) of image data under targets (should average or represent small areas instead)
target_mean, target_n = {}, {}
for ix in target_ix:
    L = class_label[ix]
    if L not in target_mean:
        target_n[L] = 0
        target_mean[L] = [0. for k in range(bands)]
    target_n[L] += 1
    for k in range(bands):
        target_mean[L][k] += target_data[ix][k]

for L in target_mean: # divide by n
    if target_n[L] > 0:
        for k in range(bands):
            target_mean[L][k] /= target_n[L]

# print("target_mean", target_mean)
output_file = open("target_mean.dat", "wb")
outputfile2 = open("target_mean_dat.csv", "wb")

ci, od, n_nan = 0, [], 0
for c in target_mean:
    for x in target_mean[c]:
        od.append(x)
    outputfile2.write((c + " " + str(ci) + " " + str(target_mean[c]) + "\n").encode())
    ci += 1

output_file.write(("\n".join([str(x) for x in od])).encode())
output_file.close()
outputfile2.close()

# 1. run nearest centres algorithm
print("calculate seed layer..") # should be parallelized in C/C++
run(cpp_path + "raster_nearest_centre.exe " + infile + " target_mean.dat")

good_labels = np.full(nrow*ncol, float("NaN"),dtype=np.float32) # will store labels of points that get classified here
whoami = os.popen("whoami").read().strip() # user name
class_file = infile + "_kmeans.bin"
seed_file = infile + "_nearest_centre.bin" # input file for kmeans
run(cpp_path + "kmeans_iter.exe " + infile + " " + seed_file + " 1. ")
# next_label += 1 # next iteration would need a higher label if it's reached..
class_file = seed_file
ncol, nrow, bands, data = read_binary(class_file) # read class map result from 

# calculate the set of kmeans labels associated with each class
kmeans_label = {}
for i in range(1, len(lines)): # for each vector point of ours
    line = lines[i]
    x, y = int(line[i_row]), int(line[i_lin]) # rowcol coords for the point
    ix = (y * ncol) + x # print("row", line[i_row], line[i_lin], line[i_xof], line[i_yof], line[i_lab], "class", data[ix])
    if ix < nrow * ncol:
        kmeans_label[ix] = data[ix]

kmeans_label_by_class = {}
for p in class_label:
    L = class_label[p]
    kmeans_label_by_class[L] = [] if (L not in kmeans_label_by_class) else (kmeans_label_by_class[L])
    kmeans_label_by_class[L].append(kmeans_label[p])

for L in kmeans_label_by_class: # what would a vectorization for an op like this look like?
    kmeans_label_by_class[L] = set(kmeans_label_by_class[L])
print("kmeans_label_by_class", kmeans_label_by_class)

'''e.g.:
kmeans_label_by_class {'fireweeddeciduous': {4.0},
                       'blowdownlichen': {2.0},
                       'fireweedgrass': {5.0},
                       'exposed': {3.0},
                       'pineburned': {0.0},
                       'pineburnedfireweed': {1.0}}
'''
found = False
lines = open(infile + "_kmeans.hdr").read()
lines = [line.strip() for line in lines]
for i in range(0, len(lines)):
    line = lines[i]
    w = line.split()
    try:
        if w[0] == "kmeans_label_by_class":
            found = True
    except:
        pass

if not found:
    open(infile + "_kmeans.hdr", "a").write("kmeans_label_by_class " + str(kmeans_label_by_class))

# check if we're done? we're never done!
bad, empty = False, set()
confused_labels = set()
confusion_intersection = []
for k in kmeans_label_by_class:
    kk = kmeans_label_by_class[k]
    for j in kmeans_label_by_class:
        if k == j:
            continue
        kj = kmeans_label_by_class[j]
        inter = kk.intersection(kj)
        if inter != empty:
            bad = True
            confused_labels.add(k)
            confused_labels.add(j)
            confusion_intersection.append(inter)

print("confused_labels", confused_labels) # which labels are confused?
print("labels", list(kmeans_label_by_class.keys()))
all_labels = set(list(kmeans_label_by_class.keys()))
non_confused_labels = all_labels.difference(confused_labels)
print("unconfused labels", non_confused_labels) # which labels aren't confused?

# init seeds array
seeds = np.full(nrow*ncol, float("NaN"),dtype=np.float32) 
kmeans_labels_good, kmeans_labels_confused = set(), set()
for L in non_confused_labels:
    for x in kmeans_label_by_class[L]:
        kmeans_labels_good.add(x)

for L in confused_labels:
    for x in kmeans_label_by_class[L]:
        kmeans_labels_confused.add(x)
print("kmeans_labels_good", kmeans_labels_good)
print("kmeans_labels_confused", kmeans_labels_confused)
for i in range(nrow*ncol):
    if data[i] in kmeans_labels_good:
        good_labels[i] = data[i]
    if data[i] in kmeans_labels_confused:
        seeds[i] = data[i] # points we should probably classify again, e.g. by knn

print("kmeans_labels_good", kmeans_labels_good)
print(kmeans_label_by_class)
write_binary(good_labels, infile + "_good.bin") # relabel the data and output
write_hdr(infile + "_good.hdr", ncol, nrow, 1)

good_kmeans_label_by_class = {}
for label in kmeans_label_by_class:
    if set(kmeans_label_by_class[label]).intersection(set(kmeans_labels_confused)) != set():
        pass  # confused
    else:
        good_kmeans_label_by_class[label] = kmeans_label_by_class[label]
my_class_string = "\nkmeans_label_by_class " + str(kmeans_label_by_class)
open(infile + "_good.hdr", "a").write(my_class_string)

print("n_nan", n_nan)
print("target_ix", target_ix)
print("target_data")
for ix in target_data:
    print(ix, class_label[ix], target_data[ix])

print("kmeans_labels_confused", kmeans_labels_confused)
print("confusion_intersection", confusion_intersection)
print("kmeans_labels_good", kmeans_labels_good)
print("kmeans_labels_confused", kmeans_labels_confused)
print("kmeans_label_by_class", kmeans_label_by_class)

run('python3 ' + cpp_path + "../py/read_multi.py " + infile + "_kmeans.bin 1")
run("eog " + infile + "_kmeans.bin.png")

''' run knn on data with confused labels and splice results back into map? Something to try'''

'''
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsClassifier
>>> neigh = KNeighborsClassifier(n_neighbors=3)
>>> neigh.fit(X, y)
KNeighborsClassifier(...)
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[0.66666667 0.33333333]]'''

neigh = KNeighborsClassifier(n_neighbors=2)
# neigh.fit(X, y)
