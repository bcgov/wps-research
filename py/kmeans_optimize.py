''' seed kmeans algorithm on multispectral data, given target (class annotation) file

1. set seed layer:
   (a) extract data from under targets
   (b) average the data from each target to get initial centres
   (c) find closest centre to each data point (this becomes seed layer)
   (d) data with NAN or INF are assigned class NAN

'''
from misc import *
from sklearn.neighbors import KNeighborsClassifier

infile = "stack.bin" # default input file
if len(args) > 1: infile = args[1]
if len(args) < 2 and not os.path.exists(infile):
    err("kmeans_optimization.py [input image to run kmeans on]")
if not os.path.exists(infile): err("failed to find input file: " + infile)

tf = infile + "_targets.csv"
if not os.path.exists(tf): error("targets file not found: " + str(tf))
lines = open(tf).read().strip().split("\n")
lines = [line.strip().split(",") for line in lines]
hdr = lines[0] # 'row', 'lin', 'xoff', 'yoff'
i_row, i_lin, i_xof, i_yof, i_lab, sep = hdr.index('row'), hdr.index('lin'), hdr.index('xoff'), hdr.index('yoff'), hdr.index('feature_id'), os.path.sep
path = sep.join(__file__.split(sep)[:-1]) + sep  # path to this file
print("path", path)
path = os.path.abspath(os.path.expanduser(os.path.expandvars(path))) + sep

print(path)
p = path + "../cpp/"
run("rm -f " + p + "kmeans_iter.exe")
if not exist(p + "kmeans_iter.exe"):
    run("g++ -w -O3 " + p + "kmeans_iter.cpp " + p + "misc.cpp -o " + p + "kmeans_iter.exe -lpthread")

run("rm -f " + p + "raster_nearest_centre.exe")
if not exist(p + "raster_nearest_centre.exe"):
    run("g++ -w -O3 " + p + "raster_nearest_centre.cpp " + p + "misc.cpp -o " + p + "raster_nearest_centre.exe -lpthread")
cpp_path = p

ncol, nrow, bands = read_hdr(infile[:-3] + 'hdr') # read info from image file
ncol, nrow, bands = int(ncol), int(nrow), int(bands)
ncol, nrow, bands, dat_img = read_binary(infile)

# class_label[ix] = label
target_ix = set()
c, class_label = {}, {} # start K at number of labels
for i in range(1, len(lines)): # iterate over the vector labels
    line = lines[i] # csv data
    label = line[i_lab] # text label from csv.. target label
    x, y = int(line[i_row]), int(line[i_lin]) # image coordinates of point
    ix = (y * ncol) + x  # linear image coordinates of the point!
    if ix < nrow * ncol:  # skip if out of bounds
        class_label[ix] = label  # map pix/line coords to target label
        c[label] = (c[label] + 1) if label in c else 1 # count the occurrence for each label
        if ix < nrow*ncol:
            target_ix.add(ix)
K = len(c) # starting number of classes: number of distinct labels that ocurred

# extract data under target points
for ix in target_ix:
    if ix < nrow * ncol:
        for k in range(0, bands):
            pass # print(dat_img[nrow*ncol*k + ix])
    else: print("warning: target out of bounds")

target_data = {ix: [dat_img[nrow * ncol * k + ix] for k in range(0, bands)] for ix in target_ix}
for ix in target_data:
    print(ix, target_data[ix])

# mean (by label) of image data under targets (should average over small windows)..
target_mean = {}
target_n = {}
for ix in target_ix:
    L = class_label[ix]
    if L not in target_mean:
        target_n[L] = 0
        target_mean[L] = [0. for k in range(bands)]
    target_n[L] += 1
    for k in range(bands):
        target_mean[L][k] += target_data[ix][k]

# divide by n
for L in target_mean:
    if target_n[L] > 0:
        for k in range(bands):
            target_mean[L][k] /= target_n[L]


# print("target_mean", target_mean)
from array import array
output_file = open("target_mean.dat", "wb")
outputfile2 = open("target_mean_dat.csv", "wb")

ci = 0
od = []
for c in target_mean:
    for x in target_mean[c]:
        od.append(x)
    outputfile2.write((c + " " + str(ci) + " " + str(target_mean[c]) + "\n").encode())
    ci += 1

output_file.write(("\n".join([str(x) for x in od])).encode())
output_file.close()
outputfile2.close()

n_nan = 0
print("calculate seed layer..") # should be parallelized in C/C++
run(cpp_path + "raster_nearest_centre.exe " + infile + " target_mean.dat")

go = True
iteration = 0
next_label = K
print("next_label", next_label)
good_labels = np.full(nrow*ncol, float("NaN"),dtype=np.float32) #None # will store labels of points that are finally classified, here..

while go: # could have turned this into a recursive function!
    whoami = os.popen("whoami").read().strip()
    class_file = infile + "_kmeans.bin"
    seed_file = infile + "_nearest_centre.bin"
    if iteration > 0:
        seed_file = infile + "_reseed.bin" #  class_file
    run(cpp_path + "kmeans_iter.exe " + infile + " " + seed_file + " 1. ") # + ("" if iteration == 0 else (" " + str(next_label))))
    next_label += 1 # next iteration would need a higher label if it's reached..
    ncol, nrow, bands, data = read_binary(class_file) # read the class map data resulting from kmeans

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


    '''kmeans_label_by_class {'fireweeddeciduous': {4.0}, 'blowdownlichen': {2.0},
                              'fireweedgrass': {5.0}, 'exposed': {3.0}, 'pineburned': {0.0},
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

    # check if we're done
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
    print("confused_labels", confused_labels)
    print("labels", list(kmeans_label_by_class.keys()))
    all_labels = set(list(kmeans_label_by_class.keys()))
    non_confused_labels = all_labels.difference(confused_labels)
    print("unconfused labels", non_confused_labels)

    '''
      1. store the "good" labels to keep (final)... write out good label map.... (next iteration will need to merge with that one!!!!!!)
      2. for the confused classes, write a new seed file with original seeds PLUS ONE SEED one more
      3. new iteration should shard off the good stuff (if there is any) and keep on dividing the stuff that isn't good yet..
    '''

    seeds = np.full(nrow*ncol, float("NaN"),dtype=np.float32)  # new seeds will be saved here..

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
            seeds[i] = data[i]

    # put the good stuff on ice, now randomly select centres and crank up the N until something pops off...
    # each iteration, save the good stuff. and attack the rest again!
    print("kmeans_labels_good", kmeans_labels_good)
    print(kmeans_label_by_class)
    write_binary(good_labels, infile + "_good.bin") # relabel the data and output
    write_hdr(infile + "_good.hdr", ncol, nrow, 1)
    good_kmeans_label_by_class = {}
    for label in kmeans_label_by_class:
        if set(kmeans_label_by_class[label]).intersection(set(kmeans_labels_confused)) != set():
            pass # confused
        else:
            good_kmeans_label_by_class[label] = kmeans_label_by_class[label]
    open(infile + "_good.hdr", "a").write("\nkmeans_label_by_class " + str(kmeans_label_by_class))

    # write_binary(seeds, infile + "_reseed.bin") # relabel the data and output
    # write_hdr(infile + "_reseed.hdr", ncol, nrow, 1)
    print("n_nan", n_nan)
    # RUN KNN ON DATA WITH CONFUSED LABELS ONLY!!!!
    # SPLICE THE RESULTS BACK INTO THE CLASS MAP
    # LOOK AT BRAD NEW DATA

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

    print("target_ix", target_ix)
    print("target_data")
    for ix in target_data:
        print(ix, class_label[ix], target_data[ix])

    print("kmeans_labels_confused", kmeans_labels_confused)
    print("confusion_intersection", confusion_intersection)
    print("kmeans_labels_good", kmeans_labels_good)
    print("kmeans_labels_confused", kmeans_labels_confused)
    print("kmeans_label_by_class", kmeans_label_by_class)

    run(cpp_path + "../py/read_multi.py " + infile +"_kmeans.bin 1")
    run("eog " + infile + "_kmeans.bin.png")
    neigh = KNeighborsClassifier(n_neighbors = 2)
    # neigh.fit(X, y)
    sys.exit(1)


