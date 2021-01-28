''' optimization for hkmeans algorithm, given target file

1. set seed layer:
   (a) extract data from under targets
   (b) average the data from each target to get initial centres
   (c) find closest centre to each data point (this becomes seed layer)
   (d) data with NAN or INF are assigned class NAN

first iteration start with seed layer as above.

2. iterations:
  (a) run kmeans_iter
  (b) read revised labels
  (c) for each target label, form list with no repetitions of kmeans labels associated with that label
  (d) check for intersections. If there are no intersections we're done, proceed to 3. 
  (e) define intersection list: list target labels (and associated KMEANS labels) associated with the intersection
  (f) initialize K as length of intersection list (number of confused target classes).. print out intersection list (and target class labels + kmeans labels outside of intersect list)
        - define seed layer so that kmeans classes outside of those associated with confused target classes are given the label NAN
        - i.e., only refine confused classes!
        - for kmeans classes associated with confused target classes, seed by calculating nearest labels to the target-class-means already calculated..
  (g) if intersection list was unchanged from last iteration, increase number of K? Seed uniformly?
  (h) record intersection list for reuse..

3. Cleanup
  (a) list all labels that some pixel(s) actually use (incl. NAN)
  (b) 


method above will be out of date... need to revise documentation
'''
from misc import *
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
    else:
        print("warning: target out of bounds")

target_data = {ix: [dat_img[nrow*ncol*k + ix] for k in range(0, bands)] for ix in target_ix}

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

print("calculate seed layer..") # should be parallelized in C/C++
if not exist(infile + "_seed.bin"):
    # form seed layer by choosing each point's label by taking the closest mean (where the mean is calculated over different points with the same label)
    seed = []
    dx = math.ceil(nrow * ncol / 1000.)
    for ix in range(nrow * ncol):
        if ix % dx == 0:
            print(100. * ix / (nrow * ncol), "%")
        bad = False
        for k in range(bands):
            if math.isnan(dat_img[nrow*ncol*k + ix]):
                bad = True
        if bad:
            seed.append(float("NaN"))
            continue    

        min_c, min_d = None, float("NaN")
        ci = 0
        for c in target_mean:
            d = 0.
            for k in range(bands):
                dd = target_mean[c][k] - dat_img[nrow*ncol*k + ix]
                d += dd * dd
            d = math.sqrt(d)
            if min_c is None:
                min_c = ci
                min_d = d
            else:
                if d < min_d:
                    min_d = d
                    min_c = ci # represent class by number
            ci += 1
        seed.append(min_c)

    print("len(seed)", len(seed))
    write_binary(np.array(seed, dtype=np.float32), infile + "_seed.bin")
    write_hdr(infile + "_seed.hdr", ncol, nrow, 1) 

go = True
iteration = 0
next_label = K
print("next_label", next_label)
good_labels = np.full(nrow*ncol, float("NaN"),dtype=np.float32) #None # will store labels of points that are finally classified, here..

while go: # could have turned this into a recursive function!
    whoami = os.popen("whoami").read().strip()
    class_file = infile + "_kmeans.bin"
    seed_file = infile + "_seed.bin"
    if iteration > 0:
        seed_file = infile + "_reseed.bin" #  class_file
    run(cpp_path + "kmeans_iter.exe " + infile + " " + seed_file + " 1. " + ("" if iteration == 0 else (" " + str(next_label))))
    print(">>>>>>> DONE RUN >>>>>>>>>>!!!!!!!%^&*%^&*%^&*")
    next_label += 1 # next iteration would need a higher label if it's reached..
    ncol, nrow, bands, data = read_binary(class_file) # read the class map data resulting from kmeans

    # calculate the set of kmeans labels associated with each class
    kmeans_label = {}
    for i in range(1, len(lines)): # for each vector point of ours
        line = lines[i]
        x, y = int(line[i_row]), int(line[i_lin]) # rowcol coords for the point
        ix = (y * ncol) + x # print("row", line[i_row], line[i_lin], line[i_xof], line[i_yof], line[i_lab], "class", data[ix])
        if ix < nrow * ncol: kmeans_label[ix] = data[ix]

    kmeans_label_by_class = {}
    for p in class_label:
        L = class_label[p]
        kmeans_label_by_class[L] = [] if (L not in kmeans_label_by_class) else (kmeans_label_by_class[L])
        kmeans_label_by_class[L].append(kmeans_label[p])

    for L in kmeans_label_by_class: # what would a vectorization for an op like this look like?
        kmeans_label_by_class[L] = set(kmeans_label_by_class[L])
    print("kmeans_label_by_class", kmeans_label_by_class)


    # kmeans_label_by_class {'fireweeddeciduous': {4.0}, 'blowdownlichen': {2.0}, 'fireweedgrass': {5.0}, 'exposed': {3.0}, 'pineburned': {0.0}, 'pineburnedfireweed': {1.0}}
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
    for k in kmeans_label_by_class:
        kk = kmeans_label_by_class[k]
        for j in kmeans_label_by_class:
            if k == j: continue
            kj = kmeans_label_by_class[j]
            inter = kk.intersection(kj)
            if inter != empty:
                bad = True
                confused_labels.add(k)
                confused_labels.add(j)
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

    write_binary(seeds, infile + "_reseed.bin") # relabel the data and output
    write_hdr(infile + "_reseed.hdr", ncol, nrow, 1)
    sys.exit(1)

     
    # DONT FORGET THE MERGE STEP@@@@@@@@@@
   

    # before returning from the recursive function, merge the labels that are good with the labels that are returned from refinement..

    # recurse into equivalence classes of things that are confused.. if equivalence classes of confused labels are unchanged after an iteration, uniform initialize over the selected data and add +1 to K
    # extract label data from non-confused classes! Those won't be changed.
    # rewrite seed file.. NB need to keep the label data that doesn't need to be refined, where does that go...
    # recurse into confused sets...RECURSION!!!!! Call function onto each equivalence class of confused sets..
  
    if not bad:
        # CLEAN UP AND FINISH....
        print("good")
        # clean up labels so that everything outside the known classes is NAN, and all clusters for class get same label..
        used_labels = set()
        for k in kmeans_label_by_class:
            for j in kmeans_label_by_class[k]:
                used_labels.add(j)
        print("used_labels", used_labels)        

        lookup = {}
        for k in range(0, K):
            k = float(k)
            if k not in used_labels:
                lookup[k] = float("NaN")
        ci = 0
        for k in kmeans_label_by_class:
            for j in kmeans_label_by_class[k]:
                lookup[j] = ci
            ci += 1

        print("lookup", lookup)  # now apply lookup
        for i in range(0, nrow* ncol):
            data[i] = lookup[good_labels[i]]

        write_binary(data, class_file) # relabel the data and output
        break # kmeans_label_by_class: {'fireweedandaspen': [0.0], 'blowdownwithlichen': [1.0, 0.0], 'pineburned': [1.0, 1.0, 1.0]}
    
    K += 1 # try adding a class!
    iteration += 1
    
print("kmeans_label_by_class", kmeans_label_by_class, "lookup", lookup)

# translate the lookup
for label in kmeans_label_by_class:
    labels = list(kmeans_label_by_class[label])
    labels = [lookup[i] for i in labels]
    kmeans_label_by_class[label] = set(labels)
print("kmeans_label_by_class", kmeans_label_by_class)


#  do the plotting! 

import matplotlib.pyplot as plt
hdr = hdr_fn(infile)
npx = nrow * ncol
data = data.reshape((nrow, ncol))

fig, ax = plt.subplots()
img = ax.imshow(data, cmap='Spectral')
# ax.set_aspect("auto")
cbar = plt.colorbar(img)#  .legend([0, 1, 2, 3], ['0', '1', '2', '3'])\
tick_labels = [] # "noise"]
ci = 0 
for label in kmeans_label_by_class:
    tick_labels.append(label)
    x = kmeans_label_by_class[label]
    if set([ci]) != x:
        print(str(set([ci])), str(x))
        err("color index problem")
    ci += 1
cbar.set_ticks(np.arange(len(tick_labels)))
print("tick_labels", tick_labels)
cbar.ax.set_yticklabels(tick_labels) #"bad", "good", "other", "more", "what"])
plt.tight_layout()
plt.show()
# run("python3 " + path + "read_multi.py " + infile + "_kmeans.bin")
