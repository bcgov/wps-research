''' optimization for k-means algorithm, given target file'''
from misc import *

infile = "stack.bin"

if len(args) < 2 and not os.path.exists("stack.bin"):
    err("kmeans_optimization.py [input image to run kmeans on]")
else:
    if len(args) > 1:
        infile = args[1]

if not os.path.exists(infile):
    err("failed to find input file: " + infile)

tf = infile + "_targets.csv"
if not os.path.exists(tf):
    error("targets file not found: " + str(tf))
lines = open(tf).read().strip().split("\n")
lines = [line.strip().split(",") for line in lines]
hdr = lines[0] # 'row', 'lin', 'xoff', 'yoff'
i_row, i_lin, i_xof, i_yof, i_lab, sep = hdr.index('row'), hdr.index('lin'), hdr.index('xoff'), hdr.index('yoff'), hdr.index('feature_id'), os.path.sep
path = sep.join(__file__.split(sep)[:-1]) + sep  # path to this file
print("path", path)
path = os.path.abspath(os.path.expanduser(os.path.expandvars(path))) + sep

# read info from image file
ncol, nrow, bands = read_hdr(infile[:-3] + 'hdr')
ncol, nrow, bands = int(ncol), int(nrow), int(bands)

# start K at number of labels
c, class_label = {}, {}
for i in range(1, len(lines)):
    line = lines[i]
    label = line[i_lab]
    x, y = int(line[i_row]), int(line[i_lin])
    ix = (y * ncol) + x
    if ix < nrow * ncol:  # skip if out of bounds
        class_label[ix] = label
        c[label] = (c[label] + 1) if label in c else 1
K = len(c) # starting number of classes
K -= 1 # for testing, delete this line later

go = True
while go:
    whoami = os.popen("whoami").read().strip()
    exe = path + "../cpp/kmeans_multi.exe"
    print("exe", exe)
    exe = os.path.normpath(exe)
    print("exe", exe)
    run(exe + " " + infile + " " + str(K))

    class_file = infile + "_kmeans.bin"
    ncol, nrow, bands, data = read_binary(class_file)

    kmeans_label = {}
    for i in range(1, len(lines)):
        line = lines[i]
        x = int(line[i_row])
        y = int(line[i_lin])
        ix = (y * ncol) + x
        # print("row", line[i_row], line[i_lin], line[i_xof], line[i_yof], line[i_lab], "class", data[ix])
        if ix < nrow * ncol:
            kmeans_label[ix] = data[ix]

    kmeans_label_by_class = {}
    for p in class_label:
        L = class_label[p]
        kmeans_label_by_class[L] = [] if (L not in kmeans_label_by_class) else (kmeans_label_by_class[L])
        kmeans_label_by_class[L].append(kmeans_label[p])

    for c in kmeans_label_by_class: # what would a vectorization for an op like this look like?
        kmeans_label_by_class[c] = set(kmeans_label_by_class[c])
    print(kmeans_label_by_class)

    # check if we're done
    bad, empty = False, set()
    for k in kmeans_label_by_class:
        kk = kmeans_label_by_class[k]
        for j in kmeans_label_by_class:
            if k == j:
                continue
            kj = kmeans_label_by_class[j]
            if kk.intersection(kj) != empty:
                bad = True
  
    if not bad:
        print("good")

        # clean up labels so that everything outside the known classes is 0, and all clusters for class get same label..
        used_labels = set()
        for k in kmeans_label_by_class:
            for j in kmeans_label_by_class[k]:
                used_labels.add(j)
        print("used_labels", used_labels)        

        lookup = {}
        for k in range(0, K):
            k = float(k)
            if k not in used_labels:
                lookup[k] =  0.

        ci = 1
        for k in kmeans_label_by_class:
            for j in kmeans_label_by_class[k]:
                lookup[j] = ci
            ci += 1

        print("lookup", lookup)
        # apply lookup

        for i in range(0, nrow* ncol):
            data[i] = lookup[data[i]]

        write_binary(data, class_file) # relabel the data and output
        break
          
    # kmeans_label_by_class: {'fireweedandaspen': [0.0], 'blowdownwithlichen': [1.0, 0.0], 'pineburned': [1.0, 1.0, 1.0]}
    K += 1 # try adding a class!
print("kmeans_label_by_class", kmeans_label_by_class)
print("lookup", lookup)

# translate the lookup
for label in kmeans_label_by_class:
    labels = list(kmeans_label_by_class[label])
    labels = [lookup[i] for i in labels]
    kmeans_label_by_class[label] = set(labels)

print(kmeans_label_by_class)

import matplotlib.pyplot as plt
hdr = hdr_fn(infile)
npx = nrow * ncol
#samples, lines, bands = read_hdr(hdr)
#samples, lines, bands = int(samples), int(lines), int(bands)
# npx = lines * samples # number of pixels
# data = read_float(sys.argv[1]).reshape((bands, npx))
data = data.reshape((nrow, ncol))

fig, ax = plt.subplots()
img = ax.imshow(data, cmap='Spectral')
# ax.set_aspect("auto")
cbar = plt.colorbar(img)#  .legend([0, 1, 2, 3], ['0', '1', '2', '3'])\
n_labels = 5
cbar.set_ticks(np.arange(n_labels))
tick_labels = ["noise"]
ci = 1
for label in kmeans_label_by_class:
    tick_labels.append(label)
    x = kmeans_label_by_class[label]
    if set([ci]) != x:
        err("color index problem")
    ci += 1
cbar.ax.set_yticklabels(tick_labels) #"bad", "good", "other", "more", "what"])
plt.show()
# run("python3 " + path + "read_multi.py " + infile + "_kmeans.bin")
