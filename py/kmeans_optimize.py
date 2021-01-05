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
i_row = hdr.index('row')
i_lin = hdr.index('lin')
i_xof = hdr.index('xoff')
i_yof = hdr.index('yoff')
i_lab = hdr.index('feature_id')
sep = os.path.sep
path = sep.join(__file__.split(sep)[:-1]) + sep  # path to this file

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
    class_label[ix] = label
    c[label] = (c[label] + 1) if label in c else 1
K = len(c) # starting number of classes
K -= 1 # for testing, delete this line later

go = True
while go:
    whoami = os.popen("whoami").read().strip()
    run(path + "../cpp/kmeans_multi.exe stack.bin " + str(K))

    class_file = infile + "_kmeans.bin"
    ncol, nrow, bands, data = read_binary(class_file)

    kmeans_label = {}
    for i in range(1, len(lines)):
        line = lines[i]
        x = int(line[i_row])
        y = int(line[i_lin])
        ix = (y * ncol) + x
        # print("row", line[i_row], line[i_lin], line[i_xof], line[i_yof], line[i_lab], "class", data[ix])
        kmeans_label[ix] = data[ix]
     
    print(class_label)
    print(kmeans_label)

    kmeans_label_by_class = {}
    for p in class_label:
        L = class_label[p]
        kmeans_label_by_class[L] = [] if (L not in kmeans_label_by_class) else (kmeans_label_by_class[L])
        kmeans_label_by_class[L].append(kmeans_label[p])
    print(kmeans_label_by_class)

    run("python3 " + path + "read_multi.py " + infile + "_kmeans.bin")
    # check if we're done


    # kmeans_label_by_class: {'fireweedandaspen': [0.0], 'blowdownwithlichen': [1.0, 0.0], 'pineburned': [1.0, 1.0, 1.0]}
    K += 1
