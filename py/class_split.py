'''class_split.py: split float valued map, into binary class maps..

..this was developed to translate the VRI into a one-hot encoded format.

it will use an associated *.lut file, if available to code the file names
for the binary class maps produced '''
import os
import sys
from misc import *

# don't split files with more than this many labels
max_labels = 20

# count files
n_bin_files, n_lut_files, bin_files_skipped = 0, 0, []
n_files_written = 0

args, d = sys.argv, None

if len(args) < 2:
    err('class_split.py [input directory]')
else:
    d = args[1].strip(os.path.sep)  # input directory

b = d + os.path.sep + "binary"
print("init binary class folder")
if not exist(b):
    os.mkdir(b)

# list .bin files in the directory:
bin_files = []
listing = os.listdir(d)
for f in listing:
    if f.split('.')[-1] == 'bin':
        bin_files.append(f)

# for every .bin file:
for f in bin_files:
    n_bin_files += 1
    w = f.split(".")
    if len(w) != 2:
        err("unexpected filename format")
    f_short = f
    f = d + os.path.sep + f_short
    if not exist(f):
        err("count not find file: " + f)

    # read bin file
    samples, lines, bands, data = read_binary(f)

    # count number of observations for each data value
    count = hist(data)
    n_labels = len(count.keys())
    if n_labels > max_labels:
        bin_files_skipped.append(f)
        continue
    print("\tnumber of class labels,", n_labels)

    # default filenames for value: the value
    value_to_name = {value: str(value) for value in count.keys()}

    s = ",".join([f, str(samples), str(lines), str(bands)])
    lut_file = d + os.path.sep + w[0] + ".lut"
    if exist(lut_file):
        lines, lookup = open(lut_file).readlines(), {}
        for line in lines:
            w = line.strip().split(",")
            lookup[float(w[1])] = w[0]

        # check that each observed value in bin file, also in lut file
        for value in count:
            if value not in lookup:
                err("failed to look up: " + str(value))

        # use lookup to change output filename for class
        print("lookup", str(lookup))

        for value in count:
            if value not in count:
                err("lookup failed")
            # use lut value if available
            value_to_name[value] = lookup[value].strip().replace(" ", "_")

        n_lut_files += 1
        print(s, lut_file)
    else:
        print(s)

    # write a file for each observed value
    for c in count.keys():
        of = None
        try:
            of = b + os.path.sep + f_short[0:-4] + '_eq_' + value_to_name[c]
        except Exception:
            print("value_to_name", str(value_to_name))
            err("lookup not found:" + str(c) +
                " count.keys():" + str(count.keys()))
    
        ofn, hfn = of + ".bin", of + ".hdr"
        output = copy.deepcopy(data)
        for i in range(0, len(data)):
            output[i] = (1. if data[i] == c else 0.)
        write_binary(output, ofn)
        write_hdr(hfn, samples, lines, bands)
        n_files_written += 1

# if there is a lut, check it matches

print("bin files skipped", bin_files_skipped)
print("number of bin files skipped", len(bin_files_skipped))
print("number of bin files", n_bin_files)
print("number of lut files", n_lut_files)
print("number of binary class files written", n_files_written)
