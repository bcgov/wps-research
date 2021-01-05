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

K = 5
whoami = os.popen("whoami").read().strip()
run(path + "../cpp/kmeans_multi.exe stack.bin " + str(K))

class_file = infile + "_kmeans.bin"
ncol, nrow, bands, data = read_binary(class_file)

for i in range(1, len(lines)):
    line = lines[i]
    x = int(line[i_row])
    y = int(line[i_lin])
    print("row", line[i_row], line[i_lin], line[i_xof], line[i_yof], line[i_lab], "class", data[(y * ncol) + x])

#cmd = "python3 " + path + "read_multi.py " + infile + "_kmeans.bin" 
#run(cmd)
