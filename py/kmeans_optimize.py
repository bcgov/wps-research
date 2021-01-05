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

sep = os.path.sep
path = sep.join(__file__.split(sep)[:-1]) + sep  # path to this file

K = 5
whoami = os.popen("whoami").read().strip()
cmd = path + "../cpp/kmeans_multi.exe stack.bin " + str(K)

run(cmd)

class_file = infile + "_kmeans.bin"

samples, lines, bands, data = read_binary(class_file)

#cmd = "python3 " + path + "read_multi.py " + infile + "_kmeans.bin" 
#run(cmd)
