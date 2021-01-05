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

sep = os.path.sep
path = sep.join(__file__.split(sep)[:-1]) + sep 
print(path)

K = 5
whoami = os.popen("whoami").read().strip()
cmd = path + "kmeans_multi.exe stack.bin " + str(K)

print(cmd)
# run(cmd)

cmd = "python3 " + path + "read_multi.py stack.bin_means.bin"

print(cmd)
