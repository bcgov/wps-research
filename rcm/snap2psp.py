'''convert .img files (with byte-order =1) to .bin files (with byte order 0) by swapping byte order..
..this is for converting .img files produced by SNAP, to PolSARPro format files

20220515: update this to read/write headers by copy/modify'''
import os
import sys
args = sys.argv
sep = os.path.sep

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0:
        err("command failed:\n\t" + c)

if len(args) < 2:
    print("snap2psp.py [input folder name] # convert snap byte-order= 1 .img data to byte-order 0 .bin data")

p = os.path.abspath(args[1]) + sep
cmd = "ls -1 " + p + "*.img"
files = [x.strip() for x in os.popen(cmd).readlines()]
for f in files:
    if os.path.isfile(f):
        of = f[:-3] + "bin"
        run("sbo " + f + " " + of + " 4")
