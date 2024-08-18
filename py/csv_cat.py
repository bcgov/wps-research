'''20230222 concatenate csv files, no validation, matching header lines assumed
Output on stdout
'''
from misc import err
import sys

args = sys.argv

if len(args) < 3:
    err("csv_cat [input csv file 1] .. [input csv file n]")

files = args[1:]

dat = {}
f0 = None
for f in files:
    lines = [x.strip() for x in open(f).readlines()]
    dat[f] = lines
    
    if f0 is None:
        f0 = lines[0]
    else:
        if lines[0] != f0:
            err("headers not exactly equal")
print(f0)
for f in files:
    lines = dat[f][1:]
    for line in lines:
        print(line)
    
