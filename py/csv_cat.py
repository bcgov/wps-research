'''20230222 concatenate csv files, no validation, matching header lines assumed
20241001 Output to file csv_cat.csv
simple CSV format assumed'''
from misc import err
import sys
import os

args, files = sys.argv, [x.strip() for x in os.popen('ls -1 *.csv').readlines()]

if len(args) == 1:
    print("default: cat all csv")
else:
    if len(args) < 3:
        err("csv_cat [input csv file 1] .. [input csv file n]")
    files = args[1:]

dat = {}
f0 = None
for f in files:
    lines = [x.strip() for x in open(f).readlines()]
    print("+r", f, 'EMPTY FILE' if len(lines) == 0 else '')

    if len(lines) == 0:
        continue

    dat[f] = lines
    
    if f0 is None:
        f0 = lines[0]
    else:
        if lines[0] != f0:
            err("headers not exactly equal")
print(f0)

out_file = open('csv_cat.csv', 'w')
out_file.write(f0) # write the header line once

for f in files:
    print(f)
    if f not in dat:
        continue

    lines = dat[f][1:]  # Assuming you're skipping the first line

    for line in lines:
        split_line = line.split(',')
        if len(split_line) != len(f0):
            err('nonsimple CSV format')
        out_file.write("\n" + ','.join([x.strip() for x in split_line])) 

out_file.close()
