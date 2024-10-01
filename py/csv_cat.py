'''20230222 concatenate csv files, does not apply full validation/QA, matching header lines at beginning of each input file assumed
20241001 Output to file csv_cat.csv
simple CSV format assumed'''
from misc import err, args, exists
import sys
import os

if exists('csv_cat.csv'):
    err('output file: csv_cat.csv already exists')

files = [x.strip() for x in os.popen('ls -1 *.csv').readlines()]  # all csv files in present folder

if len(args) == 1:
    print("default: cat all csv present in existing folder")
else:
    if len(args) < 3:  # need to specify at least two files to concatenate, to proceed 
        err("csv_cat [input csv file 1] .. [input csv file n]")
    files = args[1:]  # use files specified on the command-line

dat, f0 = {}, None
for f in files:
    lines = [x.strip() for x in open(f).readlines()]
    print("+r", f, 'EMPTY FILE' if len(lines) == 0 else '')

    if len(lines) == 0:
        continue

    dat[f] = lines  # record the lines from this file
    
    if f0 is None:
        f0 = lines[0]
    else:
        if lines[0] != f0:
            err("headers not exactly equal")

f0_split = f0.split(',')
out_file = open('csv_cat.csv', 'w')
out_file.write(f0) # write the header line out just once

for f in files:
    print(f)
    if f not in dat:
        continue

    lines = dat[f][1:]  # Assume skipping the first line to avoid repeating header

    for line in lines:
        split_line = line.split(',')

        if len(split_line) != len(f0_split):
            err('nonsimple CSV format')

        out_file.write("\n" + ','.join([x.strip() for x in split_line]))  # add newline at the front so there isn't any extra newline at the end of the file
out_file.close()
