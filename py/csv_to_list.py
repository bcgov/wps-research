'''20221216: convert one cols from CSV, to python list'''
import os
import sys
import csv
args = sys.argv

if len(args) < 3:
    print("csv_to_list.py [csv file] [col name 1]")
    sys.exit(1)
reader = csv.reader(open(args[1]), delimiter=',', quotechar='"')

ri = 0
arrow = []
fields = []
domain_i = -1
for row in reader:
    r = [x.strip() for x in row]
    if ri == 0:
        fields = r
        lookup = {fields[i]: i for i in range(len(fields))}
        print(lookup)
        domain_i = lookup[args[2]]
    else:
        arrow += [r[domain_i]]
    ri += 1
print(arrow)
