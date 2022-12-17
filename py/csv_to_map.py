'''20221216: convert two cols from CSV, to python map'''
import os
import sys
import csv
args = sys.argv

if len(args) < 4:
    print("csv_to_map.py [csv file] [col name 1] [col name 2]")
    sys.exit(1)
reader = csv.reader(open(args[1]), delimiter=',', quotechar='"')

ri = 0
arrow = {}
fields = []
domain_i, range_i = -1, -1
for row in reader:
    r = [x.strip() for x in row]
    if ri == 0:
        fields = r
        lookup = {fields[i]: i for i in range(len(fields))}
        print(lookup)
        domain_i = lookup[args[2]]
        range_i = lookup[args[3]]
    else:
        if r[domain_i] != r[range_i]:
            arrow[r[domain_i]] = r[range_i]
    ri += 1
print(arrow)
