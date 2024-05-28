'''20240227 deduplicate a csv file. Warning: does not consider whitespace
'''
import sys
import os

def err(m):
    print('Error:', m); sys.exit(1)
args = sys.argv

if len(args) < 2:
    err('python3 unique.py [csv file name]')

lines = open(args[1]).readlines() # read lines from file

records = set()  # initialize empty set

for line in lines:
    records.add(line)  # add each record to set 

open(args[1] + "_unique.csv", "wb").write(''.join(list(records)).encode())  # write unique records
