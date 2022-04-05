'''compare two file listings!
find members present in one list but not the other'''
import os
import sys
from ../misc import args, err

if len(args) != 3:
    err("compare.py [file listing 1] [file listing 2]")

lines1 = [x.strip().split() for x in open(args[1]).readlines()]
lines2 = [x.strip().split() for x in open(args[2]).readlines()]

for i in range(len(lines1)):
    lines1[i] = lines1[i][-1]
for i in range(len(lines2)):
    lines2[i] = lines2[i][-1]

set1 = set(lines1)
set2 = set(lines2)

print("in set1 but not set2:")
for x in set1:
    if x not in set2:
        print(x)

print("int set2 but not set1:")
for x in set2:
    if x not in set1:
        print(x)
