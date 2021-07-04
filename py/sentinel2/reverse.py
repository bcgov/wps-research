'''output a file line by line, in reverse order.. i.e. lines order reversed'''
import os
import sys

lines = open(sys.argv[1]).read().strip().split("\n")
lines = [x.strip() for x in lines]

for i in range(len(lines)):
    print(lines[len(lines) -i - 1])
