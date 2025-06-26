'''20240723 stack sentinel2 .bin files in temporal order

one tile only supported'''
from misc import run
import os

lines = [x.strip() for x in os.popen('ls -1 S*.bin').readlines()]

# process in order
lines = [[line.split('_')[2], line] for line in lines]
lines.sort()
lines = [line[1] for line in lines]

for line in lines:
    print(line)

print(" ".join(lines))
