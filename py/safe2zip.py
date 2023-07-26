from misc import err, exists
import sys
import os

lines = [x.strip() for x in os.popen('ls -1').readlines()]

for line in lines:
    if line[-5:] == '.SAFE':
        print(line)




