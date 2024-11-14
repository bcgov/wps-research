'''20241114 if our directory structure got too deep ( extracted into folder with directory name two levels deep ) 
make it shallow again! 
'''
import os
import sys
sep = os.path.sep

lines = os.popen('ls -1').readlines()
lines = [x.strip() for x in lines]

for line in lines:
    if os.path.isdir(line):
        if os.path.isdir(line + sep + line):
            print(line + sep + line)
            cmd = 'mv -v ' + line + sep + line + sep + '* ' + line + sep
            print(cmd)
            a = os.system(cmd)
