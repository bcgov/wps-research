import os
import sys
sep = os.path.sep

lines = os.popen('find ./ -name "PRS*.zip"').readlines()
lines = [x.strip() for x in lines]


c = {}
for line in lines:
    f = os.path.abspath(line)
    if os.path.exists(f):
        w = f.split(sep)
        fn = w[-1].strip()
        
        if fn not in c:
            c[fn] = 1
        else:
            c[fn] += 1

for x in c:
    print(c[x], x)  

L = list(c.keys())
print(len(L), "unique files")

