import os
import sys
args, sep = sys.argv, os.path.sep

lines = os.popen('find ./ -name "PRS*.zip"').readlines()
lines = [x.strip() for x in lines]


c = {}
example = {}
for line in lines:
    f = os.path.abspath(line)
    if os.path.exists(f):
        w = f.split(sep)
        fn = w[-1].strip()
        
        if fn not in c:
            c[fn] = 1
        else:
            c[fn] += 1
        example[fn] = f

for x in c:
    print(c[x], x)  

L = list(c.keys())
print(len(L), "unique files")


if len(args) > 1:
    dst = os.path.abspath(args[1]) + sep
    if not os.path.exists(dst) or not os.path.isdir(dst):
        print("Error: dst folder undefined")
        sys.exit(1)
    
    for x in L:
        if not os.path.exists(dst + x):
            cmd = 'cp -v ' + example[x] + ' ' + dst
            print(cmd)

        else:
            print("found: ", x)
