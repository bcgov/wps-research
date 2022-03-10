'''extract PRISMA data in separate folder per frame'''
import os
import sys
sep = os.path.sep
exists = os.path.exists

def err(m):
    print("Error", m); sys.exit(1)

def run(c):
    print(c)
    if os.system(' '.join(c) if type(c) == list else c) != 0:
        err("command failed")

files = [x.strip()[:-4] for x in
         os.popen('ls -1 *.zip').readlines()]

for f in files:
    if not exists(f):
        run(['mkdir -p', f])

for f in files:
    he5 = f + sep + f + '.he5'
    if not exists(he5):
        run(['unzip -e', 
             f + '.zip',
             '-d',
             f])
