'''extract PRISMA data in separate folder per frame'''
import os
import sys
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep # source dir

def err(m):
    print("Error", m); sys.exit(1)

def run(c):
    print(c)
    if os.system(' '.join(c) if type(c) == list else c) != 0:
        err("command failed")

def ff(f): return f + sep + f
def h5(f): return ff(f) + '.he5'
def sc(f): return ff(f) + '_SWIR_Cube.bin'
def vc(f): return ff(f) + '_VNIR_Cube.bin'
def cc(f): return ff(f) + '.bin'  # should unif func w decl

files = [x.strip()[:-4] for x in
         os.popen('ls -1 *.zip').readlines()]

for f in files:
    if not exists(f):
        run(['mkdir -p', f])

for f in files:
    if not exists(h5(f)):
        run(['unzip -e', 
             f + '.zip',
             '-d',
             f])

for f in files:
    if not exists(sc(f)):
        run(['python3',
            pd + 'prisma_read.py',
            h5(f)])

for f in files:
    if not exists(cc(f)):
        run(['python3',
            pd + 'raster_stack.py',
            sc(f),
            vc(f),
            cc(f)])
