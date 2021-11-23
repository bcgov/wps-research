#!/usr/bin/env python
import os
import sys

def err(m):
    print "Error: " + str(m)
    sys.exit(1)

def run(c):
    print "run(\"" + c + "\")"
    a = os.system(c)
    if a != 0:
        err("command failed")
    return a

files = os.popen("ls -1 ../*.bin").readlines()

for f in files:
    f = f.strip()
    f2 = f.split("/")[-1] + '_sub.bin'
    run('gdal_translate -of ENVI -ot Float32 -srcwin 684 200 410 401 ' + f + ' ' + f2)
