import os
import sys

def run(c):
    print(c)
    a = os.system(c)
    if a != 0:
        print("Error: exit code not 0")

files = [f.strip() for f in os.popen("ls -1 *.zip").readlines()]
for f in files:
    if os.path.isfile(f):
        d = f[:-4] + '.SAFE'
        run("unzip -d " + d + " " + f)

