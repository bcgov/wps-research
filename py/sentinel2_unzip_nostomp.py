'''
20210718 unzip all sentinel-2 zip files in a folder, unzipping the contents into .SAFE folders..
.. this is for the google-format data solution..
'''
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

