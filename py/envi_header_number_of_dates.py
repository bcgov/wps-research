import os
import sys

args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0: err("failed to run: " + str(c))

band_names = os.popen(' '.join(['python3 ',
                                pd + 'envi_header_band_names.py',
                                args[1]])).readlines()


dates = set([x.split(':')[0].strip().split()[0].strip() for x in band_names])
for d in dates:
    print(d)
