# functions for reuse, for image processing, etc.
import os
import sys
import copy
import math
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

# print message and exit
def err(c):
    print('Error: ' + c)
    sys.exit(1)

def read_hdr(hdr):
    samples, lines, bands = 0, 0, 0
    for line in open(hdr).readlines():
        words = line.split('=')
        if len(words) == 2:
            for f in ['samples', 'lines', 'bands']:
                if words[0].strip() == f:
                    exec(f + ' = int(' + words[1].strip() + ')')
    return samples, lines, bands

# require a filename, or list of filenames, to exist
def assert_exists(fn):
    try:
        if type(fn) != str:
            iterator = iter(fn)
            for f in fn:
                assert_exists(f)
            return
    except:
        # not iterable
        pass

    if not os.path.exists(fn):
        err("couldn't find required file: " + fn)
