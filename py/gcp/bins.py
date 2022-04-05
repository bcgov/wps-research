'''open .bin files listed in bins.txt, with imv'''
import os
import sys
from ../misc import run
lines = open("bins.txt").readlines()

for line in lines:
    run("imv " + line.strip())
