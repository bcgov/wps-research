#!/usr/bin/env python3
import os
import sys

lines = os.popen("ls -1 *.bin").readlines()
lines = [x.strip() for x in lines]

for line in lines:
    a = os.system("imv " + line)


