'''20250104 repeat a command perpetually at a specified interval. Adapted from ../../bin/src/
'''
import os
import sys
import time
from misc import args

default = 10
delay, arg = default, args[1:]

if len(args) > 1:
    try:
        delay = int(args[1].strip('-'))
        arg = args[2:]
    except:
        pass

cmd = ' '.join(arg)
print(cmd)

while(True):
    a = os.system(cmd)
    time.sleep(delay)
