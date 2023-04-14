'''20220414 add swap file and turn it on'''
#!/usr/bin/env python3
import sys
import os
from misc import err, args, run

if len(args) < 2:
    err("swapadd.py [number of GB swap file to add in this location]")

size = int(sys.argv[1])

if os.path.exists('swapfile'):
    err("swapfile exists in this location")

run('sudo fallocate -l ' + str(size) + 'G ./swapfile')
run('sudo chmod 600 ./swapfile')
run('sudo mkswap ./swapfile')
run('sudo swapon ./swapfile')
run('free -m')

'''
free -m shows memory use
'''
