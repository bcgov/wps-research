'''add folders expected by sen2cor, if they are missing..why?
..Google cloud platform doesn't save empty folders'''
import os
from ../misc import args, sep, exists
f = args[1]

def md(f):
    if not exists(f):
        print('mkdir', f)
        os.mkdir(f)

md(f + sep + 'AUX_DATA')
md(f + sep + 'HTML')
