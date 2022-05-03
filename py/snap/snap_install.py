'''20220503 install snap version 8...

link to esa-snap, not snap (solve ubuntu issue)

*** include snappy install!'''
use_mirror = False
main = 'https://download.esa.int/step/snap/8.0/installers/esa-snap_all_unix_8_0.sh'
mirror = 'https://step.esa.int/downloads/8.0/installers/esa-snap_all_unix_8_0.sh'

import os
import sys
import stat
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + "..")  # relative import
from misc import run, exists, pd, sep

path = mirror if use_mirror else main
fn = path.split('/')[-1]
target = pd + 'snap' + sep + fn

# check for wget utility. Could use urllib2 instead?
x = os.popen('wget 2>&1').read().strip().split('\n')[0].strip()
if x != 'wget: missing URL':
    print('need superuser to install wget..')
    run('sudo apt install wget')

if not exists(target):
    run(['wget', path, '-O', target])
else:
    print('+r', target)

# check for execute permission
x = str(oct(stat.S_IMODE(os.lstat(target).st_mode)))[-3:]
if x != '744':
    run(['sudo', 'chmod', '744', target])
