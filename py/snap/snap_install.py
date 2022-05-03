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
from misc import run, exists, pd, sep, err

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

if not exists('/usr/local/bin/esa-snap'):
    def re_name():
        run('sudo mv /usr/local/bin/snap /usr/local/bin/esa-snap')

    if exists('/usr/local/bin/snap'):
        re_name()
    else:
        print('installing ESA snap..' +
              'Please select all defaults incl. python support')
        run('sudo ' + target)
        re_name()

try:
    import snappy
except Exception:
    print('configuring snappy..')
    def pyc():
        x = os.popen('which python3 2>&1').read().strip()
        if x == '' or x.split(':')[-1].strip() == 'command not found':
            return None
        else:
            return x

    if pyc() == None:
        print('python3 not found, attempting install..')
        run('sudo apt install python3')

    py_cmd = pyc()
    if py_cmd == None:
        err('python3 not found, pls. check python path')

    sc = '/opt/snap/bin/snappy-conf'
    if not exists(sc):
        err(sc + ' not found, please check snap install path')
    
    result = os.popen(' '.join([sc, py_cmd, '2>&1'])).read().strip()
    if len(result.split('Please check the log file')) > 1:
        logfile = None
        lines = [x.strip() for x in result.split('\n')]
        for line in lines:
            w = line.split('Please check the log file')
            if len(w) > 1:
                logfile = w[-1].strip().strip('.').strip("'")
            
        print(logfile)
        lines = [x.strip() for x in open(logfile).read().strip().split('\n')]
        
        for line in lines:
            w = line.split("ERROR: The module 'jpy' is required to run snappy")
            if len(w) > 1:
                print('attempting to install jpy..')
                print(line)
    


