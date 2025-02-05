'''20220503 install snap version 8...

link to esa-snap, not snap (solve ubuntu issue)

*** includes snappy install and config!
tested on Ubuntu 20
'''

use_mirror = False
main = 'https://download.esa.int/step/snap/8.0/installers/esa-snap_all_unix_8_0.sh'
mirror = 'https://step.esa.int/downloads/8.0/installers/esa-snap_all_unix_8_0.sh'
jpy = 'https://github.com/bcdev/jpy/archive/refs/heads/master.zip'

import os
import sys
import stat
import shutil
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + "..")  # relative import
from misc import run, exists, pd, sep, err
sys.path.append('/home/' + os.popen('whoami').read().strip() +
                '/.snap/snap-python')
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

# the version of jdk SNAP expects
JDK = '/usr/lib/jvm/java-8-openjdk-amd64/bin'
if not exists(JDK + '/javac'):
    run('sudo apt install openjdk-8-jdk')
try:
    import setuptools
except Exception:
    run('sudo apt install python3-setuptools')

def re_name():
    run('sudo mv /usr/local/bin/snap /usr/local/bin/esa-snap')

if not exists('/usr/local/bin/esa-snap'):
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
    if not exists(os.popen('which mvn').read().strip()):
        run('sudo apt install python3-pip maven')
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
    print([sc, py_cmd, '2>&1'])
    result = os.popen(' '.join([sc, py_cmd, '2>&1'])).read().strip()
    if len(result.split('Please check the log file')) > 1:
        logfile = None
        lines = [x.strip() for x in result.split('\n')]
        for line in lines:
            w = line.split('Please check the log file')
            if len(w) > 1:
                logfile = w[-1].strip().strip('.').strip("'")

        print('+r', logfile)
        lines = [x.strip() for x in open(logfile).read().strip().split('\n')]

        jpy_err = "ERROR: The module 'jpy' is required to run snappy"
        for line in lines:
            w = line.split(jpy_err)
            if len(w) > 1:
                print('attempting to install jpy..')
                print(line)

                # download jpy source
                jpyz = pd + 'snap' + sep + 'jpy-master.zip'
                if not exists(jpyz):
                    run(['wget', jpy, '-O', jpyz])

                # unzip jpy source
                jpyd = pd + 'snap' + sep + 'jpy-master'
                if not exists(jpyd):
                    run(['unzip', jpyd])

                jpy_wheel = jpyd + sep + 'dist'
                jpyc = jpyd + sep + 'setup.py'
                jpy_cmd = 'cd ' + jpyd + \
                          '; python3 setup.py build maven bdist_wheel'

                print(jpy_cmd)
                results = [x.strip() for x in
                           os.popen(jpy_cmd).read().split('\n')]
                print(results)
                run('cp -v ' + jpy_wheel + sep +
                    '*.whl ~/.snap/snap-python/snappy/')
    
                homeset = 'environment variable "JAVA_HOME" must be set'
                for x in results:
                    if len(x.split(homeset)) > 1:
                        print('need to set JAVA_HOME')
                        java_home = JDK
                        print(java_home)
                        path_jpy = jpy_d + '/src/main/java/org'
                        bashrc_fn = '/home/' + \
                                    os.popen('whoami').read().strip() + \
                                    sep + '.bashrc'
                        bashrc = open(bashrc_fn).read().split('\n')
                        bashrc += ['',
                                   '# set JAVA_HOME for esa SNAP install',
                                   'export JAVA_HOME=' + java_home,
                                   'export PATH=$PATH:' + path_jpy,
                                   '']
                        print('cp', bashrc_fn, bashrc_fn + '.bak')
                        shutil.copyfile(bashrc_fn, bashrc_fn + '.bak')
                        print('+w', bashrc_fn)
                        out_data = ('\n'.join(bashrc)).encode()
                        open(bashrc_fn, 'wb').write(out_data)

                        jpy_cmd = 'export JAVA_HOME=' + java_home + '; ' + \
                                  jpy_cmd
                        print(jpy_cmd)
                        run('cd ' + jpyd + 
                            '; python3 setup.py build maven bdist_wheel')
                        
                        run('cp -v ' + jpy_wheel + sep + 
                            '*.whl ~/.snap/snap-python/snappy/')
else:
    print('=' * 40)
    print("SUCCESS")
    print("(*) make sure to reference this folder in your python code:")
    print("  sys.path.append('/home/' +")
    print("                  os.popen('whoami').read().strip() +")
    print("                  '/.snap/snap-python')")
    print('=' * 40)
