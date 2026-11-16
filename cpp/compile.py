#!/usr/bin/env python
'''compile all .cpp files in folder to .exe files. N.B. remove -w flag from g++ to examine warnings
20220813 add work queue
20220813 add python program wrapping
    Compile all python files in python folder using a C wrapper (access from common folder)
20261116 add python program wrapping for rcm folder
    Also wrap python files in rcm folder (top level only, not subfolders) the same way, into the same folder
    Skip any rcm script with the same name as a script wrapped from py/ (report skipped scripts at the end)
'''
import os
import sys
import multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py'))
abspath = os.path.abspath
sep = os.path.sep
def g_pd():
    return os.path.abspath(sep.join(abspath(__file__).split(sep)[:-2])) + sep + 'py' + sep # python directory i.e. path to here
pd = g_pd()
print("python dir", pd)
def g_cd():
    return os.path.abspath(sep.join((abspath(__file__).split(sep))[:-1]) + sep) + sep
cd = g_cd()
print("c dir", cd)

def run(c):
    return os.system(c)

def err(m):
    print("Error", m); sys.exit(1)

def me():
    return os.popen("whoami").read().strip()
exists = os.path.exists


ncpu = multiprocessing.cpu_count()
print(ncpu)

run('rm ' + cd + 'wrap-py*')

files = os.popen('ls -1 ' + cd + '*.cpp').readlines()
files = [f.strip() for f in files]

i = 0
of = open('compile.sh', 'wb')
of.write('#!/usr/bin/env bash'.encode())
for f in files:
    print(i % ncpu, f, i)
    if f == 'misc.cpp':
        run('g++ -c misc.cpp -pthread')
        continue
    s = ''
    fn = f[:-4]
    symlink =  '' 
    print(fn, os.path.islink(fn))
    if fn != "misc":
        run('rm -v ' + fn)
        symlink = 'ln -s ' + fn + '.exe ' + fn if not os.path.islink(fn) else ''# symbolic link to path without .exe!
        if f[:4] == 'cuda':
            s = '\ntest ! -f ' + fn + '.exe && nvcc ' + fn + '.cpp misc.cpp -o ' + fn + '.exe' #+ symlink
            print(s)
        else:
            s = '\ntest ! -f ' + fn + '.exe && g++ -w -O4 ' + fn + '.cpp  misc.cpp -o ' + fn + '.exe -lpthread' #  + symlink

        if (i + 1) % ncpu != 0:
            s += ' &'
        of.write((s + "\n").encode())

        if s != '' and symlink != '':
            s = symlink
            if (i + 1) % ncpu != 0:
                s += ' &'
            of.write((s + "\n").encode())
   
        if (i + 1) % ncpu == 0:
            of.write("wait\n".encode())

        i += 1
of.write("\nwait\n".encode())

# wrap the python files
files = [x.strip() for x in os.popen('ls -1 ' + pd + '*.py').readlines()]
files += [x.strip() for x in os.popen('ls -1 ' + pd + 'gcp' + sep + '*.py').readlines()]

# also wrap the python files in the rcm folder (top level only, not subfolders), using the same method.
# skip any rcm script whose name matches a script wrapped above from py/, so its wrapper is not overwritten
rd = os.path.abspath(sep.join(abspath(__file__).split(sep)[:-2])) + sep + 'rcm' + sep # rcm directory
print("rcm dir", rd)
py_names = set([x.split(sep)[-1] for x in files])
rcm_skipped = []
for x in [x.strip() for x in os.popen('ls -1 ' + rd + '*.py').readlines()]:
    if x.split(sep)[-1] in py_names:
        rcm_skipped.append(x)
    else:
        files.append(x)
print(files)
for f in files:
    f = os.path.abspath(f)
    f = f.strip()
    fn = f.split(os.path.sep)[-1]

    if fn == 'compile':
        continue

    wrap_file = 'wrap-py_' + fn + '.cpp'
    print("+w", wrap_file)
    cf = open(wrap_file, 'wb')
    lines = ['#include<stdlib.h>',
             '#include<iostream>',
             '#include<string>',
             'using namespace std;',
             'int main(int argc, char ** argv){',
             '  string cmd("python3 ");',
             '  cmd += string("' + f + '");',
             # '  cmd += string(";")',
             '  for(int i = 1; i < argc; i++){',
             '    cmd += string(" ") + string(argv[i]);',
             '  }',
             '  std::cout << cmd << endl;',
             '  system(cmd.c_str());',
             '  return(0);',
             '}']
    cf.write('\n'.join(lines).encode())
    cf.close()
    cmd = 'g++ -w -O3 -o ' + fn + ' ' + wrap_file #wrap_py.cpp '
    print('\t' + cmd)
    # cmds += " " + cmd # a = os.system(cmd)

    if (i + 1) % ncpu != 0:
        cmd += ' &'

    of.write((cmd + '\n').encode())

    if (i + 1) % ncpu == 0:
        of.write("wait\n".encode())

    i += 1
of.write("\nwait\n".encode())
of.close()

run("chmod 755 compile.sh")
run("./compile.sh")
run('rm ' + cd + 'wrap-py*')

print('''
# if not already installed, don't forget to install a current GDAL with 
sudo apt install libopenjp2-7-dev libopenjp2-7 # or other newest jpeg2000 library
cmake ..
cmake --build .
cmake --build . --target install
''')

# report rcm scripts skipped because a py/ script has the same name
if len(rcm_skipped) > 0:
    print("rcm scripts skipped (same name as a script in py/, whose wrapper was kept):")
    for x in rcm_skipped:
        print("  " + x)
else:
    print("rcm scripts skipped: none")
