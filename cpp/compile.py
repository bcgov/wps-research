#!/usr/bin/env python
'''compile all .cpp files in folder to .exe files. N.B. remove -w flag from g++ to examine warnings
20220813 add work queue
20220813 add python program wrapping
    Compile all python files in python folder using a C wrapper (access from common folder)
'''
import os
import sys
import multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'py'))
abspath = os.path.abspath
sep = os.path.sep
def g_pd():
    return os.path.abspath(sep.join(abspath(__file__).split(sep)[:-1])) + sep  # python directory i.e. path to here
pd = g_pd()

def g_cd():
    return os.path.abspath(sep.join(abspath(__file__).split(sep)[:-2]) + sep + 'cpp') + sep
cd = g_cd()

def run(c):
    return os.system(c)

def err(m):
    print("Error", m); sys.exit(1)

def me():
    return os.popen("whoami").read().strip()
exists = os.path.exists


ncpu = multiprocessing.cpu_count()
print(ncpu)


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
files = os.popen('ls -1 ' + pd + '*.py').readlines()
for f in files:
    f = os.path.abspath(f)
    f = f.strip()
    fn = f.split(os.path.sep)[-1]

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
