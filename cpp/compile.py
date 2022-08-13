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
from misc import run, me, cd, exists
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
of.write("\nwait".encode())

'''
if True:  # process files in python folder
    files = os.popen('ls -1 /home/' + me() + 'GitHub' + 
    elif ext == 'py':
        wrap_file = 'wrap-py_' + fn + '.cpp'
        # for py files, write a cpp wrapper so we can call from bin folder
        cf = open(wrap_file, 'wb') #'wrap_py.cpp', 'wb')
        lines = ['#include<stdlib.h>',
                 '#include<iostream>',
                 '#include<string>',
                 'using namespace std;',
                 'int main(int argc, char ** argv){',
                 #'  string cmd("/cygdrive/c/Program\\\\ Files/Python35/python.exe ");',
                 '  string cmd("/cygdrive/c/Python27/python.exe ");',
                 '  cmd += string("R:/' + os.popen("whoami").read().strip() + '/bin/py/' + fn + '.py");',
                 '  for(int i=1; i<argc; i++){',
                 '    cmd += string(" ") + string(argv[i]);',
                 '  }',
                 'std::cout << cmd << endl;',
                 'system(cmd.c_str());',
                 'return(0);',
                 '}']
        cf.write('\n'.join(lines).encode())
        cf.close()
        cmd = 'g++ -w -O3 -o ' + fn + '.exe ' + ' ' + wrap_file #wrap_py.cpp '
        print('\t' + cmd)
        cmds += " " + cmd # a = os.system(cmd)
        if(True): # 20191113 might set to true later?
            cmd =  ("rm -f " + wrap_file) #wrap_py.cpp")
            cmds += "; " + cmd;
'''

of.close()

run("chmod 755 compile.sh")
run("./compile.sh")
