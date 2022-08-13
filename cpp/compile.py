#!/usr/bin/env python
'''compile all .cpp files in folder to .exe files. N.B. remove -w flag from g++ to examine warnings
20220813 add work queue
20220813 add python program wrapping
    Compile all python files in python folder using a C wrapper (access from common folder)
'''
import os
import sys
from misc import run, me

files = os.popen('ls -1 ' + get_cd() + '*.cpp').readlines()
files = [f.strip() for f in files]

of = open('compile.sh', 'wb')
of.write('#!/usr/bin/env bash'.encode())
for f in files:
    s = ''
    fn = f[:-4]
    symlink = 'ln -s ' + fn + '.exe ' + fn # symbolic link to path without .exe!
    if fn != "misc":
        if f[:4] == 'cuda':
            s = '\ntest ! -f ' + fn + '.exe && nvcc ' + fn + '.cpp misc.cpp -o ' + fn + '.exe; ' + symlink
            print(s)
        else:
            s = '\ntest ! -f ' + fn + '.exe && g++ -w -O4 ' + fn + '.cpp  misc.cpp -o ' + fn + '.exe -lpthread; ' + symlink
        of.write(s.encode())
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
run("multicore ./compile.sh")

