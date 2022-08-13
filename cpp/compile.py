#!/usr/bin/env python
'''
compile all .cpp files in folder to .exe files
remove -w flag from g++ to examine warnings
20220813 add work queue
20220813 add python program wrapping
'''
import os
import sys
import multiprocessing as mp
i, n_cpu = 0, mp.cpu_count()

files = os.popen('ls -1 *.cpp').readlines()
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
        # s += '' if ((i + 1) % n_cpu == 0) else ' &'  # don't need this since we'll use multicore
        of.write(s.encode())
        i += 1
of.write("\nwait".encode())
of.close()

a = os.system("chmod 755 compile.sh")
a = os.system("multicore ./compile.sh")

