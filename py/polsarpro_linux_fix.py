'''20241113 fix polsarpro source code to compile on Ubuntu 20

Code needs to be modified to be compatible with a "modern" GCC (v >= 10.) 
https://forum.step.esa.int/t/polsar-pro-installation-problem-on-linux/42647/4

Solution? Add -fcommon to every CFLAGS arg in makefile
'''
import os
import sys


lines = [x.strip() for x in os.popen('find ./ -name "Makefile*linux"').readlines()]

for line in lines:
    # print('+r', line)
    changed = False
    data = open(line).read().split('\n')
    for i in range(len(data)):
        d = data[i]
        if len(d.split('CFLAGS')) > 1:
            if d.strip().split()[0] == 'CCFLAGS':
                if len(d.split('-fcommon')) < 2 and len(d.split('+=')) < 2:
                    print(line, d)
                    print(d)
                    data[i] = (d + ' -fcommon')
                    changed = True

    if changed:
        print("+w", line)
        open(line, 'wb').write(('\n'.join(data)).encode())
