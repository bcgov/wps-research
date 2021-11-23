'''Group RCM files / folders, into folders labelled by the beam mode'''
import os
import sys
sep = os.path.sep
exists = os.path.exists

def run(c):
    print(c)
    return os.system(c)

files = [x.strip() for x in os.popen('ls -1').readlines()]
for f in files:
    if f[:3] == 'RCM':
        w = f.split('_')
        beam = w[4]
        print(beam, f)
        if not exists(beam):
            print('+w', beam)
            os.mkdir(beam)
        run('mv -v ' + f + ' ' + beam)
