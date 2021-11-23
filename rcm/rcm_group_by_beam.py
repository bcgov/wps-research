'''Group RCM files / folders, into folders labelled by the beam mode..
..and then subfolders labelled by set id'''
import os
import sys
import shutil
sep = os.path.sep
exists = os.path.exists

def run(c):
    print(c)
    return os.system(c)

beams = []
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
        beams.append(beam)

no_rcm = True
for f in files:
    if f[:3] == 'RCM':
        no_rcm = False

if no_rcm:
    files = [x.strip() for x in os.popen('ls -1').readlines()]
    beams = files

for beam in beams:
    print("beam", beam)
    files = [x.strip() for x in os.popen('ls -1 ' + beam).readlines()]
    for f in files:
        if f[:3] == 'RCM':
            print('\t' + f)
            w = f.strip().split('_')
            ds = beam + sep + w[3]
            print('\t\t' + ds)
            if not exists(ds):
                os.mkdir(ds)

            src = beam + sep + f
            dst = ds + sep + f
            shutil.move(src, dst)
