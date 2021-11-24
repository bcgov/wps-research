'''after:
    1) rcm_group_beam.py
    2) snap_TC_box.py

run mf3cc on each dataset.'''
import os
import sys
import shutil
sep = os.path.sep
exists = os.path.exists

def run(c):
    print(c)
    return os.system(c)
def err(m):
    print('error:',m)
    sys.exit(1)

beams = []
files = [x.strip() for x in os.popen('ls -1').readlines()]
for f in files:
    if f[:3] == 'RCM':
        w = f.split('_')
        beam = w[4]
        # print(beam, f)
        beams.append(beam)

no_rcm = True
for f in files:
    if f[:3] == 'RCM':
        no_rcm = False

if no_rcm:
    files = [x.strip() for x in os.popen('ls -1').readlines()]
    beams = files

for beam in beams:
    #print("beam", beam)
    files = [x.strip() for x in os.popen('ls -1 ' + beam).readlines()]
    for f in files:
        ds = beam + sep + f
        #print('\tset',ds)
        fils = [x.strip() for x in os.popen('ls -1 ' + ds + sep + '*rgb.bin').readlines()]
        
        dates = []
        g_d = {}
        for g in fils:
            #print('\t\t',g)
            date = (g[:-4].split(sep)[-1].split('_')[5])
            dates.append(date)
            g_d[date] = g
        if len(dates) != len(list(set(dates))):
            err("multiple dates per beam/set")

        for d in dates:
            g = g_d[d]
            df = ds + sep + d
            if not exists(df):
                os.mkdir(df)
                #print('mkdir',df)
            src, tgt = g, df + sep + 'rgb.bin'
            c = (' '.join(['cp', src, tgt]))
            if not exists(tgt):
                run(c)
            else:
                pass #print(c)

            src, tgt = g[:-4] + '.hdr', df + sep + 'rgb.hdr'
            c = (' '.join(['cp', src, tgt]))
            if not exists(tgt):
                run(c)
            else:
                pass # print(c)

