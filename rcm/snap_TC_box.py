'''20211123 in SNAP /snappy..: for COMPACT-pol datasets..
..on all SLC folders in working directory:
    1) perform terrain correction followed by
    2) box filter..
    3) finally, swap the byte order for use in PolSARPro (or outside SNAP)
This actually placed the C2 matrix, in a ENVI/PolSARPro file


20220515 warning: complex output from TC feature was removed, but is now put back again (still not in the online version of SNAP yet)
    https://forum.step.esa.int/t/outputcomplex-argument-removed-from-sentinel-1-terrain-correction/35013

'''
import os
import sys
sep = os.path.sep
exist = os.path.exists

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    w = c.strip().split(' ')
    print('run ' + w[0])
    for i in range(1, len(w)):
        print('    ' + w[i])
    a = os.system(c)
    if a != 0: err('command failed: ' + c)

# folders = [x.strip() for x in os.popen("ls -1 -d *SLC").readlines()] # snap = '/home/' + os.popen('whoami').read().strip()
folders = os.popen('find ./').readlines()
folders_use = []
for f in folders:
    f = f.strip()
    if f[-3:] == 'SLC':
        if os.path.isdir(f):
            folders_use.append(f.strip())

for f in folders_use:
    print(f)

snap, ci = '/usr/local/snap/bin/gpt', 1 # assume we installed snap
if not exist(snap):
    snap = '/opt/snap/bin/gpt'  # try another location if that failed
folders = folders_use

for p in folders:
    print('*** processing ' + str(ci) + ' of ' + str(len(folders)))
    if os.path.abspath(p) == os.path.abspath('.'): continue

    in_1 = p + sep + 'manifest.safe'  # input
    in_2 = p + sep + 'tc.dim'  # terrain corrected output
    c1 = ' '.join([snap,
                  'Terrain-Correction',
                  '-PoutputComplex=true',
                  '-PnodataValueAtSea=false', # '-PsaveLayoverShadowMask=true',
                  '-PimgResamplingMethod="NEAREST_NEIGHBOUR"', # why? because we will add an index for reverse geocoding...
                  in_1, '-t ' + in_2])

    in_3 = p + sep + 'b7.dim'  # box filtered output # how to get parameters: ./gpt Polarimetric-Speckle-Filter -h
    c2 = ' '.join([snap,
                  'Polarimetric-Speckle-Filter',
                  '-Pfilter="Box Car Filter"',
                  '-PfilterSize=7',
                  in_2, '-t', in_3])  # -t is for output file

    use_C = False  # default to T matrix.
    if not exist(in_2):
        run(c1)
    if not exist(in_3):
        run(c2)
    r_f, r_h = p + '_rgb.bin', p + '_rgb.hdr'
    hf = in_3[:-3] + 'data' + sep + 'T11.hdr'
    if exist(hf):
        if not exist(r_h):
            run('cp ' + hf + ' ' + r_h)

    else:
        hf = in_3[:-3] + 'data' + sep + 'C11.hdr'
        if exist(hf):
            use_C = True
            if not exist(r_h):
                run('cp ' + hf + ' ' + r_h)
        else:
            err('not found', hf)

    dat = open(r_h).read().strip()
    dat = dat.replace("bands = 1", "bands = " + ("3" if not use_C else "4"))
    dat = dat.replace("byte order = 1", "byte order = 0")
    if not use_C:
        dat = dat.replace("band names = { T11 }",
                          "band names = {T22,\nT33,\nT11}")
    else:
        dat = dat.replace("band names = { C11 }",
                          "band names = {C11,\nC22,\nC12_real,\nC12_imag}")

    open(r_h, 'wb').write(dat.encode())  # write revised header
    print(r_h, ':')
    a = os.system('cat ' + r_h)

    if not exist(r_f):
        file_pre = 'T' if not use_C else 'C'  # file prefix: T or C mtx
        c, t = 'cat', in_3[:-3] + 'data' + sep + file_pre # ('T' if not use_C else 'C')
        file_i = ['22.bin', '33.bin', '11.bin'] if not use_C else ['11.bin', '22.bin', '12_real.bin', '12_imag.bin']
        for i in file_i:
            # (['22.bin', '33.bin', '11.bin'] if not use_C else ['11.bin', '22.bin', '12_real.bin', '12_imag.bin']) :
            ti = t + i
            if not exist(ti):
                run('./sbo ' + (ti[:-3] + 'img') + ' ' + ti + ' 4')
            c += (' ' + t + i)
        c += ' > ' + r_f
        run(c)
    ci += 1
