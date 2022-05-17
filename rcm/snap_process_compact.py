''' 20220515 for CP-pol datasets
    1) calibration
    2) terrain correction
    3) box filter x3
    4) run the CP-pol decompositions

5) todo: mosaic (on date), stack and project to match

20220515 warning: complex output from TC feature was removed, but is now put back again (still not in the online version of SNAP yet)
    https://forum.step.esa.int/t/outputcomplex-argument-removed-from-sentinel-1-terrain-correction/35013

from jun_lu (Feb 28, 2022 SNAP forum):
    "The complex output option has been removed from terrain correction operator based on the suggestions of the ESA scientists. This is because the result is scientifically totally wrong. Sorry about the confusion"

NB don't forget to export dem, lat/long, geometric parameters'''
import os
import sys
sep = os.path.sep
from datetime import date
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + "../py/")
from misc import run, pd, exist, args, cd, err

FILTER_SIZE = 7
RE_CALC = True

# look for snap binary
snap, ci = '/usr/local/snap/bin/gpt', 0 # assume we installed snap
if not exist(snap):
    snap = '/opt/snap/bin/gpt'  # try another location if that failed

# find SLC folders in present directory
folders = os.popen('ls -1 | grep _SLC').readlines()
folders_use = []
for f in folders:
    f = f.strip()
    if f[-3:] == 'SLC':
        if os.path.isdir(f):
            folders_use.append(f.strip())
folders = folders_use
for p in folders_use:
    print(p)

# process each folder
for p in folders:
    ci += 1
    if os.path.abspath(p) == os.path.abspath('.'):
        continue

    p_1 = p + sep + 'manifest.safe'  # input
    p_2 = p + sep + '01_calib.dim' # calibrated product
    p_3 = p + sep + '02_tc.dim'  # terrain corrected output
    p_4 = p + sep + '03_filter.dim' # filtered output

    if not exist(p_2) or RE_CALC:
        run([snap,
             'Calibration',
             '-Ssource=' + p_1,
             '-PoutputImageInComplex=true'])  # the disappearing option!
        run(['cp -rv', 'target.dim', p_2])
        run(['cp -rv', 'target.data', p_2[:-4] + '.data'])

    if not exist(p_3) or RE_CALC:
        run([snap,
            'Terrain-Correction',
            '-PoutputComplex=true',
            '-PnodataValueAtSea=false',
            # '-PsaveLayoverShadowMask=true',
            # '-PimgResamplingMethod="NEAREST_NEIGHBOUR"', # why? because we will add an index for reverse geocoding...
            '-Ssource=target.dim', # + p_2,
            '-t ' + p_3])

    if not exist(p_4) or RE_CALC:
        run([snap,
            'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE), # 3',
            '-Ssource=' + p_3,
            '-t ' + p_4])  # -t is for output file

    # run all the CP-pol decoms
    decoms = ["'M-Chi Decomposition'",
              "'M-Delta Decomposition'",
              "'H-Alpha Decomposition'",
              "'2 Layer RVOG Model Based Decomposition'",
              "'Model-free 3-component decomposition'"]

    decom_folders = []
    for decom in decoms:
        decom_name = decom.strip("'").replace(' ', '_')
        p_5 = p + sep + decom_name + '.dim'
        decom_folders.append(p_5)
        
        if not exist(p_5) or RE_CALC:
            run([snap,
                 'CP-Decomposition',
                 '-Pdecomposition=' + decom,
                 '-PwindowSizeXStr=' + str(FILTER_SIZE),
                 '-PwindowSizeYStr=' + str(FILTER_SIZE),
                 '-Ssource=' + p_4,
                 '-t ' + p_5])

        stack_f = p_5[:-3] + 'data' + sep + 'stack.bin'
        print(stack_f)
        if not exist(stack_f) or RE_CALC:
            cmd = ['python3',
                    pd + 'snap2psp_inplace.py',
                    p_5[:-3] + 'data' + sep,
                    '1']
            run(cmd)

'''
    # 20211125 analysis
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
'''
