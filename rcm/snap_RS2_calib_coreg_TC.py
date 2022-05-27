'''20220527 in SNAP /snappy..: for QUAD-pol radarsat2 datasets..
..on all SLC folders in working directory:
    1) calibrate
    2) coregister
    3) Terrain correction
'''

import os
import sys
sep = os.path.sep
from datetime import date
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + ".." + sep + 'py')
from misc import run, pd, sep, exist, args, cd, err

folders = os.popen('ls -1').readlines()
folders_use = []
for f in folders:
    f = f.strip()
    if f[:3] == 'RS2' and f[-3:] == 'SLC':
        if os.path.isdir(f):
            folders_use.append(f.strip())

for f in folders_use:
    print(f)

snap, ci = '/usr/local/snap/bin/gpt', 0 # assume we installed snap
if not exist(snap):
    snap = '/opt/snap/bin/gpt'  # try another location if that failed
if not exist(snap):
    snap = '/home/' + os.popen('whoami').read().strip() + sep + 'snap' + sep + 'bin' + sep + 'gpt'
folders = folders_use

for p in folders:
    ci += 1
    if os.path.abspath(p) == os.path.abspath('.'):
        continue

    p_1 = p + sep + 'product.xml' # manifest.safe'  # input
    p_2 = p + sep + '01_calib.dim' # calibrated product
    #p_3 = p + sep + '02_tc.dim'  # terrain corrected output
    #p_4 = p + sep + '03_filter.dim' # filtered output

    if not exist(p_2) or RE_CALC:
        run([snap,
             'Calibration',
             '-Ssource=' + p_1,
             '-PoutputImageInComplex=true'])  # the disappearing option!
        run(['cp -rv', 'target.dim', p_2])
        run(['cp -rv', 'target.data', p_2[:-4] + '.data'])


'''
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
'''

