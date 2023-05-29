''' 20230528 snap processing for CP-pol datasets
    1) calibration
    2) box filter (5)
    2) terrain correction (30m ESA dem)
    3) box filter (5)
    4) run the CP-pol decompositions

Mosaic in GDAL? 
zero_to_nan and c2_stokes

5) todo: mosaic (on date), stack and project to match
NB don't need snappy installed / configured to run this.

To get help:
  /opt/snap/bin/gpt -h

Had this error:
https://forum.step.esa.int/t/java-verify-error/36630

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
from misc import pd, exist, args, cd, err

def run(c):
    if type(c) == list:
        c = ' '.join(c)
    print(c)
    return os.system(c)

FILTER_SIZE = 5
RE_CALC = False

# look for snap binary
snap, ci = '/usr/local/snap/bin/gpt', 0 # assume we installed snap
if not exist(snap):
    snap = '/opt/snap/bin/gpt'  # try another location if that failed

zips = [x.strip() for x in os.popen("ls -1 *.zip").readlines()]
for z in zips:
    slc = '.'.join(z.split(".")[:-1])
    if not exist(slc):
        run("unzip " + z + " " + "-d " + slc)

    if exist(slc + sep + slc):
        run("mv -v " + slc + sep + slc + sep + "* " + slc + sep) 
        run("rmdir " + slc + sep + slc + sep)


# find SLC folders in present directory
folders = [x.strip() for x in os.popen('ls -1 | grep _SLC').readlines()]
folders_use = []
for f in folders:
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
    p_2 = p + sep + '01_Cal.dim' # calibrated product
    p_3 = p + sep + '02_Cal_Spk.dim'  # terrain corrected output
    p_4 = p + sep + '03_Cal_Spk_Mlk.dim'
    p_5 = p + sep + '04_Cal_Spk_Mlk_TC.dim' # filtered output
    p_6 = p + sep + '05_Cal_Spk_Mlk_TC_Spk.dim'

    if not exist(p_2) or RE_CALC:
        run([snap,
             'Calibration',
             '-Ssource=' + p_1,
             '-t ' + p_2,
             '-PoutputImageInComplex=true'])

    if not exist(p_3) or RE_CALC:
        run([snap,
            'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_2,
            '-t ' + p_3])  # output folder

    if not exist(p_4) or RE_CALC:
        run([snap,
             'Multilook',
             '-PnAzLooks=2',
             '-PnRgLooks=2',
             '-Ssource=' + p_3,
             '-t ' + p_4])

    if not exist(p_5) or RE_CALC:
        run([snap,
            'Terrain-Correction',
            '-PnodataValueAtSea=false',
            '-Ssource=' + p_4, 
            '-t ' + p_5])

    if not exist(p_5) or RE_CALC:
        run([snap,
            'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_4,
            '-t ' + p_5])  # output foldera

    if not exist(p_6) or RE_CALC:
        run([snap,
            'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_5,
            '-t ' + p_6])  # output folder
