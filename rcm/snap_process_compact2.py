''' 20230528 snap processing for CP-pol datasets
    1) calibration
    2) box filter 5x5
    3) multilook 2x2
    4) terrain correction (30m ESA dem)
    5) box filter 5x5

       convert to ENVI format
    6) merge by date

    7) project series onto first date

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
    print(c)
    c = ' '.join(c) if type(c) == list else c
    return os.system(c)

FILTER_SIZE = 5

# look for snap binary
snap = '/usr/local/snap/bin/gpt' # assume we installed snap
snap = '/opt/snap/bin/gpt' if not exist(snap) else snap  # try another location if that failed
if not exist(snap):
    err("could not find snap/bin/gpt")

folders, zips = [], [x.strip() for x in os.popen("ls -1 RCM*.zip").readlines()]
for z in zips:
    slc = '.'.join(z.split(".")[:-1])
    if not exist(slc):
        run("unzip " + z + " " + "-d " + slc)
    if exist(slc + sep + slc):
        run("mv -v " + slc + sep + slc + sep + "* " + slc + sep) 
        run("rmdir " + slc + sep + slc + sep)
    folders += [slc]
    print(slc)

# process each folder
i = 0
for p in folders:
    print("i=", str(i + 1), "of", str(len(folders)))
    p_1 = p + sep + 'manifest.safe'  # input
    p_2 = p + sep + '01_Cal.dim' # calibrated product
    p_3 = p + sep + '02_Cal_Spk.dim'  # terrain corrected output
    p_4 = p + sep + '03_Cal_Spk_Mlk.dim'
    p_5 = p + sep + '04_Cal_Spk_Mlk_TC.dim' # filtered output
    p_6 = p + sep + '05_Cal_Spk_Mlk_TC_Spk.dim'
    p_7 = p + sep + '05_Cal_Spk_Mlk_TC_Spk.data/stack.bin'

    if not exist(p_2):
        run([snap, 'Calibration',
             '-Ssource=' + p_1,
             '-t ' + p_2,
             '-PoutputImageInComplex=true'])

    if not exist(p_3):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_2,
            '-t ' + p_3])  # output

    if not exist(p_4):
        run([snap, 'Multilook',
             '-PnAzLooks=2',
             '-PnRgLooks=2',
             '-Ssource=' + p_3,
             '-t ' + p_4])  # output

    if not exist(p_5):
        run([snap, 'Terrain-Correction',
            '-PnodataValueAtSea=false',
            '-Ssource=' + p_4, 
            '-t ' + p_5])  # output

    if not exist(p_5):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_4,
            '-t ' + p_5]) # output

    if not exist(p_6):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_5,
            '-t ' + p_6])  # output

    if not exist(p_7):
        dat = '.'.join(p_6.split('.')[:-1]) + '.data'
        run(['snap2psp.py',  # convert to PolSARPro format
             dat,
             '1'])  # stack the bands

        run(['raster_zero_to_nan', p_7])

    i += 1
# now merge things of the same date


date = {}
last = [x.strip() for x in os.popen('find ./ -name "05_Cal_Spk_Mlk_TC_Spk.data"').readlines()]
for L in last:
    slc = L.split(sep)[-2]
    d = slc.split("_")[5]
    if d not in date: date[d] = []
    date[d] += [L]

for d in date:
    print(d, date[d])
    files = [x + sep + 'stack.bin' for x in date[d]]

    ofn = str(d) + '.bin'
    cmd = 'merge.py ' + ' '.join(files) + ' ' + ofn
    print([cmd])
    if not os.path.exists(ofn):
        a = os.system(cmd)
