'''20230123 process JAXA data retrieved from EODMS
*** Assume each folder in present directory, is a dataset'''
FILTER_SIZE = 5
import os
import sys
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + ".." + sep + 'py')
from misc import pd, sep, exist, args, cd, err

def run(x):
    cmd = ' '.join(x)
    print(cmd)
    a = os.system(cmd)


snap, ci = '/usr/local/snap/bin/gpt', 0 # assume we installed snap
if not exist(snap):
    snap = '/opt/snap/bin/gpt'  # try another location if that failed
if not exist(snap):
    snap = '/home/' + os.popen('whoami').read().strip() + sep + 'snap' + sep + 'bin' + sep + 'gpt'

print(snap)

dirs = [f for f in os.listdir() if os.path.isdir(f)]
# print(dirs)


i = 0
for d in dirs:
    print("i=", str(i + 1), "of", str(len(dirs)))
    print(d)

    # look for VOL file
    vol_files = [f for f in os.listdir(d) if len(f.split('VOL')) > 1]
    if len(vol_files) > 1:
        err('expected only one *VOL* file')

    p_0 = d + sep + vol_files[0] # 'manifest.safe'  # input
    p_1 = d + sep + '01_Mlk.dim'
    p_2 = d + sep + '02_Cal.dim' # calibrated product
    p_3 = d + sep + '03_Mtx.dim'
    p_4 = d + sep + '04_Box.dim'
    p_5 = d + sep + '05_Ter.dim'
    p_6 = d + sep + '06_Box.dim'
    print(p_0)
    
    if not exist(p_1):
        run([snap,
             'Multilook',
             '-PnAzLooks=2',
             '-PnRgLooks=4',
             '-Ssource=' + p_0,
             '-t ' + p_1])
    print(p_1)

    if not exist(p_2):
        run([snap, 'Calibration',
             '-Ssource=' + p_1,
             '-t ' + p_2,
             '-PoutputImageInComplex=true'])
    print(p_2)

    if not exist(p_3):
        run([snap, 'Polarimetric-Matrices',
             '-Ssource=' + p_2,
             '-t ' + p_3,
             '-Pmatrix=T4'])
    print(p_3)

    if not exist(p_4):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_3,
            '-t ' + p_4]) # output
    print(p_4)
 
    if not exist(p_5):
        run([snap, 'Terrain-Correction',
            '-PnodataValueAtSea=true',
            '-Ssource=' + p_4,
            # '-PpixelSpacingInMeter=10.0',
            ' -PdemName="Copernicus 30m Global DEM"',
            '-t ' + p_5])  # output
    print(p_5)
    '''
    if not exist(p_6):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_5,
            '-t ' + p_6]) # output
    print(p_6)
    '''
    # sys.exit(1)  # comment out to run on first set only

    i += 1
