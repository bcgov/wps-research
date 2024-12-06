'''20241205 simple processing of RCM dual-pol SAR data in GRD format

run unzp command first
'''
FILTER_SIZE = 3
import os
import sys
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + ".." + sep + 'py')
from misc import pd, sep, exist, args, cd, err, find_snap

def run(x):
    cmd = ' '.join(x)
    print(cmd)
    a = os.system(cmd)


snap = find_snap()  # find snap gpt binary

sets = [x.strip() for x in os.popen('ls -1d RCM*GRD').readlines()]

for d in sets:
    p0 = d + sep + 'manifest.safe'
    p1 = d + sep + '01_calib.dim'
    p2 = d + sep + '02_TC.dim'
    if not exist(p1):
        run([snap, 'Calibration',
             '-Ssource=' + p0,
             '-t ' + p1,
             '-PoutputImageInComplex=true']) # '-PoutputBetaBand=true'])

    if not exist(p2):
        run([snap, 'Terrain-Correction',
             '-PnodataValueAtSea=true',
             '-Ssource=' + p1,
             '-PdemName="Copernicus 30m Global DEM"',
             '-t ' + p2])
