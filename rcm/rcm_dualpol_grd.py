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
    p1 = d + sep + '01_Cal.dim'
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


    p3 = d + sep + '02_TC.data' + sep + 'Sigma_HH.bin'
    
    if not exist(p3):
        a = os.system('cd ' + d + sep + '02_TC.data; snap2psp_inplace.py')
    
    p4 = d + sep + '02_TC.data' + sep + d + '_rgb.bin'
    if not exist(p4):
        cmd = ('cd ' + d + sep + '02_TC.data; raster_div Sigma0_HH.bin Sigma0_HV.bin Sigma0_HH_HV.bin; cp Sigma0_HH.hdr Sigma0_HH_HV.hdr; raster_stack.py Sigma0_HH.bin Sigma0_HV.bin Sigma0_HH_HV.bin ' + d + '_rgb.bin; raster_zero_to_nan ' + d + '_rgb.bin')
        print(cmd)
        a = os.system(cmd)

    hf = d + sep + '02_TC.data' + sep + d + '_rgb.hdr'
    dd = open(hf).read().replace('Sigma0_HH}', 'Sigma0_HH/HV}')
    open(hf, 'wb').write(dd.encode())

    
