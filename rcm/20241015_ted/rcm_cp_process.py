'''20241015 process RCM data (in zip format) retrieved from EODMS
'''
FILTER_SIZE = 3
import os
import sys
import glob
import json
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + ".." + sep + '..' + sep + 'py')
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

zip_files = glob.glob(os.path.join(os.getcwd(), 'RCM*.zip'))
zip_files.sort()
print(zip_files)

json_file = 'query_results_MLC.json'
j = json.load(open(json_file))

'''type
name
crs
features
'''
stuff=['Acquisition Start Date',
       'Acquisition End Date',
       'Satellite ID',
       'Beam Mnemonic',
       'Beam Mode Type',
       'Beam Mode Description',
       'Beam Mode Version',
       'Orbit Direction',
       'Relative Orbit', 
       'Absolute Orbit']
data = {}
print(stuff)
out_csv = open('datasets.csv', 'wb')
out_csv.write((','.join(stuff)).encode())
for f in j['features']:
    p = f['properties']
    print(p)
    Granule = p['Granule']
    z = Granule + '.zip'
    if not os.path.exists(z):
        err('data not found: ' + z)

    my_list = []
    for s in stuff:
        if s not in data:
            data[s] = []
        data[s] += [p[s]]
        my_list += [p[s]]
    print(my_list)
    out_csv.write(('\n' + ','.join(my_list)).encode())
out_csv.close()

for d in data:
    s = set(data[d])
    print(d, len(list(s)), len(data[d]),  s)

'''
{'EODMS RecordId': 26712791, 'Granule': 'RCM2_OK3028444_PK3278392_1_SC30MCPB_20240922_014242_CH_CV_MLC', 'Acquisition Start Date': '2024-09-22T01:42:42Z', 'Acquisition End Date': '2024-09-22T01:42:51Z', 'Satellite ID': 'RCM-2', 'Beam Mnemonic': 'SC30MCPB', 'Beam Mode Type': 'Medium Resolution 30m', 'Beam Mode Description': 'Compact-Pol ScanSAR 30m Resolution 125km Swath B', 'Beam Mode Version': 2, 'Spatial Resolution': 30.0, 'Polarization Data Mode': 'Compact', 'Polarization': 'CH CV', 'Polarization in Product': 'CH CV XC', 'Number of Azimuth Looks': 2, 'Number of Range Looks': 2, 'Incidence Angle (Low)': 15.0, 'Incidence Angle (High)': 43.0, 'Orbit Direction': 'Ascending', 'LUT Applied': 'Unity-beta', 'Product Format': 'GeoTIFF', 'Product Type': 'MLC', 'Product Ellipsoid': 'WGS 1984', 'Sample Type': 'Mixed', 'Sampled Pixel Spacing': 7.9, 'Data Type': 'Floating-Point', 'SIP Size (MB)': 378, 'Relative Orbit': 166, 'Absolute Orbit': 28766, 'Orbit Data Source': 'Definitive', 'thumbnailUrl': 'https://www.eodms-sgdot.nrcan-rncan.gc.ca/wes/getObject?FeatureID=62f0e816-8006-4768-8f32-6ef4008e6895-26712791&ObjectType=Thumbview&collectionId=RCMImageProducts&AFB=true'}
'''


n = 0

for z in zip_files:
    print(z)
    p_0 = z 
    p_1 = z + '_TF.dim'
    p_2 = z + '_TF_TC.dim'
    p_3 = z + '_TF_TC_box.dim'

    if not exist(p_1) and not exist(p_2) and not exist(p_3):
        run([snap,
             'Terrain-Flattening',
             '-Ssource=' + p_0,
             '-t ' + p_1])

    if not exist(p_2) and not exist(p_3):
        run([snap,
             'Terrain-Correction',
             '-PnodataValueAtSea=true',
             '-PoutputComplex=true',
             '-Ssource=' + p_1,
             '-t ' + p_2])  # output
      

    if not exist(p_3):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_2,
            '-t ' + p_3]) # output
    n += 1
    if n > 1:
        sys.exit(1)

'''
Usage:
  gpt Terrain-Flattening [options] 

Description:
  Terrain Flattening


Source Options:
  -Ssource=<file>    Sets source 'source' to <filepath>.
                     This is a mandatory source.

Parameter Options:
  -PadditionalOverlap=<double>                The additional overlap percentage
                                              Valid interval is [0, 1].
                                              Default value is '0.1'.
  -PdemName=<string>                          The digital elevation model.
                                              Default value is 'SRTM 1Sec HGT'.
  -PdemResamplingMethod=<string>              Sets parameter 'demResamplingMethod' to <string>.
                                              Default value is 'BILINEAR_INTERPOLATION'.
  -PexternalDEMApplyEGM=<boolean>             Sets parameter 'externalDEMApplyEGM' to <boolean>.
                                              Default value is 'false'.
  -PexternalDEMFile=<file>                    Sets parameter 'externalDEMFile' to <file>.
  -PexternalDEMNoDataValue=<double>           Sets parameter 'externalDEMNoDataValue' to <double>.
                                              Default value is '0'.
  -PnodataValueAtSea=<boolean>                Mask the sea with no data value (faster)
                                              Default value is 'true'.
  -PoutputSigma0=<boolean>                    Sets parameter 'outputSigma0' to <boolean>.
                                              Default value is 'false'.
  -PoutputSimulatedImage=<boolean>            Sets parameter 'outputSimulatedImage' to <boolean>.
                                              Default value is 'false'.
  -PoversamplingMultiple=<double>             The oversampling factor
                                              Valid interval is [1, 4].
                                              Default value is '1.0'.
  -PsourceBands=<string,string,string,...>    The list of source bands.
'''


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
             '-Pmatrix=C2'])
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
