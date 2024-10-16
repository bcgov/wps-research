'''20241015 process RCM data (in zip format) retrieved from EODMS

Join the raster and csv data?
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
stuff=['Granule',
       'Acquisition Start Date',
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
        print('data not found: ' + z)

    my_list = []
    for s in stuff:
        my_list += [p[s]]
    print(my_list)
    out_csv.write(('\n' + ','.join([str(x) for x in my_list])).encode())

    if Granule not in data:
        data[Granule] = {}

    for s in stuff:
        data[Granule][s] = p[s]
out_csv.close()

for d in data:
    s = set(data[d])
    print(d, len(list(s)), len(data[d]),  s)

'''
{'EODMS RecordId': 26712791, 'Granule': 'RCM2_OK3028444_PK3278392_1_SC30MCPB_20240922_014242_CH_CV_MLC', 'Acquisition Start Date': '2024-09-22T01:42:42Z', 'Acquisition End Date': '2024-09-22T01:42:51Z', 'Satellite ID': 'RCM-2', 'Beam Mnemonic': 'SC30MCPB', 'Beam Mode Type': 'Medium Resolution 30m', 'Beam Mode Description': 'Compact-Pol ScanSAR 30m Resolution 125km Swath B', 'Beam Mode Version': 2, 'Spatial Resolution': 30.0, 'Polarization Data Mode': 'Compact', 'Polarization': 'CH CV', 'Polarization in Product': 'CH CV XC', 'Number of Azimuth Looks': 2, 'Number of Range Looks': 2, 'Incidence Angle (Low)': 15.0, 'Incidence Angle (High)': 43.0, 'Orbit Direction': 'Ascending', 'LUT Applied': 'Unity-beta', 'Product Format': 'GeoTIFF', 'Product Type': 'MLC', 'Product Ellipsoid': 'WGS 1984', 'Sample Type': 'Mixed', 'Sampled Pixel Spacing': 7.9, 'Data Type': 'Floating-Point', 'SIP Size (MB)': 378, 'Relative Orbit': 166, 'Absolute Orbit': 28766, 'Orbit Data Source': 'Definitive', 'thumbnailUrl': 'https://www.eodms-sgdot.nrcan-rncan.gc.ca/wes/getObject?FeatureID=62f0e816-8006-4768-8f32-6ef4008e6895-26712791&ObjectType=Thumbview&collectionId=RCMImageProducts&AFB=true'}
'''


n = 0

for z in zip_files:
    Granule = ('.'.join(z.split('.')[:-1])).split(sep)[-1]

    if str(data[Granule]['Relative Orbit']) != '84': #  or str(data[Granule]['Satellite ID']) != 'RCM-1':
        print(Granule, z, 'PASS')
        continue
    else:
        print(Granule, z, "RUN *** ")

    if Granule not in data:
        err('frame not found in metadata')
    p_0 = z 
    p_1 = z + '_MLK.dim'
    p_2 = z + '_MLK_TF.dim'
    p_3 = z + '_MLK_TF_TC.dim'
    p_4 = z + '_MLK_TF_TC_box.dim'
 
    if not exist(p_1) and not exist(p_2) and not exist(p_3) and not exist(p_4):
        run([snap,
             'Multilook',
             '-PnAzLooks=4',
             '-PnRgLooks=4',
             '-Ssource=' + p_0,
             '-t ' + p_1])

    if not exist(p_2) and not exist(p_3) and not exist(p_4):
        run([snap,
             'Terrain-Flattening',
             '-Ssource=' + p_1,
             '-t ' + p_2])

    if not exist(p_3) and not exist(p_4):
        run([snap,
             'Terrain-Correction',
             '-PnodataValueAtSea=true',
             '-PoutputComplex=true',
             '-Ssource=' + p_2,
             '-t ' + p_3])  # output
   
    if not exist(p_4):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_3,
            '-t ' + p_4]) # output
    n += 1
    #if n > 2:
    #    sys.exit(1)

