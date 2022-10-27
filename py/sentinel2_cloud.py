'''get cloud and scene classif masks'''
from misc import * 
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep
project = pd + 'raster_project_onto.py' # project onto command

d = os.path.abspath(args[1] if len(args) > 1 else os.getcwd())
print(d)

if d[-5:] != '.SAFE':
    print('expected folder ending in .SAFE')
    err('python3 sentinel2_cloud.py [Sentinel2 L2 folder: ___.SAFE] ')

CLD = os.popen('find ' + d + os.path.sep + ' -name "*CLD*20m.jp2"').readlines()
CLD = [x.strip() for x in CLD]

if len(CLD) > 1:
    print(CLD)
    err('too many files, check folder')
CLD = CLD[0]

d += sep

out_20m = d + 'cloud_20m.bin'
if not exists(out_20m):
    run(' '.join(['gdal_translate',
                  '-of ENVI',
                  '-ot Float32', 
                  CLD,
                  out_20m]))

out_file = d + 'cloud.bin'
f10m = os.popen('find ' + d + ' -name "SENTINEL2_L2A_EPSG*_10m.bin"').read().strip()
if len(f10m.split('\n')) > 1:
    err('multiple files found when one expected:' + str(f10m))

USE_20M = False
if f10m.strip() != '':
    f10m = os.path.abspath(f10m)

    if not exists(out_file):
        run(' '.join(['python3',
                      project,
                      out_20m,
                      f10m, # d + 'SENTINEL2_L2A_EPSG_32610_10m.bin',
                      out_file]))
else:
    USE_20M = True # we just need cloud at 20m!

''' From https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
Figure 3: Scene Classification Values Label 	Classification
0 	NO_DATA
1 	SATURATED_OR_DEFECTIVE
2 	DARK_AREA_PIXELS
3 	CLOUD_SHADOWS
4 	VEGETATION
5 	NOT_VEGETATED
6 	WATER
7 	UNCLASSIFIED
8 	CLOUD_MEDIUM_PROBABILITY
9 	CLOUD_HIGH_PROBABILITY
10 	THIN_CIRRUS
11 	SNOW
'''

cmd = 'find ' + d + ' -name "*SCL_20m.jp2"'
SCL = os.popen(cmd).readlines()
SCL = [x.strip() for x in SCL]

if len(SCL) > 1:
    print(SCL)
    err('too many files, check folder')
SCL = SCL[0]

out_20m = d + 'class_20m.bin'
if not exists(out_20m):
    run(' '.join(['gdal_translate',
                  '-of ENVI',
                  '-ot Float32',
                  SCL,
                  out_20m]))

if not USE_20M:
    out_file = d + 'class.bin'
    if not exists(out_file):
        run(' '.join(['python3',
                      project,
                      out_20m,
                      f10m, #d + 'SENTINEL2_L2A_EPSG_32610_10m.bin',
                      out_file]))
