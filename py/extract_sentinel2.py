from misc import *

if len(args) < 2:
    err('python3 extract_sentinel2.py [input sentinel2 zip file name]')

# check gdal version
info = os.popen("gdalinfo --version").read().strip().split(',')
info = info[0].split()[1].replace('.', '')
if int(info) < 223:
    err('GDAL version 2.2.3 or higher required')
    
fn = args[1]
if fn[-4:] != '.zip':
    err('expected zip format input')

if not os.path.exists(fn):
    err('could not find input file')

df = fn[:-4] + '.SAFE'
print(df)
if not os.path.exists(df):
    a = os.system('unzip ' + fn)
    import time
    time.sleep(1.)

if not os.path.exists(df):
    err('failed to unzip')

xml = os.popen('gdalinfo ' + fn + '  |  grep SUBDATA').readlines()

cmds = []
for line in xml:
    line = line.strip()
    if len(line.split('.xml')) > 1:
        print('\t' + line)
        try:
            # need to match MTD_MSIL1C.xml and SENTINEL2_L1C
            # print("df", str([df]))
            df = df.split(os.path.sep)[-1] # print("split on:", df)
            dfw = line.split(df)
            
            terminator = dfw[-1].strip(os.path.sep).split(':')[0] # print("terminator", terminator)

            # print("dfw", str([dfw]))
            ident = dfw[0].split('=')[1].split(':')[0]
            # print("ident", str([ident]))
            ds = ident + ':' + df + dfw[1]
            # print("ds", str([ds]))
            of = (df + dfw[1]).replace(terminator, ident).replace(':', '_') + '.bin'
            # print("of", str([of]))
            cmd = ['gdal_translate', ds, '--config GDAL_NUM_THREADS 8', '-of ENVI', '-ot Float32', of]
            cmds.append(' '.join(cmd))
            print('\t' + cmd)
        except Exception:
            pass
for cmd in cmds:
    # print(cmd)
    run(cmd)

'''
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:10m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:20m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:60m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:TCI:EPSG_32610
'''
# gdal_translate SENT--roi_x_y=INEL2_L1C:S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:10m:EPSG_32610   --config GDAL_NUM_THREADS 32 -of ENVI -ot Float32 out.bin
