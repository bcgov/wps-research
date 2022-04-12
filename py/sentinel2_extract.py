'''Can Use this method to open Sentinel-2 data in Level1 from ESA Copernicus, or from Google Cloud Platform (after running gcp/fix_s2.py and zipping the folder)

* have to revisit what the no_stomp was used for. '''
import time
from misc import *
ehc = pd + 'envi_header_cleanup.py' # envi header cleanup command.. makes file open in "imv"

if len(args) < 2:
    err('python3 extract_sentinel2.py [input sentinel2 zip file name] # [optional parameter: no_stomp=True]')

no_stomp = False
if len(args) > 2:
    if args[2] == 'no_stomp' or args[2] == 'no_stomp=True':
        no_stomp = True

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

df = fn[:-4] + '.SAFE'  # extracted data folder
# print(df)
if not os.path.exists(df):
    if no_stomp == False:
        a = os.system('unzip ' + fn)
    else:
        a = os.system('mkdir -p ' + df)  # special no_stomp mode!! needed for using google cloud drive script
        a = os.system('unzip -d ' + df + ' ' + fn)
    time.sleep(1.)

if not os.path.exists(df):
    err('failed to unzip: cant find folder: ' + df)

# print("try gdalinfo..")
gdfn = fn[:-4] + '.SAFE/MTD_MSIL1C.xml' if no_stomp else fn
cmd = 'gdalinfo ' + gdfn + ' | grep SUBDATA'
print(cmd)
xml = os.popen(cmd).readlines()
# print("gdalinfo done.")

cmds = []  # commands to run after this section
data_files = []
found_line = False
for line in xml:
    line = line.strip()
    #print('  ' + line)
    if len(line.split('.xml')) > 1:
        #print('\t' + line)
        try:
            df = df.split(os.path.sep)[-1]
            safe = df # .SAFE directory
            dfw = line.split(df)
            terminator = dfw[-1].strip(os.path.sep).split(':')[0]
            ident = dfw[0].split('=')[1].split(':')[0]
            ds = ident + ':' + df + dfw[1]
            of = (df + dfw[1]).replace(terminator, ident).replace(':', '_') + '.bin'
            # print("DS: " + ds)
            # sys.exit(1)

            cmd = ' '.join(['gdal_translate', ds, '--config GDAL_NUM_THREADS 8', '-of ENVI', '-ot Float32', of])
            print('\t' + cmd)
            data_files.append(cmd)
            if not os.path.exists(of):  # don't extract if we did already!
                cmds.append(cmd)

            hfn = of[:-4] + '.hdr'
            cmd = ' '.join(['python3', ehc, hfn]) # of[:-4] + '.hdr']
            print('\t' + cmd)
            if not os.path.exists(hfn):
                 cmds.append(cmd)
                # print('\t' + cmd)

            found_line = True  # what line were we looking for?
        except Exception:
            pass

if not found_line:
    # we must be in google mode?
    # print("in google mode?")
    if no_stomp:
        # must be in google mode!
        print("definitely in google mode")
        sys.exit(1)

for cmd in cmds:
    # print(cmd)
    run(cmd)

'''e.g. outputs:
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:10m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:20m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:60m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:TCI:EPSG_32610
'''
# gdal_translate SENT--roi_x_y=INEL2_L1C:S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:10m:EPSG_32610   --config GDAL_NUM_THREADS 32 -of ENVI -ot Float32 out.bin
