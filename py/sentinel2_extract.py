'''Can use this method to open Level1 format Sentinel-2 data downloaded either from:
    *** ESA Copernicus ***,
     or from
    *** Google Cloud Platform ***

Two cases:
* after running gcp/fix_s2.py and zipping the folder
* if there is no zip file (having downloaded from GCP

NOTE: this script extracts products with different resolutions separately. They are NOT resampled into a common stack! 
NOTE: no_stomp parameter avoids unzipping the zip file (again)'''
import time
from misc import *
ehc = pd + 'envi_header_cleanup.py' # envi header cleanup command.. makes file open in "imv"

if len(args) < 2:
    err('python3 extract_sentinel2.py [input sentinel2 zip file name] ' +
        '# [optional parameter: no_stomp=True]' +
        ' #the optional parameter avoids creating a .SAFE folder if' +
        ' there already is one')

no_stomp = False
if len(args) > 2:
    if args[2] == 'no_stomp' or args[2] == 'no_stomp=True':
        no_stomp = True

# check gdal version
info = os.popen("gdalinfo --version").read().strip().split(',')
info = info[0].split()[1].replace('.', '')
if int(info) < 223:
    err('GDAL version 2.2.3 or higher required')

fn = os.path.abspath(args[1])  # don't want sep at end if folder
# print('input data target:', fn)
df = fn[:-4] + '.SAFE'  # extracted data folder
if fn[-4:] != '.zip':
    if fn[-4:] == 'SAFE':
        no_stomp = True
        df = fn # extracted data folder
    else:
        err('expected input should be .zip (file) or .SAFE (folder)')

# check indicated input (.zip or .SAFE) exists
if not exist(fn):
    err('could not find input target:' + fn)

# check if extracted folder is there
if not os.path.exists(df):
    if no_stomp == False:
        a = os.system('unzip ' + fn)
    else:
        a = os.system('mkdir -p ' + df)  # special no_stomp mode!! needed for using google cloud drive script
        a = os.system('unzip -d ' + df + ' ' + fn)
    time.sleep(1.)  # not sure if this is needed

if not os.path.exists(df):
    err('failed to unzip: cant find folder: ' + df)

gdfn = df + sep + 'MTD_MSIL1C.xml' if no_stomp else fn
cmd = 'gdalinfo ' + gdfn + ' | grep SUBDATA'  # try gdalinfo
print(cmd)
xml = os.popen(cmd).readlines()

cmds = []  # commands to run after this section
data_files = []
found_line = False
for line in xml:
    line = line.strip()
    if len(line.split('.xml')) > 1:
        try:
            df = df.split(os.path.sep)[-1]
            safe = df # .SAFE directory
            dfw = line.split(df)
            terminator = dfw[-1].strip(os.path.sep).split(':')[0]
            ident = dfw[0].split('=')[1].split(':')[0]
            ds = ident + ':' + df + dfw[1]
            of = (df + dfw[1]).replace(terminator, ident).replace(':', '_') + '.bin'

            cmd = ' '.join(['gdal_translate',  # gdal to translate data formats
                            ds, '--config GDAL_NUM_THREADS 8',  # use 8 threads
                            '-of ENVI',   # raw / ENVI binary format
                            '-ot Float32',  # 32-bit float format like PolSARPro
                            of])  # output .SAFE folder
            print('\t' + cmd)
            data_files.append(cmd)
            if not os.path.exists(of):  # don't extract if we did already!
                cmds.append(cmd)

            hfn = of[:-4] + '.hdr'
            cmd = ' '.join(['python3', ehc, hfn]) # of[:-4] + '.hdr']
            print('\t' + cmd)
            if not os.path.exists(hfn):
                 cmds.append(cmd)

            found_line = True  # what line were we looking for?
        except Exception:
            pass

if not found_line:
    if no_stomp:
        err('Downloaded from GCP? check inputs')

for cmd in cmds:
    run(cmd)

'''e.g. outputs:
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:10m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:20m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:60m:EPSG_32610
S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:TCI:EPSG_32610
'''
# gdal_translate SENT--roi_x_y=INEL2_L1C:S2A_MSIL1C_20191129T190741_N0208_R013_T10UFB_20191129T204451.SAFE/MTD_MSIL1C.xml:10m:EPSG_32610   --config GDAL_NUM_THREADS 32 -of ENVI -ot Float32 out.bin
