from misc import *
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep
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

df = fn[:-4] + '.SAFE'  # extracted data folder
if not os.path.exists(fn) and not os.path.exists(df):
    err('could not find input file')

if not os.path.exists(df):
    if no_stomp == False:
        a = os.system('unzip ' + fn)
    else:
        a = os.system('mkdir -p ' + df)  # special no_stomp mode!! needed for using google cloud drive script
        a = os.system('unzip -d ' + df + ' ' + fn)

    import time
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
bins = []
found_line = True
for line in xml:
    found_line = False
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
            bins.append(of)
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

print(bins)
bins = bins[0:3]
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

# resample everything to 10m:
if True:
    m10, m20, m60 = bins
    print("  10m:", m10)
    print("  20m:", m20)
    print("  60m:", m60)

    #for b in bins:
    #    print('  ' + b) # print('  ' + b.split(sep)[-1])

    # names for files resampled to 10m
    m20r, m60r = m20[:-4] + '_10m.bin', m60[:-4] + '_10m.bin'

    def resample(src, ref, dst): # resample src onto ref, w output file dst
        cmd = ['python3 ' + pd + 'raster_project_onto.py',
           src, # source image
           ref, # project onto
           dst] # result image
        cmd = ' '.join(cmd)
        print(cmd)
        if not exists(dst):
            run(cmd)

    resample(m20, m10, m20r) # resample the 20m
    resample(m60, m10, m60r) # resample the 60m

    # now do the stacking..
    print(m10)

    sfn = safe + sep + m10.split(sep)[-1].replace("_10m", "")[:-4] + '_10m.bin'  # stacked file name..
    print(sfn)

    cmd = ['cat',
            m10,
            m20r,
            m60r,
            '>',
            sfn] # cat bands together, don't forget to "cat" the header files after..

    cmd = ' '.join(cmd)
    print(cmd)
    if not exists(sfn):
        run(cmd)

    # add a date prefix
    dp = '"' + safe.split('T')[0].strip().split('_')[-1].strip()
    dp10 = dp + ' 10m: "'
    dp20 = dp + ' 20m: "'
    dp60 = dp + ' 60m: "'

    # now "cat" the header files together
    shn = sfn[:-4] + '.hdr' # header file name for stack

    cmd = ['python3', # envi_header_cat.py is almost like a reverse-polish notation. Have to put the "first thing" on the back..
           pd + 'envi_header_cat.py',
           m20r[:-4] + '.hdr',
           m10[:-4] + '.hdr',
           shn,
           dp20,
           dp10]

    cmd = ' '.join(cmd)
    run(cmd)

    cmd = ['python3',
            pd + 'envi_header_cat.py',
            m60r[:-4] + '.hdr',
            shn[:-4] + '.hdr',
            shn,
            dp60]

    cmd = ' '.join(cmd)
    stack_fn = os.popen(cmd).readlines()[1].strip().split()[-1][:-4] + '.bin'
    print("stack resampled to 10m:")
    print("\t" + stack_fn)

    # clean up the header file a bit! Readability, and the nm part expected format for other progs
    d = open(shn).read().strip()
    open(shn, "wb").write((d.replace(' central wavelength', '').replace(' nm', 'nm')).encode())

    run(['python3',
         pd + 'raster_reorder_increasing_nm.py',
         stack_fn])
