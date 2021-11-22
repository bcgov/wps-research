'''20211121 same as sentinel2_extract_stack_all.py except this one doesn't extract from zip. 

Assumption:
    we've created L2 data with sen2cor

A) extract Sentinel2, B) resample to 10m c) prefix bandnames with dates..
   D) stack everything!
   E) worry about masks later

Need xml reader? such as:
https://docs.python.org/3/library/xml.etree.elementtree.html '''
import os
import sys
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c); a = os.system(c)
    if a != 0: err("failed to run: " + str(c))

# extract = pd + "sentinel2_extract.py" # command to extract a zip
raster_files = [] # these will be the final rasters to concatenate

'''before processing, sort zip files by date. Note, they would be already except the prefix
varies with S2A / S2B'''

safes, files = [], [x.strip() for x in os.popen('ls -1').readlines()]
for f in files:
    if f[-5:] == '.SAFE':
        w = f.split('_'); print(w)
        if w[1] == 'MSIL2A':
                safes.append(w)

# sort on w[2]
srt = [[w[2], w] for w in safes]
srt.sort()
safes = [w[1] for w in srt]
for s in safes:
    print(s)
sys.exit(1)

x = []
for z in zips:
    w = z.split('_')
    if w[0][:2] != "S2":
        x.append(z) # don't have a rule for sorting, if not S2!
    else:
        x.append([w[2:], z])
''' e.g. [['20190210T200551', 'N0211', 'R128', 'T09VUE', '20190210T222054.zip'],
           'S2A_MSIL2A_20190210T200551_N0211_R128_T09VUE_20190210T222054.zip'] '''
x.sort()
zips = [i[1] for i in x]  # finally, these files should be sorted by date..

for z in zips:
    safe = z[:-4] + ".SAFE" # extracted location..

    print(safe)
    if not os.path.exists(safe):
        cmd = "python3 " + extract + " " + z + (" no_stomp=True" if no_stomp else "")
        print(cmd)
        a = os.system(cmd)

    ''' ls -1 *.bin
        SENTINEL2_L2A_10m_EPSG_32610.bin
        SENTINEL2_L2A_20m_EPSG_32610.bin
        SENTINEL2_L2A_60m_EPSG_32610.bin
        SENTINEL2_L2A_TCI_EPSG_32610.bin'''
    bins = [x.strip() for x in os.popen("ls -1 " + safe + os.path.sep + "*m_EPSG_*.bin").readlines()]
    # don't pull the TCI true colour image. Already covered in 10m

    if len(bins) != 3:
        err("unexpected number of bin files (expected 3): " + str('\n'.join(bins)))

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

        if not exists(dst):
            run(' '.join(cmd))

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

    if not exists(sfn):
        run(' '.join(cmd))

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
    run(cmd)
    raster_files.append(sfn)

# cat the bin files together, combining headers
cmd = ['python3', pd + 'raster_stack.py']
cmd = cmd + raster_files + ['raster.bin']
a = os.system(' '.join(cmd))
