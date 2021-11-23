'''20211121 same as sentinel2_extract_stack_all.py except this one doesn't extract from zip. 

Assumption:
    we've created L2 data with sen2cor

A) extract Sentinel2, B) resample to 10m c) prefix bandnames with dates..
   D) stack everything!
   E) worry about masks later

Need xml reader? such as:
https://docs.python.org/3/library/xml.etree.elementtree.html


MAKE SURE WE SORTED BY TIME AND FREQUENCY !!!'''
import os
import sys
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep
ehc = pd + 'envi_header_cleanup.py' # envi header cleanup command.. makes file open in "imv"

import multiprocessing as mp
sep = os.path.sep

def parfor(my_function, my_inputs, n_thread=mp.cpu_count()): # evaluate a function in parallel, collect the results
    pool = mp.Pool(n_thread)
    result = pool.map(my_function, my_inputs)
    return(result)

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print([c]); a = os.system(c)
    return a
    # if a != 0: err("failed to run: " + str(c))

# extract = pd + "sentinel2_extract.py" # command to extract a zip
raster_files = [] # these will be the final rasters to concatenate

'''before processing, sort zip files by date. Note, they would be already except the prefix
varies with S2A / S2B'''

safes, files = [], [x.strip() for x in os.popen('ls -1').readlines()]
for f in files:
    if f[-5:] == '.SAFE':
        w = f.split('_'); # print(w)
        if w[1] == 'MSIL2A':
                safes.append(w)

# sort on w[2]
srt = [[w[2], w] for w in safes]
srt.sort()
safes = [w[1] for w in srt]
safes = ['_'.join(s) for s in safes]

for safe in safes: 
    cmds = []; #print(safe)
    ''' ls -1 *.bin
     SENTINEL2_L2A_10m_EPSG_32610.bin
     SENTINEL2_L2A_20m_EPSG_32610.bin
     SENTINEL2_L2A_60m_EPSG_32610.bin
     SENTINEL2_L2A_TCI_EPSG_32610.bin '''
    # have to make those files!
    gdfn = safe + sep + 'MTD_MSIL2A.xml'
    if not exists(gdfn):
        err("expected file: " + gdfn)
    
    print('detect:')
    for line in [x.strip() for x
                 in os.popen('gdalinfo ' +
                             gdfn +
                             ' | grep SUBDATA').readlines()]:
        if len(line.split('.xml')) > 1:
            df = safe.split(sep)[-1]
            dfw = line.split(df)
            term = dfw[-1].strip(sep).split(':')[0]
            iden = dfw[0].split('=')[1].split(':')[0]
            ds = iden + ':' + df + dfw[1]
            of = (df + dfw[1]).replace(term, iden).replace(':', '_') + '.bin'
            cmd = ' '.join(['gdal_translate',
                            ds,
                            '--config GDAL_NUM_THREADS 8',
                            '-of ENVI',
                            '-ot Float32',
                            of])
            hfn = of[:-4] + '.hdr'; print('  ' + hfn)
            if not exists(of):
                cmd += '; ' + (' '.join(['python3',
                                         ehc,
                                         hfn]))
                cmds.append(cmd)
    parfor(run, cmds, 4)  # 4 hw mem channels a good guess?
    bins = [x.strip() for x in os.popen("ls -1 " +
                                        safe +
                                        sep +
                                        "*m_EPSG_*.bin").readlines()]
    # don't pull the TCI true colour image?  Already covered in 10m

    print('extract:')
    for b in bins:
        print('  ' + b)

    if len(bins) != 3:
        continue
        err('unexpected number of bin files (expected 3): ' +
            str('\n'.join(bins)))

    m10, m20, m60 = bins
    print('10m:', m10); print('20m:', m20); print('60m:', m60)

    # names for files resampled to 10m
    m20r, m60r = (m20[:-4] + '_10m.bin',
                  m60[:-4] + '_10m.bin')

    def resample(pars): # resample src onto ref, w output file dst
        src, ref, dst = pars
        cmd = ['python3 ' +
                pd + 'raster_project_onto.py',
                src, # source image
                ref, # project onto
                dst] # result image
        #if not exists(dst): 
        run(' '.join(cmd))
        return 0

    a = parfor(resample,
              [[m20, m10, m20r], # resample the 20m
              [m60, m10, m60r]],
              2) # resample the 60m

    sfn = (safe + sep + m10.split(sep)[-1].replace("_10m", "")[:-4]
            + '_10m.bin')  # name of stacked file
    print(sfn)
    cmd = ['cat', # cat bands together, remember to cat the header files after
            m10,
            m20r,
            m60r,
            '>',
            sfn]
    run(' '.join(cmd))  # things got wierd by not recomputing this

    # add a date prefix
    dp = '"' + safe.split('T')[0].strip().split('_')[-1].strip()
    dp10, dp20, dp60 = (dp + ' 10m: "',
                        dp + ' 20m: "',
                        dp + ' 60m: "')
    shn = sfn[:-4] + '.hdr' # header file name for stack
    cmd = ['python3', # envi_header_cat.py like an rpn, first thing goes on back
           pd + 'envi_header_cat.py',
           m20r[:-4] + '.hdr',
           m10[:-4] + '.hdr',
           shn,
           dp20,
           dp10]
    run(' '.join(cmd))

    cmd = ['python3',
            pd + 'envi_header_cat.py',
            m60r[:-4] + '.hdr',
            shn[:-4] + '.hdr',
            shn,
            dp60]
    run(' '.join(cmd))

    # raster_files.append(sfn)
    mod = open(shn).read().replace('central wavelength ',
                                   '').replace(' nm,',
                                               'nm,').replace(' nm}', 'nm}').encode()
    open(shn, 'wb').write(mod)
    print('+w', shn)
 
    cmd = ['python3',
           pd + 'raster_reorder_increasing_nm.py',
           sfn]
    run(' '.join(cmd))
    
    raster_files.append(sfn) # reorder is "in-place"... so not + '_reorder.bin')
    # sys.exit(1) # would turn this on to debug one frame

    # reorder bands should go here
    # for s in safes:
    #     print(s)

# cat the bin files together, combining headers
cmd = ['python3', pd + 'raster_stack.py']
cmd = cmd + raster_files + ['raster.bin']
a = os.system(' '.join(cmd))
