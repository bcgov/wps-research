'''20230526 note: should have target CRS in EPSG format as a parameter!

20230515 add swir-only option. NOte: need to modify the band extraction to select the relevant bands only.
20221026 Adapted from sentinel2_stack_all.py.
(*) This version creates a 20m product. 
(*) This version will (in progress) replace atmospheric effects with NAN

20211121 same as sentinel2_extract_stack_all.py EXCEPT:
    this one doesn't extract from zip!!!!!!!! 

WHY?????????????

Assumption:
    we've created L2 data with sen2cor

   A) extract Sentinel2,
   B) resample to 20m (not 10m) 
   C) prefix bandnames with dates..
   D) stack everything!
   E) SORT BY TIME AND FREQUENCY
   F) APPLY cloud threshold (7%)
   G) Apply scene classification (below) NOT IMPLEMENTED
   H) Convert all-zero areas to NaN

From https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
=================================================================
Figure 3: Scene Classification Values Label     Classification
0       NO_DATA
1       SATURATED_OR_DEFECTIVE
2       DARK_AREA_PIXELS
3       CLOUD_SHADOWS
4       VEGETATION
5       NOT_VEGETATED
6       WATER
7       UNCLASSIFIED
8       CLOUD_MEDIUM_PROBABILITY
9       CLOUD_HIGH_PROBABILITY
10      THIN_CIRRUS
11      SNOW
=================================================================

Need xml reader? such as:
https://docs.python.org/3/library/xml.etree.elementtree.html
'''
import os
import sys
import shutil
import multiprocessing as mp
from envi import envi_header_modify
from misc import args, sep, exists, parfor, err, run, pd, band_names, read_hdr


print(args)
swir_only = len(args) > 2

N_THREAD = mp.cpu_count()
ehc = pd + 'envi_header_cleanup.py' # envi header cleanup command.. makes file open in "imv" 
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

for safe in safes:   # S2A_MSIL2A_20170804T190921_N0205_R056_T10UFB_20170804T191650
    print(safe)
    w= safe.split('_')
    TILE_ID = w[5]
    DATE = w[2].split('T')[0]
    sfn = safe.split('.')[0] + '.bin' # TILE_ID + '_' + DATE + '.bin' # output file name
    if exists(sfn):
        print("+r", sfn)
        continue
    #if exists(sfn):
    #    run("raster_zero_to_nan " + sfn)
    #    continue
    #print("TILE_ID", TILE_ID)
    #print("DATE", DATE)
    #sys.exit(1)
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
                            '--config GDAL_NUM_THREADS ' + str(N_THREAD),
                            '-of ENVI',
                            '-ot Float32',
                            of])
            hfn = of[:-4] + '.hdr'; print('  ' + hfn)
            if not exists(of):
                cmd += '; ' + (' '.join(['python3',
                                         ehc,
                                         hfn]))
                if swir_only and (len(cmd.split(':10m:')) > 1 or len(cmd.split(':TCI:')) > 1):
                    pass 
                else:
                    cmds.append(cmd)
                    print(cmd)

    parfor(run, cmds, 4)  # 4 hw mem channels a good guess?
    bins = [x.strip() for x in os.popen("ls -1 " +
                                        safe +
                                        sep +
                                        "*m_EPSG_*.bin").readlines()]
    # don't pull the TCI true colour image?  Already covered in 10m

    print('extract:')
    for b in bins:
        print('* ' + b)

    if len(bins) < 3 and not swir_only:
        print("unexpected number of bin files (expected 3)")
        continue
        err('unexpected number of bin files (expected 3): ' +
            str('\n'.join(bins)))

    m10 = bins[0]  # 10m doesn't get resampled so should be first
    # m10, m20, m60 = bins
    m20, m60 = m10.replace('_10m_', '_20m_'),\
               m10.replace('_10m_', '_60m_')
    if swir_only:
        m20 = bins[0]
        m60 = bins[1]
        del m10
    else:
        print('10m:', m10)
    print('20m:', m20); print('60m:', m60)
    
    if not swir_only:
        for m_i in [m10, m20, m60]:
            if not exists(m_i):
                err('file not found:' + m_i)

    # names for files resampled to 10m
    try:
        m10r = m10[:-4] + '_20m.bin' 
    except:
        pass
    m60r = m60[:-4] + '_20m.bin'

    def resample(pars): # resample src onto ref, w output file dst
        src, ref, dst = pars
        cmd = ['python3 ' +
                pd + 'raster_project_onto.py',
                src, # source image
                ref, # project onto
                dst] # result image
        print(cmd)
        if not exists(dst): 
            run(' '.join(cmd))
        return 0
    
    if not swir_only:
        a = parfor(resample,
                   [[m10, m20, m10r], # resample the 20m
                    [m60, m20, m60r]], 2) # resample the 60m
    else:
        resample([m60, m20, m60r])

    #try:
    #    sfn = (safe + sep + m10.split(sep)[-1].replace("_10m", "")[:-4]
    #            + '_20m.bin')  # name of stacked file
    #except:
    #    sfn = (safe + sep + m20.split(sep)[-1].replace("_20m", "")[:-4]
    #            + '_20m.bin')
    #
    #sfn = TILE_ID + '_' + DATE + '.bin'

    print("sfn", sfn)
    cmd = ['cat', # cat bands together, remember to cat the header files after
            m10r if not swir_only else '',
            m20,
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
           m20[:-4] + '.hdr',
           m10r[:-4] + '.hdr' if not swir_only else "",
           shn,
           dp20,
           dp10]
    if not swir_only:
        run(' '.join(cmd))
    else:
        print(m20[:-4] + '.hdr', "-->", shn)
        shutil.copyfile(m20[:-4] + '.hdr', shn)
        run('fh ' + shn)
        band_names_20 = [(dp20 + x) for x in band_names(shn)]
        print(band_names_20)
        band_names_20 = ['"' + x.replace('"', '').replace('(', '[').replace(')',']') + '"' for x in band_names_20]
        samples, lines, bands = read_hdr(shn)
        #     err('envi_header_modify.py [.hdr file to modify] [nrow] [ncol] [nband] [band 1 name]... [band n name]')
        
        envi_header_modify(['python3 ' + pd + sep + 'envi_header_modify.py', 
                            shn,
                            str(lines), 
                            str(samples), 
                            str(bands)] + [z.replace('"', '') for z in band_names_20])
        # print([cmd])
        # run(cmd)

    cmd = ['python3',
            pd + 'envi_header_cat.py',
            m60r[:-4] + '.hdr',
            shn[:-4] + '.hdr',
            shn,
            dp60]
    run(' '.join(cmd))

    # raster_files.append(sfn)
    mod = open(shn).read().replace('central wavelength ', '')
    mod = mod.replace(' nm,', 'nm,')
    mod = mod.replace(' nm}', 'nm}').encode()
    open(shn, 'wb').write(mod)
    print('+w', shn)
 
    if swir_only:
        run('sentinel2_swir_subselect ' + sfn)
        shutil.copyfile(sfn + '_swir.bin', sfn)
        shutil.copyfile(sfn + '_swir.hdr', shn)
        run('fh ' + shn)

    cmd = ['python3',
           pd + 'raster_reorder_increasing_nm.py',
           sfn]
    run(' '.join(cmd))
    raster_files.append(sfn) # reorder is "in-place"... so not + '_reorder.bin')
    # sys.exit(1) # would turn this on to debug one frame
    # should check if "sfn" exists before doing anything on this folder.

    if False:
        # produce cloud mask
        run("sentinel2_cloud.py " + safe)

        # Filter by the cloud mask: mark cloud areas as NAN (GEQ 7% cloud probability)
        run("cloud_nan.exe " + sfn + " " + safe + os.path.sep + "cloud_20m.bin")

    # cleanup extra files
    run("rm -v " + safe + os.path.sep + "*.bin")

    # set no-data areas to NAN:
    run("raster_zero_to_nan " + sfn)

    if swir_only:  # flip the band order for SWIR data
        run("raster_reorder_increasing_nm.py " + sfn + " 1")
    
if len(args) > 1 and len(args) < 3:
    # cat the bin files together, combining headers
    cmd = ['python3', pd + 'raster_stack.py']
    cmd = cmd + raster_files + ['raster.bin']
    run(' '.join(cmd))
