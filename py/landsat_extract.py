'''20210324 extract landsat from .tar to ENVI bsq/float'''
from misc import *
fire_mapping = len(args) > 1  # add optional arg to enable fire mapping
lines = os.popen('ls -1 *.tar').readlines()

for line in lines:
    f = line.strip()
    d, N = f[:-4], int(f[3]) # directory

    if f[0] != 'L':
        err('expected L prefix for landsat')
    if not exist(d):
        os.mkdir(d)  # folder to extract into
    if N not in [7, 8]:
        err('expected Landsat 7 or 8')

    def find_bands():
        # will need to revisit exactly which bands get pulled
        x = os.popen('ls -1 ' + d + sep + '*SR_B*.TIF').readlines()
        x += os.popen('ls -1 ' + d + sep + '*ST_TRAD.TIF').readlines()
        x += os.popen('ls -1 ' + d + sep + '*ST_B10.TIF').readlines()
        x = [i.strip() for i in x]
        return x
    x = find_bands()

    if len(x) < 7:
        run(['tar xvf', f, '-C', d])
    x = find_bands()
    
    print(d)  # display this file's avail. bands
    for i in x:
        print('\t', i.strip())
        # print(os.popen('gdalinfo ' + i + ' | grep "Pixel Size"').read())
        # print(os.popen('gdalinfo ' + i + ' | grep "wave"').read())

    def av(a, b):
        return (a + b) / 2.

    C7 = {'B1':   av(  450.,   520.),  # docs.sentinel-hub.com/api/latest/data/landsat-etm/
          'B2':   av(  520.,   600.),
          'B3':   av(  630.,   690.),
          'B4':   av(  770.,   900.),
          'B5':   av( 1550.,  1750.),
          'B6':   av(10400., 12500.),
          'B7':   av( 2090.,  2350.),
          'B8':   av(  520.,   900.),
          'TRAD': av(10400., 12500.)}
    S7 = {'B1': 30,  # resolution (m)
          'B2': 30,
          'B3': 30,
          'B4': 30,
          'B5': 30,
          'B6': 30,
          'B7': 30,
          'B8': 15,
          'TRAD': 30}
    
    C8 = {'B1':        443.,  # docs.sentinel-hub.com/api/latest/data/landsat-8/
          'B2':        482.,
          'B3':        561.5,
          'B4':        654.5,
          'B5':        865.,
          'B6':       1608.5,
          'B7':       2200.5,
          'B8':        589.5,
          'B9':       1373.5,
          'B10':     10895.,
          'B11':     12005.,
          'TRAD': av(10895.,
                     12005.)}
    S8 = {'B1': 30,  # resolution (m)
          'B2': 30,
          'B3': 30,
          'B4': 30,
          'B5': 30,
          'B6': 30,
          'B7': 30,
          'B8': 15,
          'B9':  30,
          'B10': 30,
          'B11': 30,
          'TRAD': 30}

    band_names = []
    for i in x:
        ds = f.split('_')[3]
        w = i.split(sep)[-1].split('_')[-1].split('.')[0]

        CF, R = None, None
        if N == 7:
            CF, R = C7[w], S7[w]
        if N == 8:
            CF, R = C8[w], S8[w]
        
        CF, R = str(CF), str(R)
        CF = CF[:-2] if CF[-2:] == '.0' else CF

        # print('* ', CF, w, i)
        bn = ' '.join([ds,
                       R + 'm:',
                       w,
                       CF + 'nm'])
        band_names.append(bn)
    # print(band_names)
    fn = d + sep + d + '.bin'
    hfn = fn[:-4] + '.hdr' # print(fn)
    if not exists(fn):
        # merge the bands at resolution of Band #1
        run(['gdal_merge.py -of ENVI -ot Float32 -o',
            fn,
            ' -seperate',
            ' '.join(x)])

        # cleanup header and update band names w wavelengths
        run(['python3 ' + pd + 'envi_header_cleanup.py',
             hfn])
        samples, lines, bands = read_hdr(hfn)
        run(['python3 ' + pd + 'envi_header_modify.py',
             hfn,
             lines,
             samples,
             bands] +
            ['"' + i + '"' for i in band_names])
        run(['python3 ' + pd + 'raster_reorder_increasing_nm.py',
             fn])

    # fire mapping section
    if fire_mapping:
        f2 = fn + '_spectral_interp.bin'
        if not exists(f2):
            # simulate Sentinel2 Level2, from Landsat 7/8/9 Level2 
            run(['python3 ' + pd + 'raster_simulate_s2.py',
                fn])
            run(['python3 ' + pd + 'envi_header_copy_mapinfo.py',
                 hfn,
                 f2[:-4] + '.hdr'])
        else:
            print('+r', f2)
        ff = f2 + '_active.bin'
        
        if not exists(ff):
            # run fire detection filter
            run([cd + 'sentinel2_active.exe',
                f2])
            run(['python3 ' + pd + 'envi_header_copy_mapinfo.py',
                 hfn,
                 ff[:-4] + '.hdr'])
        else:
            print('+r', ff)
    #sys.exit(1)
