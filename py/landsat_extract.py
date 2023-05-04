'''20210324 extract landsat from .tar to ENVI bsq/float

20220514 update for L7 (Surf refl), and add L5 (Surf refl)'''
from misc import *
fire_mapping = len(args) > 2  # add optional arg to enable fire mapping
lines = os.popen('ls -1 *.tar').readlines()

print(fire_mapping)
print(lines)

for line in lines:
    f = line.strip()
    d, N = f[:-4], int(f[3]) # directory
    print('[' + d + ']')

    if f[0] != 'L':
        err('expected L prefix for landsat')
    if not exist(d):
        os.mkdir(d)  # folder to extract into
    if N not in [5, 7, 8, 9]:
        err('expected Landsat 5, 7, 8 or 9')

    def find_bands():
        # will need to revisit exactly which bands get pulled
        x = os.popen('ls -1 ' + d + sep + '*_B*.TIF').readlines()
        x += os.popen('ls -1 ' + d + sep + '*ST_TRAD.TIF').readlines()
        if len(x) == 0:
            x = os.popen('ls -1 ' + d + sep + '*_b*.tif').readlines()
        if N == 8:
            x += os.popen('ls -1 ' + d + sep + '*_B10.TIF').readlines()
        x = [i.strip() for i in x]
        x = list(set(x))
        return x
    x = find_bands()

    if len(x) < 7:
        run(['tar xf', f, '-C', d])
    
    x = find_bands()  # display avail. bands
    for i in x:
        print('\t', i.strip().split(sep)[-1])
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

    C7_2 = {'band1':   av(  450.,   520.),  # docs.sentinel-hub.com/api/latest/data/landsat-etm/
           'band2':   av(  520.,   600.),
           'band3':   av(  630.,   690.),
           'band4':   av(  770.,   900.),
           'band5':   av( 1550.,  1750.),
           'band6':   av(10400., 12500.),
           'band7':   av( 2090.,  2350.),
           'band8':   av(  520.,   900.)}

    S7 = {'B8': 15}  # resolution (m)
    for i in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'TRAD']:
        S7[i] = 30
    
    S7_2 = {}  # resolution (m)
    for i in ['band1', 'band2', 'band3', 'band4', 'band5', 'band6', 'band7']:
        S7_2[i] = 30


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
          'B11':     12005.}
    if N != 9:
        C8['TRAD'] = av(10895., 12005.) 

    S8 = {'B8': 15} # resolution(m)

    for i in (['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
               'B8', 'B9', 'B10', 'B11'] +
              (['TRAD'] if N != 9 else [])):  # , 'TRAD']:
        S8[i] = 30

    # get UTC timestamp information
    txt_f = os.popen('ls -1 ' + d + sep + '*MTL.txt').readlines()
    if len(txt_f) > 1:
        err('expected only one text file')
    t_s = os.popen('grep SCENE_CENTER_TIME ' + txt_f[0].strip())
    t_s = t_s.read().strip().split('=')[-1].strip().strip('"')
    t_s = t_s.split('.')[0].replace(':', '')
    band_names = []

    for i in x:
        print(f, i)
        w = f.split('_')
        # Underscore delimited vs. e..g LE070480202006060401T1-SC20220615003503 format
        ds = w[3] if len(w) > 2 else f[10:18]
        print(ds, w, f, i)
        w = i.split(sep)[-1].split('_')[-1].split('.')[0]
        print('\t', w)
        CF, R = None, None
        if N == 7 or N == 5:
            try:
                CF, R = C7[w], S7[w]
            except:
                CF = C7_2[w]
                R = S7_2[w]
        if N == 8 or N == 9:
            try:
                CF, R = C8[w], S8[w]
            except:
                continue

        CF, R = str(CF), str(R)
        CF = CF[:-2] if CF[-2:] == '.0' else CF

        # print('* ', CF, w, i)
        bn = ' '.join([ds + t_s,
                       R + 'm:',
                       w,
                       CF + 'nm'])
        band_names.append(bn)
    print('band names')
    for b in band_names:
        print('  ',b)
    # sys.exit(1)
    fn = d + sep + d + '.bin'
    hfn = fn[:-4] + '.hdr' # print(fn)
    if not exists(fn):
        # merge the bands at resolution of Band #1
        cmd = ' '.join(['gdal_merge.py -of ENVI -ot Float32 -o',
                       fn,
                       '-seperate',
                       ' '.join(x)])
        run(cmd)

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

    # sys.exit(1)
    # fire mapping section
    if fire_mapping:
        f2 = fn + '_spectral_interp.bin'
        f2h = f2[:-4] + '.hdr'
        if not exists(f2):
            # simulate Sentinel2 Level2, from Landsat 7/8/9 Level2 
            run(['python3 ' + pd + 'raster_simulate_s2.py',
                fn])
            run(['python3 ' + pd + 'envi_header_copy_mapinfo.py',
                 hfn,
                 f2h])
            run(['python3 ' + pd + 'envi_update_band_names.py',
                 hfn,
                 f2h])
        else:
            print('+r', f2)
        ff = f2 + '_active.bin'
        ffh = ff[:-4] + '.hdr'
        if not exists(ff):
            # run fire detection filter
            run([cd + 'sentinel2_active.exe',
                f2])
            run(['python3 ' + pd + 'envi_header_copy_mapinfo.py',
                 hfn,
                 ffh])
        else:
            print('+r', ff)
        #sys.exit(1)
