'''20210324 extract landsat from .tar to ENVI bsq/float'''
from misc import *
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
    print(d)
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
    
    band_names = []
    for i in x:
        w = i.split(sep)[-1].split('_')[-1].split('.')[0]
    
        CF = None
        if N == 7:
            CF = C7[w]
        if N == 8:
            CF = C8[w]

        print('* ', CF, w, i)
        band_names.append(w + ' ' + str(CF) + 'nm')

    print(band_names)
    fn = d + sep + d + '.bin'
    print(fn)
    if not exists(fn):
        run(['gdal_merge.py -of ENVI -ot Float32 -o',
            fn,
            ' -seperate',
            ' '.join(x)])
        run(['python3 ' + pd + 'envi_header_cleanup.py',
            fn[:-4] + '.hdr'])

        # need to add date, frame id, and wavelength info
    sys.exit(1)
