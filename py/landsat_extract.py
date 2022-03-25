'''20210324 extract landsat from .tar to ENVI bsq/float'''
from misc import *
lines = os.popen('ls -1 *.tar').readlines()

for line in lines:
    f = line.strip()
    d = f[:-4] # directory

    if f[0] != 'L':
        err('expected L prefix for landsat')
    if not exist(d):
        os.mkdir(d)
    
    N = int(f[3])
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

    # https://docs.sentinel-hub.com/api/latest/data/landsat-etm/
    C7 = {'B1': (450. + 520.) / 2.,
          'B2': (520. + 600.) / 2.,
          'B3': (630. + 690.) / 2.,
          'B4': (770. + 900.) / 2.,
          'B5': (1550. + 1750.) / 2.,
          'B6': (10400. + 12500.) / 2.,
          'B7': (2090. + 2350.) / 2.,
          'B8': (520. + 900.) / 2.,
          'TRAD': (10400. + 12500.) / 2.}

    # https://docs.sentinel-hub.com/api/latest/data/landsat-8/
    C8 = {'B1': 443.,
          'B2': 482.,
          'B3': 561.5,
          'B4': 654.5,
          'B5': 865.,
          'B6': 1608.5,
          'B7': 2200.5,
          'B8': 589.5,
          'B9': 1373.5,
          'B10': 10895.,
          'B11': 12005.,
          'TRAD': (10895. + 12005.) / 2.}
    
    for i in x:
        w = i.split(sep)[-1].split('_')[-1].split('.')[0]

        CF = None
        if N == 7:
            CF = C7[w]
        if N == 8:
            CF = C8[w]

        print('* ', CF, w, i)
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
