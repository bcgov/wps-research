'''20210324 extract landsat from .tar to ENVI bsq/float'''
from misc import *
lines = os.popen('ls -1 *.tar').readlines()

for line in lines:
    f = line.strip()
    d = f[:-4] # directory
    if not exist(d):
        os.mkdir(d)
    
    x = os.popen('ls -1 ' + d + sep + '*SR*.TIF').readlines()
    x += os.popen('ls -1 ' + d + sep + '*ST*.TIF').readlines()
    x = [i.strip() for i in x]
    print(d)
    for i in x:
        print('\t', i.strip())

    if len(x) < 7:
        run(['tar xvf', f, '-C', d])

    fn = d + sep + d + '.bin'
    print(fn)
    if not exists(fn):
        run(['gdal_merge.py -of ENVI -ot Float32 -o',
            fn,
            ' -seperate',
            ' '.join(x)])
        run(['python3 ' + pd + 'envi_header_cleanup.py',
            fn[:-4] + '.hdr'])
