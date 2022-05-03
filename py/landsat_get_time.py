'''get timestamps from landsat and sentinel2
and include those in them in result filenames

run this after:
    raster_accumulate.exe
and 
    raster_ts_dedup.exe
'''

FOOT_H = 'footprint3.hdr'
from misc import *
args = ['', 'landsat.bin_accumulate.bin_dedup.hdr']
bn = band_names(hdr_fn(args[1]))

ci = 1
for b in bn:
    w = b.strip().strip('.').strip(sep).split(sep)
    print(w)
    base, TS, TYPE = '', '', ''
    if w[1][0] == 'L':
        TYPE = 'landsats'
        base = sep.join(w[:2])
        txt = base + sep + w[1] + '_MTL.txt'
        if not exist(txt):
            err('file not found')
        d, t = None, None
        lines = [x.strip() for x in open(txt).readlines()]
        for line in lines:
            w = [x.strip() for x in line.split('=')]
            if w[0] == 'DATE_ACQUIRED':
                d = w[1]
            if w[0] == 'SCENE_CENTER_TIME':
                t = w[1]
        yyyy, mm, dd = d.split('-')
        h, m, s = t.strip('"').split('.')[0].split(':')
        TS = '-'.join([yyyy, mm, dd, h, m, s])
    elif w[1][0] == 'S':
        # sentinel2
        TYPE = 'sentinel'
        d,t  = w[1].split('_')[2].split('T')
        yyyy, mm, dd = d[0:4], d[4:6], d[6:8]
        h, m, s = t[:2], t[2:4], t[4:6]
        TS = '-'.join([yyyy, mm, dd, h, m, s])
    else:
        err('unrecognized')
    x = [int(i) for i in TS.split('-')]
    L = utc_to_pst(x[0], x[1], x[2], x[3], x[4], x[5])
    #print('   **', L)
    zs = str(ci).zfill(3)
    fn = 'landsat.bin_accumulate.bin_dedup.bin_' + zs + '.bin'
    #print(fn)

    hf = hdr_fn(fn)
    print(fn, hf)

    pre = 'K61884' # 'results'
    of = pre + '_' + zs + '_' + TYPE + '_' + TS + '.bin'
    oh = pre + '_' + zs + '_' + TYPE + '_' + TS + '.hdr'
    ot = pre + '_' + zs + '_' + TYPE + '_' + TS + '.tif'

    if not exist(of):
        run('cp ' + fn + ' ' + of)
    if not exist(oh):
        run('cp ' + hf + ' ' + oh)
    if not exist(ot):
        run('gdal_translate -of GTiff -ot Float32 ' + of + ' ' + ot)

    run('python3 ' + pd + 'envi_header_copy_mapinfo.py ' + FOOT_H + ' ' + oh)
    ci += 1
