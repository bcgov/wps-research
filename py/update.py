'''20220823 update sentinel2 tiles present in subfolders
- assume we just ran find_sentinel2.py
- assume existing tiles are processed to Level-2 (BOA reflectance)
- download any newer tiles:
    (*) preferably in Level2 format.
    (*) if Level1 are available, download them, run Sen2Cor and extract

- Group new acquisitions into folders yyyymmdd (possibly multiple tiles in each folder)'''
import os
from misc import run, err, sep

files = [x.strip() for x in os.popen('find ./ -name "S2*L2A*"').readlines()]
tiles = {}

for y in files:  # group by tile and refresh every tile that we can
    fn = y.split(sep)[-1]  # filename e.g. .SAFE folder or zip file
    ix = fn.split('.')[0]  # scene ID
    w = ix.split('_')
    ti = w[5]  # tile ID 
    ts = int(w[2].split('T')[0])
    # print(ti, ts, fn, ix)
    if ti not in tiles:
        tiles[ti] = []
    tiles[ti] += [[ts, ix, fn]]

ci, failed = 0, []
for ti in tiles:
    print(ti, '-' * 77)
    it = 0
    last_date = None
    tiles[ti].sort(reverse=True)
    for [ts, ix, fn] in tiles[ti]:
        # print(ts)
        if it == 0:
            last_date = ts
            print('\t', ts, ix) #, fn)
        ci += 1
        it += 1

    # find latest newer date (if there is one) this tile
    dates_l1 = []
    dates_l2 = []
    lines = [x.strip() for x in os.popen('grep _' + ti + '_ fpf_download.sh').readlines()]
    for line in lines:
        line = line.strip().replace(' & ', ' ')
        w = line.split()
        zfn = w[3]
        # print('\t', zfn)
        w2 = zfn.split('_')
        level = w2[1]
        tsi = int(w2[2].split('T')[0])
        d = [tsi, zfn.split('.')[0], line]
        if level == 'MSIL1C':
            dates_l1 += [d]
        elif level == 'MSIL2A':
            dates_l2 += [d]
        else:
            err('unrecognized processing level:', level)
    dates_l1.sort(reverse=True)
    dates_l2.sort(reverse=True)

    d_use = None
    if len(dates_l1) > 0:
        d_use = dates_l1[0]

    if len(dates_l2) > 0:
        d_2 = dates_l2[0]
        if d_use is not None:
            if d_2[0] >= d_use[0]:
                d_use = d_2
        else:
            d_use = d_2

    if d_use is None:
        print("Didn't find newer date for tile: " + ti)

    else:
        t_use = str(d_use[0])  # date string to download
        if not os.path.exists(t_use):
            os.mkdir(t_use)

        zfn = d_use[1] + '.zip'
        if not os.path.exists(t_use + sep + zfn):
            cmd = d_use[2]
            print('wget ' + zfn)
            a = os.system(cmd)
            run('mv -v ' + zfn + ' .' + sep + t_use + sep)
        else:
            print("Didn't find newer date for tile: " + ti)
        if not os.path.exists(t_use + sep + zfn):
            failed += [zfn]

if len(failed) > 0:
    print("Failed to download:")
    for i in failed:
        print('\t', i)
    err('Download(s) failed please re-run later')
