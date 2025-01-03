'''20220314 add date/timestamps to KML files, if parent folder is
a Sentinel-2 UTM date/timestamped folder

if kml file does not have a time stamp, take parent folder timestamp (assumed UTC), 
convert to PST and include that in the kml filename'''
from datetime import datetime
from pytz import timezone
from misc import *
import shutil
PST = timezone('US/Pacific')

files = [x.strip() for x in os.popen('find ./ -name "*.kml"').readlines()] if len(args) < 2 else [args[1]]

for f in files:
    parent_path = sep.join(os.path.abspath(f).split(sep)[:-1])
    parent_f = parent_path.split(sep)[-1]

    # is one-up/parent folder: an S2 folder i.e. S2*.SAFE?
    if parent_f[-5:] == '.SAFE' and parent_f[:2] == 'S2':
        w = parent_f.split('_')
        if w[6][8] != 'T':
            err('unexpected folder name format: ' + parent_path)
        ds, ts = w[6][:8], w[6][9: 15]
        YYYY, MM, DD = ds[0:4], ds[4:6], ds[6:8]
        hh, mm, ss = ts[0:2], ts[2:4], ts[4:6]
        [YYYY, MM, DD, hh, mm, ss] = [int(x) for x in
                                      [YYYY, MM, DD, hh, mm, ss]]
        d = datetime(YYYY, MM, DD, hh, mm, ss) 
        x = PST.localize(d)
        time_diff = x.tzinfo.utcoffset(x)
        local_time = d + time_diff

        w = f.split(sep)[-1]
        if len(w.split('_' + str(YYYY))) < 2:
            # no terminal timestamp
            w = w.split('.')
            L = local_time
            L = ''.join([str(L.year).zfill(4),
                         str(L.month).zfill(2),
                         str(L.day).zfill(2),
                         str(L.hour).zfill(2),
                         str(L.minute).zfill(2),
                         str(L.second).zfill(2)])
            L = '.'.join(w[:-1]) + '_' + L + '.' + w[-1]
            nf = f.split(sep)[:-1] + [L]
            nf = sep.join(nf)
            # print('mv ' + f + ' ' + nf)
            shutil.move(f, nf)
            print(nf)
        else:
            print(f)
