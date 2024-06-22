'''20240618 sentinel2_mrap_merge.py

1) scan L2_XXXXX for each grid id
2) create a new mosaic for every date there's new data
'''
from misc import sep, args, exists, run, err, parfor
import multiprocessing as mp
import os

EPSG = 3005 if len(args) < 2 else 3347  # BC Albers / Canada LCC

def resample(fn):
    ofn = fn[:-4] + '_resample.bin'
    if not exists(ofn):
        cmds = (' '.join(['gdalwarp',
                          '-wo NUM_THREADS=16',
                          '-multi',
                          '-r bilinear',
                          '-srcnodata nan',
                          '-dstnodata nan',
                          '-of ENVI',
                          '-ot Float32',
                          '-t_srs EPSG:' + str(EPSG),
                          fn,
                          ofn]))
        return [cmds, ofn]
    else:
        return ['', ofn]

def merge(to_merge, out_fn): # files to be merged, output file name
    run(' '.join(['gdalbuildvrt',
                  '-srcnodata nan',
                  '-vrtnodata nan',
                  '-resolution highest',
                  '-overwrite',
                  'merge.vrt',
                  ' '.join(to_merge)]))

    if not exists(out_fn):
        run(' '.join(['gdalwarp',
                      '-wo NUM_THREADS=16',
                      '-multi',
                      '-overwrite',
                      '-r bilinear',
                      '-of ENVI',
                      '-ot Float32',
                      '-srcnodata nan',
                      '-dstnodata nan',
                      'merge.vrt',
                      out_fn]))

    run('fh ' + hdr_fn(out_fn))
    run('envi_header_copy_bandnames.py ' + hdr_fn(to_merge[0]) + ' ' + hdr_fn(out_fn))


dirs = [x.strip() for x in os.popen('ls -1d L2_*').readlines()]
gids = [d.split('_')[-1] for d in dirs]
print("gids", gids)

dic = {}
for d in dirs:
    print(d)
    mraps = [x.strip() for x in os.popen('ls -1 ' + d + sep + '*MRAP.bin').readlines()]
    for m in mraps:
        w = m.split(sep)[-1].split('_')[2].split('T')[0]
        #print(w, m)
        if w not in dic:
            dic[w] = []
        dic[w] += [m]

    #  parfor(run, cmds, int(mp.cpu_count()))

# sort dictionary contents by date
date_mrap = [[d, dic[d]] for d in dic]
date_mrap.sort() # list of MRAP files available on each date.
cmds = []
most_recent_by_gid = {}
for d, df in date_mrap:
    # print(d)
    for f in df:
        #print('  ', f)
        # cmds += [resample(f)]
        fn = f.split(sep)[-1]
        gid = fn.split('_')[5]

        # didn't record a most recent for this gid yet
        if gid not in most_recent_by_gid:
            most_recent_by_gid[gid] = {}
            most_recent_by_gid[gid][d] = [f]
        else:
            keys = list(most_recent_by_gid[gid].keys())
            if len(keys) != 1:
                err('consistency check')
            if int(keys[0]) < int(d):
                most_recent_by_gid[gid] = {}
                most_recent_by_gid[gid][d] = [f]
            elif int(keys[0]) > int(d):
                err('consistency check2')
            elif int(keys[0]) == int(d):
                # print('** multiples this date, insert')
                most_recent_by_gid[gid][d] += [f] 
            else:
                print(int(keys[0]), int(d))
                err('unreachable')
    print(d, 'RESULT')
    # list results
    results = []
    for gid in most_recent_by_gid:
        keys = list(most_recent_by_gid[gid].keys())
        if len(keys) != 1:
            err('consistency check 3')  
        # print('    ', gid, keys[0], most_recent_by_gid[gid][keys[0]])
        results += most_recent_by_gid[gid][keys[0]]
    
    results_sort =[]
    for r in results:
        w = r.split(sep)[-1].split('_')
        results_sort.append([w[2] +'_' +  w[6], r])
    results_sort.sort()

    for r in results_sort:
        print(r)
    #   parfor(run, cmds, int(mp.cpu_count()))
# iterate through dates, keeping a list of most-recent dates (by GID) 
    to_merge = [rs[1] for  rs in results_sort]

    cmds, resampled_files = [], []
    for m in to_merge:
        print('m', m)   
        cmd, resampled_file = resample(m)

        cmds += [cmd]
        resampled_files += [resampled_file]
    parfor(run, cmds, int(mp.cpu_count()))

    merge(resampled_files, str(d) + '_mrap.bin')
