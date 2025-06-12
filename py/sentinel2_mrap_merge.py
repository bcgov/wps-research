'''20240618 sentinel2_mrap_merge.py

1) scan L2_XXXXX for each grid id
2) create a new mosaic for every date there's new data

20250605: nb, should have sentinel2_mrap.py pass in the dates that need to be (re) generated.
'''
from misc import sep, args, exists, run, err, parfor, hdr_fn 
import multiprocessing as mp
import os

EPSG = 3005 if len(args) < 2 else 3347  # BC Albers / Canada LCC

merge_dates = None
if exists('.mrap_merge_dates'):
    merge_dates = [x.strip() for x in open('.mrap_merge_dates').readlines()]

def resample(fn):
    ofn = fn[:-4] + '_resample.bin'
    if True: # not exists(ofn):
        if exists(ofn):
            os.remove(ofn)

        cmds = (' '.join(['gdalwarp',
                          '-wo NUM_THREADS=64',
                          '-multi',
                          '-r bilinear',
                          '-srcnodata nan',
                          '-dstnodata nan',
                          '-of ENVI',
                          '-ot Float32',
                          '-t_srs EPSG:' + str(EPSG),
                          fn,  # input file
                          ofn]))  # output file
        return [cmds, ofn]
    else:
        return ['', ofn]

def merge(to_merge, date, out_fn): # files to be merged, output file name
    if True: # not exists(str(date) + '_merge.vrt'):
        if exists(str(date) + '_merge.vrt'):
            os.remove(str(date) + '_merge.vrt')
        run(' '.join(['gdalbuildvrt',
                      '-srcnodata nan',
                      '-vrtnodata nan',
                      '-resolution highest',
                      '-overwrite',
                      str(date) + '_merge.vrt',  # output file
                      ' '.join(to_merge)]))

    cmd = ' '.join(['gdalwarp',
                    '-wo NUM_THREADS=64',
                    '-multi',
                    '-overwrite',
                    '-r bilinear',
                    '-of ENVI',
                    '-ot Float32',
                    '-srcnodata nan',
                    '-dstnodata nan',
                    str(date) + '_merge.vrt',
                    out_fn])  # output file

    if True: # not exists(out_fn):
        run(cmd) 
    else:
        print(cmd)

    run('fh ' + hdr_fn(out_fn))
    run('envi_header_copy_bandnames.py ' + hdr_fn(to_merge[-1]) + ' ' + hdr_fn(out_fn))


dirs = [x.strip() for x in os.popen('ls -1d L2_*').readlines()]
gids = [d.split('_')[-1] for d in dirs]
print("gids", gids)

dic = {}  # list the tile-based MRAP files, for each associated date key. yyyymmdd
for d in dirs:
    print(d)
    mraps = [x.strip() for x in os.popen('ls -1 ' + d + sep + '*MRAP.bin').readlines()]
    for m in mraps:
        w = m.split(sep)[-1].split('_')[2].split('T')[0]  # key on date yyyymmdd
        #print(w, m)
        if w not in dic:
            dic[w] = []
        dic[w] += [m]

# sort the date keys to get a list of mrap files, for each date, ordered by date yyyymmdd 
date_mrap = [[d, dic[d]] for d in dic]
date_mrap.sort() # list of MRAP files available on each date.
cmds, most_recent_by_gid = [], {}  # list the most recent MRAP file observed for this GID, for a given date yyyymmdd

for d, df in date_mrap:  # for each date ( and associated list of MRAP files observed that date yyyymmdd ) 
    # print(d)

    for f in df:
        #print('  ', f)
        fn = f.split(sep)[-1]
        gid = fn.split('_')[5]

        # didn't record a most recent for this gid yet
        if gid not in most_recent_by_gid:
            most_recent_by_gid[gid] = {}
            most_recent_by_gid[gid][d] = [f]
        else:
            # list the dates yyyymmdd observed ( this GID ) so far
            keys = list(most_recent_by_gid[gid].keys())  

            # "by definition" we should store only one "most recent" file ( per gid )  at one time.
            if len(keys) != 1:
                err('consistency check 1 failed')

            # there is only one most recent ( yyyymmdd ) date key. If we found a new date that's more recent:
            if int(keys[0]) < int(d):
                # clear the dictionary and add this file as a key:
                most_recent_by_gid[gid] = {}
                most_recent_by_gid[gid][d] = [f]

            # seems pedantic: 
            elif int(keys[0]) > int(d):
                err('consistency check 2 failed')

            # multiple files this (gid, yyyymmdd) are allowed ( reprocessings, or multiple takes same calendar day, different times ) 
            elif int(keys[0]) == int(d): # print('** multiples this date, insert')
                most_recent_by_gid[gid][d] += [f] 

            # sanity check: unreachable section: all cases should be covered
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

    # iterate through dates, keeping a list of most-recent dates (by GID) 
    to_merge = [rs[1] for  rs in results_sort]

    cmds, resampled_files = [], []
    for m in to_merge:
        print('m', m)   
        cmd, resampled_file = resample(m)

        cmds += [cmd]
        resampled_files += [resampled_file]

    if (merge_dates is not None) and (d not in merge_dates):
            # MRAP mosaic product not to be created for this date.
            continue
    else:
        # create MRAP mosaic product for this date. Thought this was every date?
        mrap_product_file = str(d) + '_mrap.bin'
        
        if exists(mrap_product_file):
            print('SKIPPING', mrap_product_file)
            continue

        parfor(run,
               cmds,
               int(mp.cpu_count()))  # perform resampling steps in parallel

        merge(resampled_files,
              d,
              mrap_product_file)  # merge step can't be performed in parallel as easily, however techincally we should be able to add an ampersand here and continue over the loop since the MRAP files are cumulative in time ( on a per-time / gid basis ) that is the MRAP merge files aren't defined in terms of previous MRAP merge files.. only MRAP files ( on a per-tile/ gid basis)
