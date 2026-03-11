'''20240618 sentinel2_mrap_merge.py

1) scan L2_XXXXX for each grid id
2) create a new mosaic for every date there's new data ( option to do this on every N-th date) 

20250605: nb, should have sentinel2_mrap.py pass in the dates that need to be (re) generated.
'''
from misc import sep, args, exists, run, err, parfor, hdr_fn
import multiprocessing as mp
import os
import threading
import queue
import time

# Extract --N [value] before other argument processing
N = 1
if '--N' in args:
    idx = args.index('--N')
    N = int(args[idx + 1])
    args.pop(idx)
    args.pop(idx)

# Extract --N_threads [value] before other argument processing
N_threads = 1
if '--N_threads' in args:
    idx = args.index('--N_threads')
    N_threads = int(args[idx + 1])
    args.pop(idx)
    args.pop(idx)

EPSG = 3005 if len(args) < 2 else 3347  # BC Albers / Canada LCC

merge_dates = None
if exists('.mrap_merge_dates'):
    merge_dates = [x.strip() for x in open('.mrap_merge_dates').readlines()]

# --- timing stats shared across threads ---
stats_lock = threading.Lock()
resample_times   = []   # seconds per resample call
merge_times      = []   # seconds per merge call
jobs_done        = [0]  # mutable counter: completed job count
jobs_total       = [0]  # set once job list is known

def _fmt(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def _print_global_status():
    with stats_lock:
        done       = jobs_done[0]
        total      = jobs_total[0]
        avg_rs     = (sum(resample_times)  / len(resample_times))  if resample_times  else None
        avg_mg     = (sum(merge_times)     / len(merge_times))     if merge_times     else None
        remaining  = total - done

    parts = [f'[STATUS] {done}/{total} jobs done']
    if avg_rs is not None:
        parts.append(f'avg_resample={avg_rs:.1f}s')
    if avg_mg is not None:
        parts.append(f'avg_merge={avg_mg:.1f}s')

    if avg_rs is not None and avg_mg is not None and remaining > 0:
        eta_s = remaining * (avg_rs + avg_mg)   # rough single-thread equivalent
        eta_s /= N_threads                       # parallelism speedup estimate
        parts.append(f'ETA={_fmt(eta_s)}')
    elif remaining == 0:
        parts.append('ETA=done')

    print(' | '.join(parts), flush=True)


def resample(fn, target_label):
    """Resample fn -> /ram/, return output path."""
    basename = os.path.basename(fn)
    ofn = os.path.join('/ram', basename[:-4] + '_resample.bin')

    if exists(ofn):
        os.remove(ofn)

    cmd = ' '.join(['gdalwarp',
                    '-wo NUM_THREADS=4',
                    '-multi',
                    '-r bilinear',
                    '-srcnodata nan',
                    '-dstnodata nan',
                    '-of ENVI',
                    '-ot Float32',
                    '-t_srs EPSG:' + str(EPSG),
                    fn,
                    ofn])
    return [cmd, ofn]


def merge(to_merge, date, out_fn, target_label):
    """Build VRT in /ram/, warp final product to cwd."""
    vrt = os.path.join('/ram', str(date) + '_merge.vrt')

    if exists(vrt):
        os.remove(vrt)

    run(' '.join(['gdalbuildvrt',
                  '-srcnodata nan',
                  '-vrtnodata nan',
                  '-resolution highest',
                  '-overwrite',
                  vrt,
                  ' '.join(to_merge)]))

    cmd = ' '.join(['gdalwarp',
                    '-wo NUM_THREADS=4',
                    '-multi',
                    '-overwrite',
                    '-r bilinear',
                    '-of ENVI',
                    '-ot Float32',
                    '-srcnodata nan',
                    '-dstnodata nan',
                    vrt,
                    out_fn])
    run(cmd)

    run('fh ' + hdr_fn(out_fn))
    run('envi_header_copy_bandnames.py ' + hdr_fn(to_merge[-1]) + ' ' + hdr_fn(out_fn))


def process_job(job):
    """Execute one full resample+merge job. Called by each worker thread."""
    d, to_merge, mrap_product_file = job
    target_label = mrap_product_file   # human-readable reference throughout

    # --- resample phase ---
    t0 = time.time()
    cmds, resampled_files = [], []
    for m in to_merge:
        cmd, resampled_file = resample(m, target_label)
        cmds.append(cmd)
        resampled_files.append(resampled_file)

    parfor(run, cmds, int(mp.cpu_count()))
    rs_elapsed = time.time() - t0

    with stats_lock:
        resample_times.append(rs_elapsed)
    print(f'[RESAMPLE done] target={target_label} | files={len(cmds)} | elapsed={rs_elapsed:.1f}s', flush=True)
    _print_global_status()

    # --- merge / assembly phase ---
    t1 = time.time()
    merge(resampled_files, d, mrap_product_file, target_label)
    mg_elapsed = time.time() - t1

    with stats_lock:
        merge_times.append(mg_elapsed)
        jobs_done[0] += 1
    print(f'[MERGE done]    target={target_label} | elapsed={mg_elapsed:.1f}s', flush=True)
    _print_global_status()

    # clean up resampled intermediates from /ram/
    for rf in resampled_files:
        if exists(rf):
            os.remove(rf)


def worker(work_queue):
    """Thread worker: pull jobs until queue is empty."""
    while True:
        try:
            job = work_queue.get_nowait()
        except queue.Empty:
            break
        try:
            process_job(job)
        finally:
            work_queue.task_done()


# ---------------------------------------------------------------------------
# scan directories and build date → MRAP file mapping
# ---------------------------------------------------------------------------
dirs = [x.strip() for x in os.popen('ls -1d L2_*').readlines()]
gids = [d.split('_')[-1] for d in dirs]
print("gids", gids)

dic = {}
for d in dirs:
    print(d)
    mraps = [x.strip() for x in os.popen('ls -1 ' + d + sep + '*MRAP.bin').readlines()]
    for m in mraps:
        w = m.split(sep)[-1].split('_')[2].split('T')[0]
        if w not in dic:
            dic[w] = []
        dic[w] += [m]

date_mrap = [[d, dic[d]] for d in dic]
date_mrap.sort()
cmds, most_recent_by_gid = [], {}

for date_idx, (d, df) in enumerate(date_mrap):

    for f in df:
        fn  = f.split(sep)[-1]
        gid = fn.split('_')[5]

        if gid not in most_recent_by_gid:
            most_recent_by_gid[gid] = {}
            most_recent_by_gid[gid][d] = [f]
        else:
            keys = list(most_recent_by_gid[gid].keys())

            if len(keys) != 1:
                err('consistency check 1 failed')

            if int(keys[0]) < int(d):
                most_recent_by_gid[gid] = {}
                most_recent_by_gid[gid][d] = [f]
            elif int(keys[0]) > int(d):
                err('consistency check 2 failed')
            elif int(keys[0]) == int(d):
                most_recent_by_gid[gid][d] += [f]
            else:
                print(int(keys[0]), int(d))
                err('unreachable')

    print(d, 'RESULT')
    results = []
    for gid in most_recent_by_gid:
        keys = list(most_recent_by_gid[gid].keys())
        if len(keys) != 1:
            err('consistency check 3')
        results += most_recent_by_gid[gid][keys[0]]

    results_sort = []
    for r in results:
        w = r.split(sep)[-1].split('_')
        results_sort.append([w[2] + '_' + w[6], r])
    results_sort.sort()

    for r in results_sort:
        print(r)

# ---------------------------------------------------------------------------
# build job list (all filtering done up front so we know the total count)
# ---------------------------------------------------------------------------
job_list = []

most_recent_by_gid = {}   # reset for second pass to rebuild job list cleanly

for date_idx, (d, df) in enumerate(date_mrap):

    for f in df:
        fn  = f.split(sep)[-1]
        gid = fn.split('_')[5]

        if gid not in most_recent_by_gid:
            most_recent_by_gid[gid] = {d: [f]}
        else:
            keys = list(most_recent_by_gid[gid].keys())
            if len(keys) != 1:
                err('consistency check 1b failed')
            if int(keys[0]) < int(d):
                most_recent_by_gid[gid] = {d: [f]}
            elif int(keys[0]) > int(d):
                err('consistency check 2b failed')
            elif int(keys[0]) == int(d):
                most_recent_by_gid[gid][d] += [f]
            else:
                err('unreachable b')

    # apply filters before adding to job list
    if (merge_dates is not None) and (d not in merge_dates):
        continue
    if date_idx % N != 0:
        print(f'SKIPPING (--N stride) {d}', flush=True)
        continue

    mrap_product_file = str(d) + '_mrap.bin'
    if exists(mrap_product_file):
        print(f'SKIPPING (exists) {mrap_product_file}', flush=True)
        continue

    results = []
    for gid in most_recent_by_gid:
        keys = list(most_recent_by_gid[gid].keys())
        if len(keys) != 1:
            err('consistency check 3b')
        results += most_recent_by_gid[gid][keys[0]]

    results_sort = []
    for r in results:
        w = r.split(sep)[-1].split('_')
        results_sort.append([w[2] + '_' + w[6], r])
    results_sort.sort()

    to_merge = [rs[1] for rs in results_sort]
    job_list.append((d, to_merge, mrap_product_file))

jobs_total[0] = len(job_list)
print(f'[PLAN] {len(job_list)} jobs to run across {N_threads} thread(s), N={N}', flush=True)

# ---------------------------------------------------------------------------
# populate work queue and launch worker threads
# ---------------------------------------------------------------------------
work_queue = queue.Queue()
for job in job_list:
    work_queue.put(job)

threads = []
for _ in range(N_threads):
    t = threading.Thread(target=worker, args=(work_queue,), daemon=True)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print('[DONE] all jobs complete.', flush=True)

