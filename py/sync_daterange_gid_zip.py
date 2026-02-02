'''20260202: added single-date option!

20260127: progress monitor added 

20230627: sync a date range for selected GID, in Level-2 to zip file format.

python3 sync_daterange_gid_zip.py [yyyymmdd] [yyyymmdd2] # optional: list of GID
'''
use_L2 = False
data_type = None
''''MSIL2A'
if not use_L2:
    data_type = 'MSIL1C'
'''
from misc import args, sep, exists, parfor, run, timestamp, err
from aws_download import aws_download
import multiprocessing as mp
from pathlib import Path
import datetime
import argparse
import shutil
import time
import json
import sys
import os

# Global status tracking
download_start_time = None
files_completed = 0
total_files = 0
bytes_completed = 0
total_bytes = 0

my_path_0 = '/data/'
my_path_1 = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
my_path = my_path_1
try:  # json backup
    if not exists(my_path_0):
        print('mkdir', my_path_0)
        os.mkdir(my_path_0)
    my_path = my_path_0
except:
    if not exists(my_path_1 + '.listing'):
        os.mkdir(my_path_1 + '.listing')

product_target = os.getcwd() + sep # put ARD products into present folder

# parser = argparse.ArgumentParser()
# parser.add_argument("-n", "--no_refresh", action="store_true",
#                     help="do not refresh aws bucket listing, use most recent list instead")
no_refresh = True # args.no_refresh

def print_status_update(file_name, file_size, file_download_time):
    global files_completed, total_files, bytes_completed, total_bytes, download_start_time

    elapsed_time = time.time() - download_start_time
    pct_files = (files_completed / total_files * 100) if total_files > 0 else 0
    pct_bytes = (bytes_completed / total_bytes * 100) if total_bytes > 0 else 0

    if files_completed > 0:
        time_per_file = elapsed_time / files_completed
        files_remaining = total_files - files_completed
        eta_seconds = time_per_file * files_remaining
    else:
        eta_seconds = 0

    print(f"\n{'='*60}")
    print(f"DOWNLOAD STATUS UPDATE")
    print(f"{'='*60}")
    print(f"File completed: {file_name}")
    print(f"File size: {file_size / (1024*1024):.2f} MB | File download time: {file_download_time:.1f}s")
    print(f"{'='*60}")
    print(f"Files completed:    {files_completed} / {total_files} ({pct_files:.1f}%)")
    print(f"Bytes completed:    {bytes_completed / (1024*1024*1024):.2f} / {total_bytes / (1024*1024*1024):.2f} GB ({pct_bytes:.1f}%)")
    print(f"Time elapsed:       {elapsed_time / 60:.1f} min")
    print(f"Time remaining:     {eta_seconds / 60:.1f} min | {eta_seconds / 3600:.2f} hrs | {eta_seconds / 86400:.3f} days (ETA)")
    print(f"{'='*60}\n")

def download_by_gids(gids, yyyymmdd, yyyymmdd2):
    global files_completed, total_files, bytes_completed, total_bytes, download_start_time

    if len(yyyymmdd) != 8 or len(yyyymmdd2) != 8:
        err('expected date in format yyyymmdd')
    start_d = datetime.datetime(int(yyyymmdd[0:4]),
                                int(yyyymmdd[4:6]),
                                int(yyyymmdd[6:8]))
    end_d = datetime.datetime(int(yyyymmdd2[0:4]),
                              int(yyyymmdd2[4:6]),
                              int(yyyymmdd2[6:8]))
    print("start", start_d, "end", end_d)
    date_range = []
    while start_d <= end_d:
        print(start_d)
        date_range += [str(start_d.year).zfill(4) + str(start_d.month).zfill(2) + str(start_d.day).zfill(2)]
        start_d += datetime.timedelta(days=1)

    print(date_range)

    ts = timestamp()
    cmd = ' '.join(['aws',  # read data from aws
                    's3api',
                    'list-objects',
                    '--no-sign-request',
                    '--bucket sentinel-products-ca-mirror'])
    print(cmd)
    list_dir = my_path + '.listing' + sep
    if not no_refresh:
        data = os.popen(cmd).read()
    else:
        data_files = [x.strip() for x in os.popen('ls -1 ' + list_dir).readlines()]
        data_files.sort(reverse=True)
        for d in data_files:
            print('  ', d)
        print('+r', list_dir + data_files[0])
        data = open(list_dir + data_files[0]).read() # .decode()

    if not no_refresh:
        print('caching at', list_dir)
        if not exists(list_dir):  # json backup for analysis
            os.mkdir(list_dir)
        df = list_dir + ts + '_objects.txt'  # file to write
        open(df, 'wb').write(data.encode())  # record json to file
    else:
        print('Skip caching listing at', list_dir)

    jobs, d = [], None
    try:
        d = json.loads(data)  # parse json data
    except:
        err('please confirm aws cli: e.g. sudo apt install awscli')
    data = d['Contents']  # extract the data records, one per dataset

    # First pass: collect files to download and calculate totals
    files_to_download = []
    for d in data:
        key, modified, file_size = d['Key'].strip(), d['LastModified'], d['Size']
        w = [x.strip() for x in key.split('/')]
        if w[0] == 'Sentinel-2':
            f = w[-1]
            fw = f.split('_')
            gid = fw[5]   # e.g. T10UGU

            out_dir = ("L2_" if use_L2 else "L1_") + gid
            f = out_dir + os.path.sep + f

            ts = fw[2].split('T')[0]  # e.g. 20230525
            if fw[1] != data_type or ts not in date_range:  # wrong product or outside date range
                continue
            if gids is not None and gid not in gids:  # only level-2 for selected date and gid
                continue

            if exists(f) and Path(f).stat().st_size == file_size:
                print(f, "SKIPPING")
            else:
                files_to_download.append({'key': key, 'file': f, 'size': file_size, 'out_dir': out_dir})

    # Calculate totals
    total_files = len(files_to_download)
    total_bytes = sum(item['size'] for item in files_to_download)

    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total files to download: {total_files}")
    print(f"Total download size: {total_bytes / (1024*1024*1024):.2f} GB")
    print(f"{'='*60}\n")

    # Check available disk space
    disk_usage = shutil.disk_usage(os.getcwd())
    available_space = disk_usage.free

    if available_space < total_bytes:
        print(f"\n{'='*60}")
        print(f"ERROR: INSUFFICIENT DISK SPACE")
        print(f"{'='*60}")
        print(f"Required: {total_bytes / (1024*1024*1024):.2f} GB")
        print(f"Available: {available_space / (1024*1024*1024):.2f} GB")
        print(f"{'='*60}\n")
        err('Insufficient disk space. Exiting before downloads.')

    if total_files == 0:
        print("No files to download.")
        return

    # Record start time
    download_start_time = time.time()

    # Second pass: download files
    for item in files_to_download:
        key = item['key']
        f = item['file']
        file_size = item['size']
        out_dir = item['out_dir']

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        print(f)

        file_start_time = time.time()
        aws_download('sentinel-products-ca-mirror',
                     key,
                     Path(f))
        file_end_time = time.time()
        file_download_time = file_end_time - file_start_time

        # Update global counters
        files_completed += 1
        bytes_completed += file_size

        # Print status update
        print_status_update(f, file_size, file_download_time)


    '''
    print(cmds)
    def runc(c):
        print([c])
        return os.system(c)
    parfor(runc, cmds, 2) # min(int(2), 2 * int(mp.cpu_count())))
    '''

def is_date_format(s):
    """Check if string is in yyyymmdd format (8 digits)"""
    return len(s) == 8 and s.isdigit()

# check if L2 mode is desired ( L1 mode default )
use_L2 = '--L2' in args

if '--L2' in args and '--L1' in args:
    err("Must select L2 or L1")

if '--L1' in args:
    use_L2 = False

data_type = 'MSIL2A'
if not use_L2:
    data_type = 'MSIL1C'

new_args = []
for arg in args:
    if arg[:2] != '--':
        new_args += [arg]
args = new_args


# Parse dates and GIDs from arguments
# args[0] is script name, args[1] is first date, args[2] may be second date or GID
yyyymmdd = args[1]

# Check if args[2] exists and is a date (8 digits), otherwise use same date for both
if len(args) > 2 and is_date_format(args[2]):
    yyyymmdd2 = args[2]
    gid_start_index = 3
else:
    yyyymmdd2 = yyyymmdd  # Single date: use same for start and end
    gid_start_index = 2

gids = []  # get gids from command line
if len(args) > gid_start_index:
    gids = set(args[gid_start_index:])

if len(gids) == 0:  # if no gids provided, default to all gids for BC
    from gid import bc
    gids = bc()
    print('pulling BC data..')
else:
    if 'all' in gids:
        gids = None
        print('pulling Canada data..')


if __name__ == "__main__":
    download_by_gids(gids, yyyymmdd, yyyymmdd2)
