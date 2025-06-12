'''20230627: sync a date range for selected GID, in Level-2 to zip file format.

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
import time
import json
import sys
import os
my_path_0 = '/data/.listing/'
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

def download_by_gids(gids, yyyymmdd, yyyymmdd2):
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
        start_d += datetime.timedelta(days=1)
        date_range += [str(start_d.year).zfill(4) + str(start_d.month).zfill(2) + str(start_d.day).zfill(2)]

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
    
    cmds = []
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

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            '''
            cmd = ' '.join(['aws',
                            's3',
                            'cp',
                            '--no-sign-request',
                            's3://sentinel-products-ca-mirror/' + key,
                            f])
            print([f])'''
            if exists(f) and Path(f).stat().st_size == file_size:
                print(f, "SKIPPING")
            else:
                print(f)
                #cmds += [cmd]

                aws_download('sentinel-products-ca-mirror',
                             key,
                             Path(f)) # LOCAL_PATH):
                # === CONFIGURATION ===
                # BUCKET = "sentinel-products-ca-mirror"
                # KEY = "Sentinel-2/S2MSI2A/2025/05/27/S2C_MSIL2A_20250527T191931_N0511_R099_T10VFL_20250528T002013.zip"
                # LOCAL_PATH = Path("L2_T10VFL") / Path(KEY).name


    
    '''
    print(cmds)
    def runc(c):
        print([c])
        return os.system(c)
    parfor(runc, cmds, 2) # min(int(2), 2 * int(mp.cpu_count())))  
    '''

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


gids = []  # get gids from command line
if len(args) > 3:
    gids = set(args[3:])

if len(gids) == 0:  # if no gids provided, default to all gids for BC
    from gid import bc
    gids = bc()
    print('pulling BC data..')
else:
    if 'all' in gids:
        gids = None
        print('pulling Canada data..')


if __name__ == "__main__":
    yyyymmdd, yyyymmdd2 = args[1], args[2]
    download_by_gids(gids, yyyymmdd, yyyymmdd2)
