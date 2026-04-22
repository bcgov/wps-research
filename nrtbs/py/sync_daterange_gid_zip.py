'''20230627: sync a date range for selected GID, in Level-2 to zip file format.

python3 sync_daterange_gid_zip.py [yyyymmdd] [yyyymmdd2] # optional: list of GID 
'''
use_L2 = True
data_type = 'MSIL2A'
if not use_L2:
    data_type = 'MSIL1C'

no_update_listing = False # flag to skip refreshing listing of aws archive

from misc import args, sep, exists, parfor, run, timestamp, err
from aws_download import aws_download
import multiprocessing as mp
from pathlib import Path
import datetime
import time
import json
import sys
import os

my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
product_target = os.getcwd() + sep # put ARD products into present folder

def download_by_gids(gids, yyyymmdd, yyyymmdd2):
    global no_update_listing

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
    
    data = None 

    if no_update_listing:
        latest = [x.strip() for x in os.popen('ls -r1 py/listing').readlines()]
        latest.sort()
        print("latest", latest)
        latest_path = 'py/listing/' + latest[-1]
        if len(latest) > 0 and os.path.exists(latest_path):
            print("+r", latest_path)
            data = open(latest_path).read() # .decode()
        else:
            err('probably need to run without --no_update_listing option first')
    else:
        ts = timestamp()
        cmd = ' '.join(['aws',  # read data from aws
                        's3api',
                        'list-objects',
                        '--no-sign-request',
                        '--no-verify-ssl',
                        '--bucket sentinel-products-ca-mirror'])
        print(cmd)
        data = os.popen(cmd).read()
        
        if not exists(my_path + 'listing'):  # json backup for analysis
            os.mkdir(my_path + 'listing')
        df = my_path + 'listing' + sep + ts + '_objects.txt'  # file to write
        open(df, 'wb').write(data.encode())  # record json to file


    jobs, d = [], None
    try:
        d = json.loads(data)  # parse json data
    except:
        print("Please confirm awscli is installed.")
        print("Recommended method:")
        print("\tpython3 -m pip install awscli")
        print("(note, have had problems with:")
        print("\tsudo apt install awscl  # this method did not work properly.")
        err('please confirm aws cli is installed!')
    data = d['Contents']  # extract the data records, one per dataset
    
    cmds = []
    for d in data:
        key, modified, file_size = d['Key'].strip(), d['LastModified'], d['Size']
        w = [x.strip() for x in key.split('/')]
        if w[0] == 'Sentinel-2':
            f = w[-1]
            fw = f.split('_')
            gid = fw[5]   # e.g. T10UGU

            out_dir = "L2_" + gid
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
    parfor(runc, cmds, int(mp.cpu_count()))  
'''

gids = []  # get gids from command line
if len(args) > 3:
    gids = set(args[3:])

    gids_use = set({})
    for gid in gids:
        if gid[0:2] != '--':
            gids_use.add(gid)
    gids = gids_use

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

    no_update_listing = '--no_update_listing' in args

    download_by_gids(gids, yyyymmdd, yyyymmdd2)
