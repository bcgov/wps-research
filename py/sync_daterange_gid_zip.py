'''20230627: sync a date range for selected GID, in Level-2 to zip file format.

20230627: sync a date range for selected GID

Instructions: run from path on SSD,
     with "/ram" device enabled using   
        python3 py/ramdisk.py 

20230605 need to revise away from the batch/cycling op
20230604 added looping
20230526 download data for each 5-char "UTM tiling-grid ID":
        https://eatlas.org.au/data/uuid/f7468d15-12be-4e3f-a246-b2882a324f59
specified, 
for specified date: yyyymmdd only

python3 sync_date_gid_ramdisk.py [date: yyyymmdd] e.g.
python3 sync_date_gid_ramdisk.py 20230525  # download all data over BC for 20230525
python3 sync_date_gid_ramdisk.py 20230525 all # download all data from national mirror 

e.g. for NTFS001:
    python3 ~/GitHub/s2-fire-mapping/sync_date_gid.py 20230530 10VEM 10VFM 

e.g. for NTSSO08:
    python3 ~/GitHub/s2-fire-mapping/sync_date_gid.py 20230530 11VLG 11VLH 11VMH 11VMG'''
use_L2 = True
data_type = 'MSIL2A'
if not use_L2:
    data_type = 'MSIL1C'

from misc import args, sep, exists, parfor, run, timestamp, err
import multiprocessing as mp
import datetime
import time
import json
import sys
import os
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
product_target = os.getcwd() + sep # put ARD products into present folder

def download_by_gids(gids, date_range):
    ts = timestamp()
    cmd = ' '.join(['aws',  # read data from aws
                    's3api',
                    'list-objects',
                    '--no-sign-request',
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
        err('please confirm aws cli: e.g. sudo apt install awscli')
    data = d['Contents']  # extract the data records, one per dataset
    
    cmds = []
    for d in data:
        key, modified, file_size = d['Key'].strip(), d['LastModified'], d['Size']
        w = [x.strip() for x in key.split('/')]
        if w[0] == 'Sentinel-2':
            f = w[-1]
            fw = f.split('_')
            gid = fw[5][1:]  # e.g. T10UGU
            ts = fw[2].split('T')[0]  # e.g. 20230525
            if fw[1] != data_type or ts not in date_range:  # wrong product or outside date range
                continue
            if gids is not None and gid not in gids:  # only level-2 for selected date and gid
                continue

            cmd = ' '.join(['aws',
                            's3',
                            'cp',
                            '--no-sign-request',
                            's3://sentinel-products-ca-mirror/' + key,
                            f])
     
            if exists(f):
                print(f, "SKIPPING")
            else:
                print(f)
                cmds += [cmd]
    
    print(cmds)
    def runc(c):
        print([c])
        return os.system(c)
    parfor(runc, cmds, int(mp.cpu_count()))  

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

yyyymmdd, yyyymmdd2 = args[1], args[2]
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
   

while(True):  # make it go. Need to add termination when completed.
    download_by_gids(gids, date_range)
    # print('waiting 1 min..')
    # time.sleep(60)
    break
print('done')
