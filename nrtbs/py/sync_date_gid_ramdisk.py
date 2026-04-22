'''
sync_date_gid_ramdisk:
*** This script pulls L1 data (e.g. over all-province) for a single date only. SWIR bands (B12, B11, B9) only. RAMDISK acceleration is required for this script. This script is useful for generating quick and dirty composites of present-day data (e.g. after 7 or 8pm BC time when new data may come in). L1 data comes in faster than L2. 

This approach is relevant for fire mapping (SWIR bands only, i.e. not NIR/NBR) for present-day data only (non-composited).   
***

Instructions: typically run from path on SSD,
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
    python3 ~/GitHub/s2-fire-mapping/sync_date_gid.py 20230530 11VLG 11VLH 11VMH 11VMG
'''
use_L2 = False # True
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

def download_by_gids(gids, date_string):
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
    
    for d in data:
        key, modified, file_size = d['Key'].strip(), d['LastModified'], d['Size']
        w = [x.strip() for x in key.split('/')]
        if w[0] == 'Sentinel-2':
            f = w[-1]
            fw = f.split('_')
            gid = fw[5][1:]  # e.g. T10UGU
            ts = fw[2].split('T')[0]  # e.g. 20230525
            if fw[1] != data_type or ts != date_string:  # wrong date or product
                continue
            if gids is not None and gid not in gids:  # only level-2 for selected date and gid
                continue

            cmd = ' '.join(['aws',
                            's3',
                            'cp',
                            '--no-sign-request',
                            's3://sentinel-products-ca-mirror/' + key,
                            f])
            
            product_target_file =  product_target + f[:-3] + 'bin'
            prod_target_hdr = product_target + f[:-3] + 'hdr'
            prod_file = f[:-3] + 'bin'
            prod_hdr = f[:-3] + 'hdr'
            if not exists(product_target_file):
                jobs += [{'zip_filename': f,
                          'gid': gid,
                          'date_string': ts,
                          'download_command': cmd,
                          'prod_target': product_target_file,
                          'prod_target_hdr': prod_target_hdr,
                          'prod_file': prod_file,
                          'prod_hdr': prod_hdr}]
 
    def run_job(j):
        os.chdir('/ram/')
        run(j['download_command'])
        run('sentinel2_extract_swir.py ' + j['zip_filename'])
        run('mv -v ' + j['prod_file'] + ' ' + j['prod_target'])
        run('mv -v ' + j['prod_hdr'] + ' ' + j['prod_target_hdr'])
        run('rm -v ' + j['zip_filename'])
        # run('rm ' + j['prod_file'])
        # run('rm ' + j['prod_hdr'])

    parfor(run_job, jobs, int(mp.cpu_count()))  

gids = []  # get gids from command line
if len(args) > 2:
    gids = set(args[2:])

if len(gids) == 0:  # if no gids provided, default to all gids for BC
    from gid import bc
    gids = bc()
    print('pulling BC data..')
else:
    if 'all' in gids:
        gids = None
        print('pulling Canada data..')

yyyymmdd = args[1]
if len(yyyymmdd) != 8:
    err('expected date in format yyyymmdd')

while(True):  # make it go. Need to add termination when completed.
    download_by_gids(gids, yyyymmdd)
    print('waiting 30 min..')
    time.sleep(60 * 30)
print('done')
