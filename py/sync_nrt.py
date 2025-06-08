'''20250607: sync_nrt.py: pull today's L1 data. Select GID based on what GID are present in current folder ( L1 or L2 ) 

Then, call merge2.py on those date.

based on:  20250603 sync_recent.py: sync sentinel2 data from NRCAN AWS mirror from today ( or yesterday, or back N days )

need to add options for --L1, --L2 ( default ), --n ( number of days to go back, default 1 )'''
import os
import sys
import datetime
from gid import bc
from pathlib import Path
from datetime import timedelta
from aws_download import aws_download
from misc import err, parfor, sep, assert_aws_cli_installed
# assert_aws_cli_installed()
bc_gid = bc()
print("bc row-id under obs:", bc_gid)

gids = bc_gid # default to BC gids a

# today's date
today = datetime.date.today()
N = 1  # default

# check if we're in an MRAP folder, only update the GID present in the filesystem structure:
L1_folders = os.popen("ls -d1 L1_*").read().strip().split('\n')
L2_folders = os.popen("ls -d1 L2_*").read().strip().split('\n')
print(L1_folders)
print(L2_folders)

gids = [line.split('_')[1] for line in L1_folders] + [line.split('_')[1] for line in L2_folders]
gids = list(set(gids))

for g in gids:
    # today's date
    now = today
    year, month, day = str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2)
    print([year, month, day])
    ls = 'aws s3 ls --no-sign-request'
    path = 's3://sentinel-products-ca-mirror/Sentinel-2/'
    s2 = path
    a = " "
    c1, c2 = ' '.join([ls, path + 'S2MSI1C/']), ' '.join([ls, path + 'S2MSI2A/'])
    c1 = ls + a + s2 + 'S2MSI1C/' # + '/'.join([year, month, day]) + '/' 
    c2 = ls + a + s2 + 'S2MSI2A/' # + '/'.join([year, month, day]) + '/'

    def get(c):  # collect results from cli invocation
        print(c)
        t = [x.strip() for x in os.popen(c).read().strip().split('\n')]
        return '\n'.join(t)

    start_date = now

    cd = '/'.join([str(start_date.year).zfill(4),
                   str(start_date.month).zfill(2),
                   str(start_date.day).zfill(2)]) + '/'

    c1_d = get(c1 + cd)  # Level-2 data listings for today
    lines = [x.strip() for x in c1_d.strip().split('\n')]
    for line in lines:
        if line == '':
            continue
        w = line.split()
        tile_id = w[3].split('_')[5]
        if tile_id in gids:
            print('  ' + line)
            ofn = 'L2_' + tile_id + sep # + w[3].strip()
            cmd = ' ' .join(['aws s3 cp',
                             '--no-sign-request',
                             's3://sentinel-products-ca-mirror/Sentinel-2/S2MSI1C/' + cd + w[3].strip(),
                             ofn]) # 'L2_' + tile_id + sep + w[3].strip()])
            if not os.path.exists(ofn + w[3].strip()):
                regen_dates.add(year + month + day)
                print(cmd)
                cmds += [cmd] 
                aws_download('sentinel-products-ca-mirror',
                             'Sentinel-2/S2MSI1C/' + cd + w[3].strip(),
                             Path(ofn + w[3].strip())) 
            else:
                print("  exists:", ofn + w[3].strip())

