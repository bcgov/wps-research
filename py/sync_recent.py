'''20250603 sync_recent.py: sync sentinel2 data from NRCAN AWS mirror from today ( or yesterday, or back N days )

need to add options for --L1, --L2 ( default ), --n ( number of days to go back, default 1 )
'''
import os
import sys
import datetime
from datetime import timedelta
from gid import bc
from misc import sep, assert_aws_cli_installed
assert_aws_cli_installed()
bc_gid = bc()
print("bc row-id under obs:", bc_gid)

gids = bc_gid # default to BC gids 
# today's date
today = datetime.date.today()
N = 1
for i in range(0, N + 1):
    print("i", i)
    now = today - timedelta(days=i)
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

    c1_d = get(c2 + cd)  # Level-2 data listings for today
    lines = [x.strip() for x in c1_d.strip().split('\n')]
    for line in lines:
        if line == '': continue
        w = line.split()
        tile_id = w[3].split('_')[5]
        if tile_id in gids:
            print(line)
