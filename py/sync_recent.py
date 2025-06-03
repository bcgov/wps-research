'''20250603 sync_recent.py: sync sentinel2 data from NRCAN AWS mirror from today ( or yesterday, or back N days )
'''
import os
import sys
import datetime
from gid import bc
from misc import sep
bc_gid = bc()
print("bc row-id under obs:", bc_gid)

# check that aws cli installed
if len(os.popen("aws 2>&1").read().split("not found")) > 1:
    print('Need to install aws cli: e.g.:')
    print('  sudo apt install awscli')
    sys.exit(1)

now = datetime.date.today()
year, month, day = str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2)

print([year, month, day])
ls = 'aws s3 ls --no-sign-request'
path = 's3://sentinel-products-ca-mirror/Sentinel-2/'
c1, c2 = ' '.join([ls, path + 'S2MSI1C/']), ' '.join([ls, path + 'S2MSI2A/'])
c1 = ls + a + s2 + 'S2MSI1C/' # + '/'.join([year, month, day]) + '/' 
c2 = ls + a + s2 + 'S2MSI2A/' # + '/'.join([year, month, day]) + '/'

def get(c):
    print(c)
    t = [x.strip() for x in os.popen(c).read().strip().split('\n')]
    return '\n'.join(t)

start_date = now

cd = '/'.join([str(start_date.year).zfill(4),
               str(start_date.month).zfill(2),
               str(start_date.day).zfill(2)]) + '/'

c1_d = get(c1 + cd)  # Level-1 data listings

