'''20250616 list available sentinel-2 frames, for a given date, based on selected GID ( default to alll of BC ) 
Retrieve data listing by date, e.g. to count frames on day X'''
import os
import sys
import datetime
from gid import bc
gids = []
args = sys.argv
exist = os.path.exists
def err(m):
    print("Error:", m); sys,exit(1)

def run(c):
    print(c)
    return os.popen(c).read().strip()

if len(args) < 2 or len(args[1]) != 8:
    err("sentinel2_list_date.py [yyyymmdd]")

if len(args) < 3:
    gids = bc() 
else:
    gids = args[2:]


if len(os.popen("aws 2>&1").read().split("not found")) > 1:
    err('please install aws cli, e.g. on linux:\n\tsudo apt install awscli')

now = datetime.date.today()
year, month, day = str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2)
year = args[1][0:4]
month = args[1][4:6]
day = args[1][6:8]

print([year, month, day])
ls = 'aws s3 ls --no-sign-request'
path = 's3://sentinel-products-ca-mirror/Sentinel-2/'
c1, c2 = ' '.join([ls, path + 'S2MSI1C/']), ' '.join([ls, path + 'S2MSI2A/'])
c1 = c1 + year + '/' + month + '/' + day + '/'
c2 = c2 + year + '/' + month + '/' + day + '/'

L1_files = [x.strip() for x in run(c1).split('\n')]
L2_files = [x.strip() for x in run(c2).split('\n')]

L1_files_select = []
L2_files_select = []

for x in L1_files:
    try:
        g = x.split()[-1].split('_')[5]
        if g in gids:
            print(x)
    except:
        pass

for x in L2_files:
    try:
        g = x.split()[-1].split('_')[5]
        if g in gids:
            print(x)
    except:
        pass

# 2025-06-16 17:47:40  148896246 S2C_MSIL2A_20250616T173921_N0511_R098_T15VWE_20250616T225811.zip

