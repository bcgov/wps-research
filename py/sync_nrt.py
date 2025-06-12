'''20250607: sync_nrt.py: pull today's L1 data. Select GID based on what GID are present in current folder ( L1 or L2 ) 

Then, call merge2.py on those date.

20250611: updated to pull for specified date (yyyymmdd)

based on:  20250603 sync_recent.py: sync sentinel2 data from NRCAN AWS mirror from today ( or yesterday, or back N days )

need to add options for --L1, --L2 ( default ), --n ( number of days to go back, default 1 )'''
import os
import sys
import datetime
from gid import bc
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import timedelta
from aws_download import aws_download
from misc import args, err, parfor, sep, assert_aws_cli_installed
# assert_aws_cli_installed()
bc_gid = bc()
print("bc row-id under obs:", bc_gid)
gids = bc_gid # default to BC gids a
# today = datetime.datetime.now(ZoneInfo("America/Vancouver")) 

if len(args) < 2:
    err('sync_nrt.py [yyyymmdd]')

now = sys.argv[1]
if len(now) != 8:
    err("expected date in format yyyymmdd")
try:
    now_int = int(now)
except:
    err("expected date in format yyyymmdd")

# today
year, month, day = now[0:4], now[4:6], now[6:8]

# check if we're in an MRAP folder, only update the GID present in the filesystem structure:
L1_folders = os.popen("ls -d1 L1_*").read().strip().split('\n')
L2_folders = os.popen("ls -d1 L2_*").read().strip().split('\n')
folders = set(L1_folders + L2_folders)
if '' in folders:
    folders.remove('')  # make sure there's no empty string
gids = [line.split('_')[1] for line in folders]
gids = list(set(gids))

if len(gids) == 0:
    gids = bc_gid

files = []
if True:
    # today's date
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

    now = datetime.datetime(int(year), int(month), int(day))
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
            ofn = 'L1_' + tile_id + sep # + w[3].strip()
            cmd = ' ' .join(['aws s3 cp',
                             '--no-sign-request',
                             's3://sentinel-products-ca-mirror/Sentinel-2/S2MSI1C/' + cd + w[3].strip(),
                             ofn]) # 'L2_' + tile_id + sep + w[3].strip()])
            if not os.path.exists(ofn + w[3].strip()):
                print(cmd)
                aws_download('sentinel-products-ca-mirror',
                             'Sentinel-2/S2MSI1C/' + cd + w[3].strip(),
                             Path(ofn + w[3].strip())) 
            else:
                print("  exists:", ofn + w[3].strip())
    
            files += [ofn + w[3].strip()]

def extr(i):
    return os.system("sentinel2_extract_swir.py " + i)

parfor(extr, files)

if len(args) < 2:
    err('sync_nrt.py [yyyymmdd]')

now = sys.argv[1]
if len(now) != 8: 
    err("expected date in format yyyymmdd")
try:
    now_int = int(now)
except:
    err("expected date in format yyyymmdd")

# today
year, month, day = now[0:4], now[4:6], now[6:8] #str(now.year).zfill(4), str(now.month).zfill(2), str(now.day).zfill(2)
outfile =  year + month + day + ".bin"
cmd = "merge3.py " + ' '.join([x[:-3] + 'bin' for x in files]) + ' ' + outfile
print(cmd)
a = os.system(cmd) 
print("done")
