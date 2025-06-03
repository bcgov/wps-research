'''20250603 sync_recent.py: sync sentinel2 data from NRCAN AWS mirror from today ( or yesterday, or back N days )

need to add options for --L1, --L2 ( default ), --n ( number of days to go back, default 1 )

if we're in an MRAP folder ( as evidenced by 
'''
import os
import sys
import datetime
from datetime import timedelta
from gid import bc
from misc import err, sep, assert_aws_cli_installed
assert_aws_cli_installed()
bc_gid = bc()
print("bc row-id under obs:", bc_gid)

gids = bc_gid # default to BC gids a

# check if we're in an MRAP folder, only update the GID present in the filesystem structure:
L2_folders = os.popen("ls -d1 L2_*/").read().strip()
if L2_folders != '':
    lines = [x.strip().strip(sep) for x in L2_folders.split('\n')]
    gids = [line.split('_')[1] for line in lines]
    for gid in gids:
        if gid not in bc_gid:
            err('unexpected gid')  # needs to be modified for other jurisdictions
        # check latest date available, this gid
        def check_pattern(pattern):
            L = [x.strip() for x in os.popen("ls L2_" + gid + sep + pattern).readlines()]
            print(L)
            for x in L:
                # 'L2_T10VFL/S2A_MSIL2A_20250526T191831_N0511_R056_T10VFL_20250526T222916_cloudfree.bin',
                pass
            dates = [[x.split(sep)[1].split('_')[2][:8], x] for x in L]
            dates.sort()
            for d in dates:
                print(d)
            print("most_recent_this_pattern", dates[-1])
            return dates[-1][0]   # most recent date, this pattern
        most_recent_bin = check_pattern("*cloudfree.bin")
        most_recent_zip = check_pattern("*.zip")
        most_recent = most_recent_bin if most_recent_bin > most_recent_zip else most_recent_zip
        print("most_recent", most_recent)
print(gids)

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
