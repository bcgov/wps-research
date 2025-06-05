'''20250603 sync_recent.py: sync sentinel2 data from NRCAN AWS mirror from today ( or yesterday, or back N days )

need to add options for --L1, --L2 ( default ), --n ( number of days to go back, default 1 )

if we're in an MRAP folder ( as evidenced by 
'''
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
L2_folders = os.popen("ls -d1 L2_*").read().strip()
if L2_folders != '':
    min_most_recent = None
    lines = [x.strip().strip(sep) for x in L2_folders.split('\n')]
    gids = [line.split('_')[1] for line in lines]
    for gid in gids:
        if gid not in bc_gid:
            err('unexpected gid')  # needs to be modified for other jurisdictions
        # check latest date available, this gid
        def check_pattern(pattern):
            L = [x.strip() for x in os.popen("ls -1 L2_" + gid + sep + pattern).readlines()]
            print("L", [L])
            L = [x.rstrip(":") for x in L]
            L2 = []
            for x in L:
                # 'L2_T10VFL/S2A_MSIL2A_20250526T191831_N0511_R056_T10VFL_20250526T222916_cloudfree.bin',
                if x != '':
                    L2 += [x]
            L = L2
            dates = [[x.split(sep)[1].split('_')[2][:8], x] for x in L]
            dates.sort()
            for d in dates:
                print([d])
            # print("most_recent_this_pattern", dates[-1])
            return dates[-1][0]   # most recent date, this pattern
        most_recent_bin = check_pattern("*cloudfree.bin")
        most_recent_zip = check_pattern("*.zip")
        most_recent = most_recent_bin if most_recent_bin > most_recent_zip else most_recent_zip
        print(gid, "most_recent", most_recent)
        if (min_most_recent is None):
            min_most_recent = most_recent
        else:
            if most_recent < min_most_recent:
                min_most_recent = most_recent
            
    print("min_most_recent", min_most_recent)

    # Convert to datetime objects
    today_s = ''.join([str(today.year).zfill(4), str(today.month).zfill(2), str(today.day).zfill(2)])
    date1 = datetime.datetime.strptime(today_s, "%Y%m%d")
    date2 = datetime.datetime.strptime(min_most_recent, "%Y%m%d")

    # Calculate the difference in days
    diff = (date1 - date2).days
    print(f"Difference in days: {diff} -- look back this many days")
    N = diff 
print(gids)

cmds = []
regen_dates = set()
for i in range(0, N + 1):
    print("i", i, "gids", gids)
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
        if line == '':
            continue
        w = line.split()
        tile_id = w[3].split('_')[5]
        if tile_id in gids:
            print('  ' + line)
            ofn = 'L2_' + tile_id + sep # + w[3].strip()
            cmd = ' ' .join(['aws s3 cp',
                             '--no-sign-request',
                             's3://sentinel-products-ca-mirror/Sentinel-2/S2MSI2A/' + cd + w[3].strip(),
                             ofn]) # 'L2_' + tile_id + sep + w[3].strip()])
            if not os.path.exists(ofn + w[3].strip()):
                regen_dates.add(year + month + day)
                print(cmd)
                cmds += [cmd] 
                aws_download('sentinel-products-ca-mirror',
                             'Sentinel-2/S2MSI2A/' + cd + w[3].strip(),
                             Path(ofn + w[3].strip())) 
            else:
                print("  exists:", ofn + w[3].strip())

'''
def runc(c):
    print([c])
    return os.system(c)
parfor(runc, cmds, 2)  # min(int(4), 2 * int(mp.cpu_count())))
'''
if len(cmds) == 0:
    print("All files up to date")

print("don't forget to run sentinel2_extract_cloudfree_swir_nir.py")
print("mrap mosaic dates to regenerate:")
print(list(regen_dates))

lines = os.popen('find ./ -name "S2*.zip"').readlines()
for line in lines:
    line = line.strip()
    f = line[:-3] + '_cloudfree.bin_MRAP.bin'
    print(f)
    if not os.path.exists(f):
        print('zip without MRAP file:', line.strip())

'''
aws s3 sync --no-sign-request s3://sentinel-products-ca-mirror/Sentinel-2/S2MSI2A/2025/06/02/S2C_MSIL2A_20250602T193921_N0511_R042_T10VDK_20250602T224815.zip L2_T10VDK/
'''
