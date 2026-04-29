'''20250603 sync_recent.py: sync sentinel2 data from NRCAN AWS mirror from today ( or yesterday, or back N days )

20260429: Fixed NoneType comparison crash when no *cloudfree.bin files exist (new pipeline
          produces *.bin not *_cloudfree.bin). Also updated MRAP file detection to find
          both old (*_cloudfree.bin_MRAP.bin) and new (*_MRAP.bin) naming conventions.

          Added --L1 / --L2 flags and auto-detection:
            - If only L1_* folders exist in pwd → L1 mode
            - If only L2_* folders exist in pwd → L2 mode (default)
            - If both exist → require explicit --L1 or --L2 flag
            - --L1 syncs from S2MSI1C, writes to L1_TXXXXX/ folders
            - --L2 syncs from S2MSI2A, writes to L2_TXXXXX/ folders
            - --n N overrides the automatic lookback (default: auto or 1)
'''
import os
import sys
import datetime
from gid import bc
from pathlib import Path
from datetime import timedelta
from aws_download import aws_download
from misc import err, exist, parfor, sep, assert_aws_cli_installed
# assert_aws_cli_installed()

# ---------------------------------------------------------------------------
# Parse --L1, --L2, --n flags
# ---------------------------------------------------------------------------
argv = list(sys.argv)

flag_L1 = '--L1' in argv
flag_L2 = '--L2' in argv
if flag_L1:
    argv.remove('--L1')
if flag_L2:
    argv.remove('--L2')

user_N = None
if '--n' in argv:
    idx = argv.index('--n')
    user_N = int(argv[idx + 1])
    argv.pop(idx)
    argv.pop(idx)  # pop the value too

if flag_L1 and flag_L2:
    err('Cannot specify both --L1 and --L2')

# ---------------------------------------------------------------------------
# Detect which folder types exist in pwd
# ---------------------------------------------------------------------------
has_L1_folders = os.popen("ls -d1 L1_* 2>/dev/null").read().strip() != ''
has_L2_folders = os.popen("ls -d1 L2_* 2>/dev/null").read().strip() != ''

if flag_L1:
    L_mode = 'L1'
elif flag_L2:
    L_mode = 'L2'
elif has_L1_folders and has_L2_folders:
    err('Both L1_* and L2_* folders found in pwd. Please specify --L1 or --L2.')
elif has_L1_folders:
    L_mode = 'L1'
    print('[AUTO] Detected L1_* folders — using L1 mode')
elif has_L2_folders:
    L_mode = 'L2'
    print('[AUTO] Detected L2_* folders — using L2 mode')
else:
    L_mode = 'L2'  # default when no folders exist yet
    print('[AUTO] No L1_*/L2_* folders found — defaulting to L2 mode')

# ---------------------------------------------------------------------------
# Set level-dependent constants
# ---------------------------------------------------------------------------
L_prefix = L_mode + '_'                          # 'L1_' or 'L2_'
if L_mode == 'L1':
    S2_product = 'S2MSI1C'                       # AWS S3 bucket subfolder
    MSI_tag    = 'MSIL1C'                        # filename substring in zips
else:
    S2_product = 'S2MSI2A'
    MSI_tag    = 'MSIL2A'

print(f'Level mode: {L_mode}  (S3 product: {S2_product}, filename tag: {MSI_tag})')

# ---------------------------------------------------------------------------
# GID setup
# ---------------------------------------------------------------------------
bc_gid = bc()
print("bc row-id under obs:", bc_gid)

gids = bc_gid  # default to BC gids

# today's date
today = datetime.date.today()
N = 1  # default lookback

# ---------------------------------------------------------------------------
# Check if we're in an MRAP folder — restrict GIDs to those present on disk
# and compute lookback from most-recent data dates
# ---------------------------------------------------------------------------
L_folders = os.popen("ls -d1 " + L_prefix + "* 2>/dev/null").read().strip()
if L_folders != '':
    min_most_recent = None
    lines = [x.strip().strip(sep) for x in L_folders.split('\n')]
    gids = [line.split('_')[1] for line in lines]
    for gid in gids:
        if gid not in bc_gid:
            err('unexpected gid')  # needs to be modified for other jurisdictions

        # check latest date available, this gid
        def check_pattern(pattern):
            L = [x.strip() for x in os.popen("ls -1 " + L_prefix + gid + sep + pattern).readlines()]
            print("L", [L])
            L = [x.rstrip(":") for x in L]
            L2 = []
            for x in L:
                if x != '':
                    L2 += [x]
            L = L2
            dates = [[x.split(sep)[1].split('_')[2][:8], x] for x in L]
            dates.sort()
            for d in dates:
                print([d])
            try:
                return dates[-1][0]   # most recent date, this pattern
            except:
                return None

        most_recent_bin = check_pattern("*cloudfree.bin")
        most_recent_zip = check_pattern("*.zip")

        if most_recent_bin is None:
            most_recent_bin = most_recent_zip

        if most_recent_zip is None:
            most_recent_zip = most_recent_bin

        if most_recent_zip is not None:
            most_recent = most_recent_bin if most_recent_bin > most_recent_zip else most_recent_zip
            print(gid, "most_recent", most_recent)
            if min_most_recent is None:
                min_most_recent = most_recent
            else:
                if most_recent < min_most_recent:
                    min_most_recent = most_recent

    print("min_most_recent", min_most_recent)

    if min_most_recent is not None:
        # Convert to datetime objects
        today_s = ''.join([str(today.year).zfill(4), str(today.month).zfill(2), str(today.day).zfill(2)])
        date1 = datetime.datetime.strptime(today_s, "%Y%m%d")
        date2 = datetime.datetime.strptime(min_most_recent, "%Y%m%d")

        # Calculate the difference in days
        diff = (date1 - date2).days
        print(f"Difference in days: {diff} -- look back this many days")
        N = diff

# Override N if user specified --n
if user_N is not None:
    N = user_N
    print(f"Lookback overridden by --n: {N} days")

print(gids)

# ---------------------------------------------------------------------------
# Sync from AWS
# ---------------------------------------------------------------------------
s3_base = 's3://sentinel-products-ca-mirror/Sentinel-2/'
ls_cmd  = 'aws s3 ls --no-sign-request'

cmds = []
regen_dates = set()
for i in range(0, N + 1):
    print("i", i, "gids", gids)
    now = today - timedelta(days=i)
    year  = str(now.year).zfill(4)
    month = str(now.month).zfill(2)
    day   = str(now.day).zfill(2)

    print([year, month, day])

    cd = '/'.join([year, month, day]) + '/'

    def get(c):  # collect results from cli invocation
        print(c)
        t = [x.strip() for x in os.popen(c).read().strip().split('\n')]
        return '\n'.join(t)

    # List the appropriate product level on S3
    listing_cmd = ls_cmd + ' ' + s3_base + S2_product + '/' + cd
    listing = get(listing_cmd)
    lines = [x.strip() for x in listing.strip().split('\n')]
    for line in lines:
        if line == '':
            continue
        w = line.split()
        if len(w) < 4:
            continue
        tile_id = w[3].split('_')[5]
        if tile_id in gids:
            print('  ' + line)
            ofn = L_prefix + tile_id + sep
            s3_key = 'Sentinel-2/' + S2_product + '/' + cd + w[3].strip()
            local_path = ofn + w[3].strip()
            if not os.path.exists(local_path):
                regen_dates.add(year + month + day)
                cmd = ' '.join(['aws s3 cp',
                                '--no-sign-request',
                                s3_base + S2_product + '/' + cd + w[3].strip(),
                                ofn])
                print(cmd)
                cmds += [cmd]
                aws_download('sentinel-products-ca-mirror',
                             s3_key,
                             Path(local_path))
            else:
                print("  exists:", local_path)

'''
def runc(c):
    print([c])
    return os.system(c)
parfor(runc, cmds, 2)  # min(int(4), 2 * int(mp.cpu_count())))
'''
if len(cmds) == 0:
    print("All files up to date")

# ---------------------------------------------------------------------------
# Check for zips without corresponding MRAP files.
# Supports both old naming (*_cloudfree.bin_MRAP.bin) and new (*_MRAP.bin).
# ---------------------------------------------------------------------------
print("checking for zips without MRAP files...")
lines = os.popen('find ./ -name "S2*' + MSI_tag + '*.zip"').readlines()
for line in lines:
    line = line.strip()
    stem = line[:-4]  # strip .zip
    date_s = stem.split('_')[-5].split('T')[0]

    # Check for MRAP file under either naming convention
    old_mrap = stem + '_cloudfree.bin_MRAP.bin'
    new_mrap = stem + '_MRAP.bin'
    if not os.path.exists(old_mrap) and not os.path.exists(new_mrap):
        print('zip without MRAP file:', line.strip())
        regen_dates.add(date_s)

print("mrap mosaic dates to regenerate:")
print(list(regen_dates))
regen_dates = list(regen_dates)

def min_d(dates):
    md = dates[0]
    for d in dates:
        if d < md:
            md = d
    return [md]

if len(regen_dates) > 0:
    regen_dates = min_d(regen_dates)  # find earliest date that needs regenerating

    # delete mrap files / folders for dates equal or greater to then..
    mrap_files = os.popen("ls -1 *_mrap.bin 2>/dev/null").read().strip().split('\n')

    to_delete = []
    for f in mrap_files:
        if f == '':
            continue
        ds = f.split('_')[0]

        if ds >= regen_dates[0]:
            print(ds, regen_dates[0])
            to_delete += [ds]

    print("mrap dates to delete:")
    print(to_delete)

    for d in to_delete:
        files_d = [d,
                   d + '_mrap.bin',
                   d + '_mrap.hdr',
                   'small_' + d + 'tar.gz']
        cmd = 'rm -rf ' + ' '.join(files_d)
        print(cmd)
        a = os.system(cmd)
