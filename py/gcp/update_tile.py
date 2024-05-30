'''search for Sentinel2 frames: parameters:
    * given TILE_ID
    * max cloud cover
    * date range

And download from Google Cloud Platform (GCP).

The "gsutil rsync" command is helpful for avoiding re-downloading the same data
    https://cloud.google.com/storage/docs/gsutil/commands/rsync

(*) need concurrency mechanism so we can download on multiple
folders without fetching index twice!


Note: should update this to include a higher-level folder "by tile ID"

Note: for "error while loading shared libraries: libgeos_c.so.1: cannot open shared object file"

try:
    sudo apt remove libgeos-c1v5
    sudo apt install libgeos-c1v5
'''
import os
import sys
sep = os.path.sep
from datetime import date
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + "..")
from misc import run, pd, sep, exists, args, cd

if True:
    print(' '.join(['python3 update_tile.py',
                   '[TILE_ID e.g. T10UFB]',
                   '[max cloud cover]',
                   '[min date yyyymmdd]',
                   '[max date yyyymmdd]',
                   '[optional arg: no download!]']))
if len(args) < 5:
    sys.exit(1)

TILE_ID = 'T09VUE' if len(args) < 2 else args[1]
MAX_CLOUD = 22.2 if len(args) < 3 else args[2]  # max percent cloud cover to consider!
MIN_DATE = None if len(args) < 4 else args[3]
MAX_DATE = None if len(args) < 5 else args[4]
DOWNLOAD = len(args) < 6 # skip the download step if an extra arg added!
print("TILE_ID", TILE_ID)
print("MAX_CLOUD", MAX_CLOUD)
print("MIN_DATE", MIN_DATE)
print("MAX_DATE", MAX_DATE)
MAX_CLOUD = float(MAX_CLOUD) # convert to float
MIN_DATE = '20150101' if MIN_DATE is None else int(MIN_DATE) 
td = date.today()
tds = [str(int(x)) for x in [td.year, td.month, td.day]]
tds[0] = tds[0].zfill(4)
for i in [1,2]:
    tds[i] = tds[i].zfill(2)
tds = ''.join(tds)
# print(tds)
MAX_DATE = int(tds) if MAX_DATE is None else int(MAX_DATE)

dp = os.path.abspath(pd + 'gcp' + sep) + sep  # local directory path
ifn = dp + 'index.csv.gz'
ip = dp + 'index' + sep
iff = ip + 'index.csv'

# print('+r', ifn)
#  print(os.popen('ls -1 ' + ifn).read().strip())

if not exists(ifn):  # -N: timestamped: don't re-download!
    run(' '.join(['wget',
                  '-N',
                  'https://storage.googleapis.com/gcp-public-data-sentinel-2/index.csv.gz',
                  '-O',
                  ifn]))
    # print('+w', ifn)

if not exists(ip):
    # print('mkdir', ip)
    os.mkdir(ip)

if not exists(iff): # ip + 'index.csv'):  # unzip the index data
    run('gunzip ' + ifn + ' --force --keep') # index.csv.gz')
    run('mv -v ' + dp + 'index.csv ' + iff);

csv_split = cd + 'csv_split.exe'

if not exists(csv_split): # a'csv_split.exe'):  # split the index data into columnar format
    run(' '.join(['g++',
                  '-O3',
                  cd + 'misc.cpp',
                  cd + 'csv_split.cpp',
                  '-o',
                  cd + 'csv_split.exe']))

if not exists(ip): # destination folder for extracting the index
    os.mkdir(ip)

if not exists(ip + 'index.csv:PRODUCT_ID'):  # only split if the target doesn't exist
    run(csv_split + ' ' + ip + 'index.csv')

matches = []

'''this csv_split variation has a header. -n returns 1-index'''
cmd = ['grep',
       '-n',
       '_' + TILE_ID + '_',
       ip + 'index.csv:PRODUCT_ID']
cmd = ' '.join(cmd)
print(cmd)
lines = [x.strip() for x in os.popen(cmd).readlines()]

for i in range(len(lines)): # match the records of interest 
    try:
        line_i, pid = lines[i].split(':')
        if len(pid.split(TILE_ID)) > 1:
            matches.append([int(line_i), pid])
    except:
        print('Error: line:', i, 'content:', lines[i])
        sys.exit(1)

line_idx = [m[0] - 1 for m in matches] # 1-index of record, accounting for header
# for m in matches: print(m)

product_id = [x.strip() for x in open(ip + 'index.csv:PRODUCT_ID').readlines()]

''' filter matches by cloud cover'''
line_idx_match = []  # records matching our criteria
cloud_covers = [x.strip() for x in open(ip + 'index.csv:CLOUD_COVER').readlines()]
for i in range(len(line_idx)):
    j = line_idx[i]
    if MAX_CLOUD == 100.:
        line_idx.match += [j]
    else:
        cloud_cover = float(cloud_covers[j]) # print(j, cloud_cover, product_id[j])
        if cloud_cover <= MAX_CLOUD:
            line_idx_match.append(j) # print('\t', j, cloud_cover, product_id[j])

for i in range(len(line_idx_match)):
    j = line_idx_match[i]  #print(j, cloud_covers[j], product_id[j])
print("n_matches:", len(line_idx_match))

matches = [[int(product_id[j].split('_')[2].split('T')[0]),
            j,
            cloud_covers[j],
            product_id[j]] for j in line_idx_match] 
matches.sort()  # sort on date string

print("all time matches:", len(matches))
# filter by date
matches_new = []
for m in matches:
    use = True
    if MIN_DATE is not None:
        if m[0] < int(MIN_DATE):
            use = False
    if MAX_DATE is not None:
        if m[0] > int(MAX_DATE):
            use = False
    if use:
        matches_new.append(m)
matches =  matches_new
print("matches in time window:", len(matches))

if not DOWNLOAD:
    sys.exit(1)
else:
    print("downloading..")

base_url = [x.strip() for x in open(ip + 'index.csv:BASE_URL').readlines()]
for m in matches:
    print(m, base_url[m[1]])
    
    # download L1 dataset from GCP
    cmd = ' '.join(['gsutil -m',
                    'rsync -r',
                    base_url[m[1]],
                    './',
                    '>',
                    m[3] + '.SAFE_stdout.txt',
                    '2>',
                    m[3] +  '.SAFE_stderr.txt'])
    out_dir = './' + m[3] + '.SAFE'
    
    if not exists(out_dir):
        print('mkdir', out_dir)
        a = run('mkdir -p ' + out_dir)
    else:
        print('Presumed already downloaded.. skipping')
        continue # don't redownload files 

    simple_cmd = ['gsutil -m',
                  'rsync -r',
                  base_url[m[1]],
                  out_dir]
    a = run(' '.join(simple_cmd))

    # "fix" the output by adding empty directories expected by ESA SNAP
    run(['python3',
         pd + 'gcp' + sep + 'fix_s2.py',
         out_dir])

'''don't forget what fixedwidth format was for, this is inefficient'''
