'''20260603: GCP variant of sync_daterange_gid_zip.py

Downloads Sentinel-2 L1C data from Google Cloud Platform (public bucket,
no credentials required). If --L2 is requested, runs Sen2Cor locally to
produce L2A, since GCP only stores L1C.

Usage:
    python3 sync_daterange_gid_zip_gcp.py [yyyymmdd] [yyyymmdd2] [GID ...] [flags]

Flags:
    --L1            L1C mode (default)
    --L2            download L1C + run Sen2Cor -> L2A zip
    --parallel      enable 16-worker parallel downloads / sen2cor
    --force-listing force refresh of GCP index even if a fresh copy exists

Examples:
    python3 sync_daterange_gid_zip_gcp.py 20260101
    python3 sync_daterange_gid_zip_gcp.py 20260101 20260131
    python3 sync_daterange_gid_zip_gcp.py 20260101 20260131 T10UGU T10UGV
    python3 sync_daterange_gid_zip_gcp.py 20260101 20260131 --L2 --parallel
    python3 sync_daterange_gid_zip_gcp.py 20260101 20260131 all

Workflow per dataset:
    L1 mode:  gsutil rsync -> fix_s2 -> zip (keep .SAFE)
    L2 mode:  gsutil rsync -> fix_s2 -> zip L1C -> sen2cor -> zip L2A (keep both .SAFEs)
              skip if completed zip already present at any stage

Step 6 NOTE: .SAFE downloads are currently ACTIVE. Run one tile serially
             to verify before enabling --parallel.
'''

import os
import sys
import gzip
import csv
import time
import shutil
import datetime
import zipfile

sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path)
sys.path.append(my_path + '..')

from misc import args, sep, exists, parfor, run, timestamp, err

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

GCP_INSTALL_DIR  = os.path.expanduser('~/GitHub/wps-research/py/gcp/install')
GCP_INDEX_URL    = 'https://storage.googleapis.com/gcp-public-data-sentinel-2/index.csv.gz'
GCP_INDEX_PREFIX = 'gcp_index_'   # distinguishes GCP listings from AWS listings in same folder

# Google Cloud SDK
SDK_VERSION = '472.0.0'
SDK_TARBALL = f'google-cloud-cli-{SDK_VERSION}-linux-x86_64.tar.gz'
SDK_URL     = ('https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/' + SDK_TARBALL)

# Sen2Cor standalone binary (Linux x86_64); no compilation required
# Update version + URL when ESA releases a new one:
#   https://step.esa.int/main/snap-supported-plugins/sen2cor/
SEN2COR_VERSION   = '02.11.00'
SEN2COR_INSTALLER = f'Sen2Cor-{SEN2COR_VERSION}-Linux64.run'
SEN2COR_URL       = (f'https://step.esa.int/thirdparties/sen2cor/{SEN2COR_VERSION}/'
                     + SEN2COR_INSTALLER)

N_WORKERS = 16   # parallel worker count when --parallel is active

# ---------------------------------------------------------------------------
# Listing directory: shared with the AWS sync script
# ---------------------------------------------------------------------------
_base0 = '/data/'
_base1 = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
_listing_base = _base1
try:
    if not exists(_base0):
        print('mkdir', _base0)
        os.mkdir(_base0)
    _listing_base = _base0
except Exception:
    _fb = _base1 + '.listing'
    if not exists(_fb):
        os.mkdir(_fb)

LISTING_DIR = _listing_base + '.listing' + sep

# ---------------------------------------------------------------------------
# Global progress counters
# ---------------------------------------------------------------------------
_download_start_time = None
_files_completed = 0
_total_files     = 0
_bytes_completed = 0
_total_bytes     = 0


# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def md(path):
    if not exists(path):
        print('mkdir', path)
        os.makedirs(path, exist_ok=True)


def today_str():
    td = datetime.date.today()
    return f'{td.year:04d}{td.month:02d}{td.day:02d}'


def is_date_format(s):
    return len(s) == 8 and s.isdigit()


def print_status_update(file_name, file_size, file_dl_time):
    global _files_completed, _total_files, _bytes_completed, _total_bytes, _download_start_time
    elapsed = time.time() - _download_start_time
    pct_f   = (_files_completed / _total_files  * 100) if _total_files  > 0 else 0
    pct_b   = (_bytes_completed / _total_bytes  * 100) if _total_bytes  > 0 else 0
    eta     = (elapsed / _files_completed * (_total_files - _files_completed)
               if _files_completed > 0 else 0)
    print(f"\n{'='*60}")
    print(f"DOWNLOAD STATUS UPDATE")
    print(f"{'='*60}")
    print(f"File completed: {file_name}")
    print(f"File size: {file_size/(1024**2):.2f} MB | Download time: {file_dl_time:.1f}s")
    print(f"{'='*60}")
    print(f"Files:   {_files_completed} / {_total_files} ({pct_f:.1f}%)")
    print(f"Bytes:   {_bytes_completed/(1024**3):.2f} / {_total_bytes/(1024**3):.2f} GB ({pct_b:.1f}%)")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"ETA:     {eta/60:.1f} min  |  {eta/3600:.2f} hr  |  {eta/86400:.3f} days")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Step 1+2: gsutil install
# ---------------------------------------------------------------------------

def find_gsutil():
    p = os.popen('which gsutil 2>/dev/null').read().strip()
    if p and exists(p):
        return p
    candidate = os.path.join(GCP_INSTALL_DIR, 'google-cloud-sdk', 'bin', 'gsutil')
    if exists(candidate):
        return candidate
    return None


def ensure_gsutil():
    gsutil = find_gsutil()
    if gsutil:
        print(f'gsutil found: {gsutil}')
        _ensure_crcmod()
        return gsutil

    print('gsutil not found — installing Google Cloud SDK...')
    md(GCP_INSTALL_DIR)

    local_tar = os.path.join(GCP_INSTALL_DIR, SDK_TARBALL)
    if not exists(local_tar):
        print(f'Downloading SDK: {SDK_URL}')
        run(f'wget -q --show-progress -O {local_tar} {SDK_URL}')
    else:
        print(f'SDK tarball cached: {local_tar}')

    sdk_dir = os.path.join(GCP_INSTALL_DIR, 'google-cloud-sdk')
    if not exists(sdk_dir):
        print('Extracting SDK...')
        run(f'tar -xzf {local_tar} -C {GCP_INSTALL_DIR}')
    else:
        print(f'SDK already extracted: {sdk_dir}')

    install_sh = os.path.join(sdk_dir, 'install.sh')
    run(f'bash {install_sh} --quiet --usage-reporting=false --path-update=true')

    gsutil = os.path.join(sdk_dir, 'bin', 'gsutil')
    if not exists(gsutil):
        err(f'gsutil not found after install — expected: {gsutil}')

    _ensure_crcmod()
    print(f'gsutil installed: {gsutil}')

    # Note for future reference (public bucket needs no credentials)
    cred_note = os.path.join(GCP_INSTALL_DIR, 'CREDENTIALS_NOTE.txt')
    if not exists(cred_note):
        with open(cred_note, 'w') as fh:
            fh.write(
                'The public gcp-public-data-sentinel-2 bucket needs NO credentials.\n'
                'For private bucket access run:  gsutil config\n'
                'Credentials land in ~/.config/gcloud/ and ~/.boto\n'
                'Copy those files here to archive them alongside the SDK.\n'
            )
    return gsutil


def _ensure_crcmod():
    try:
        import crcmod  # noqa: F401
        print('crcmod already installed.')
    except ImportError:
        print('Installing crcmod for gsutil performance...')
        run('sudo apt-get install -y gcc python3-dev python3-setuptools 2>/dev/null || true')
        run('pip3 install --no-cache-dir -U crcmod --break-system-packages 2>/dev/null '
            '|| pip3 install --no-cache-dir -U crcmod')


# ---------------------------------------------------------------------------
# Step 3: Sen2Cor install  (self-extracting .run; no compilation)
# ---------------------------------------------------------------------------

def find_sen2cor():
    p = os.popen('which L2A_Process 2>/dev/null').read().strip()
    if p and exists(p):
        return p
    if exists(GCP_INSTALL_DIR):
        for root, dirs, files in os.walk(GCP_INSTALL_DIR):
            if 'L2A_Process' in files:
                return os.path.join(root, 'L2A_Process')
    return None


def ensure_sen2cor():
    l2a = find_sen2cor()
    if l2a:
        print(f'L2A_Process found: {l2a}')
        return l2a

    print('Sen2Cor not found — installing...')
    md(GCP_INSTALL_DIR)

    local_installer = os.path.join(GCP_INSTALL_DIR, SEN2COR_INSTALLER)
    if not exists(local_installer):
        print(f'Downloading Sen2Cor: {SEN2COR_URL}')
        run(f'wget -q --show-progress -O {local_installer} {SEN2COR_URL}')
    else:
        print(f'Sen2Cor installer cached: {local_installer}')

    run(f'bash {local_installer} --target {GCP_INSTALL_DIR}')

    l2a = find_sen2cor()
    if not l2a:
        err(
            f'L2A_Process not found after Sen2Cor install under {GCP_INSTALL_DIR}.\n'
            f'If the ESA URL has changed, update SEN2COR_VERSION/SEN2COR_URL and retry.\n'
            f'Manual: https://step.esa.int/main/snap-supported-plugins/sen2cor/'
        )
    print(f'L2A_Process installed: {l2a}')
    return l2a


# ---------------------------------------------------------------------------
# Step 4: GCP index listing
# ---------------------------------------------------------------------------

def _listing_datestamp(filename):
    '''gcp_index_20260603_143022.csv.gz -> '20260603'; '' on failure.'''
    base = os.path.basename(filename)
    if not base.startswith(GCP_INDEX_PREFIX):
        return ''
    rest = base[len(GCP_INDEX_PREFIX):]
    return rest[:8] if len(rest) >= 8 and rest[:8].isdigit() else ''


def select_listing(end_date_str, force_update):
    '''
    Return path to the .csv.gz to use.

    Refresh logic:
      --force-listing            always refresh
      end_date <= listing_date   historical: listing covers it, reuse
      listing_date < today       stale (prev calendar day), refresh
      listing_date == today      fresh today, reuse
      no listing                 must download
    '''
    md(LISTING_DIR)
    all_gz = sorted(
        [f for f in os.listdir(LISTING_DIR) if f.startswith(GCP_INDEX_PREFIX)],
        reverse=True
    )
    newest      = all_gz[0] if all_gz else None
    newest_path = os.path.join(LISTING_DIR, newest) if newest else None
    newest_date = _listing_datestamp(newest) if newest else ''
    today       = today_str()

    need_dl = False
    if force_update:
        print('--force-listing: refreshing GCP index.')
        need_dl = True
    elif newest is None:
        print('No GCP listing found — downloading fresh copy.')
        need_dl = True
    elif end_date_str <= newest_date:
        print(f'Historical query (end={end_date_str} <= listing={newest_date}): '
              f'reusing {newest}')
    elif newest_date < today:
        print(f'Listing from {newest_date}, today={today}: refreshing.')
        need_dl = True
    else:
        print(f'Listing already fresh from today ({newest_date}): reusing {newest}')

    if need_dl:
        ts       = timestamp()
        new_fn   = GCP_INDEX_PREFIX + ts + '.csv.gz'
        new_path = os.path.join(LISTING_DIR, new_fn)
        print(f'Downloading GCP index -> {new_path}')
        run(f'wget -q --show-progress -O {new_path} {GCP_INDEX_URL}')
        newest_path = new_path
        print(f'Cached: {new_path}')

    return newest_path


def load_index(listing_gz_path):
    '''Stream-parse gzipped CSV. Returns (product_to_url, product_to_cloud).'''
    print(f'Loading GCP index: {listing_gz_path}')
    product_to_url   = {}
    product_to_cloud = {}
    with gzip.open(listing_gz_path, 'rt', encoding='utf-8', errors='replace') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row.get('PRODUCT_ID',  '').strip()
            url = row.get('BASE_URL',    '').strip()
            cc  = row.get('CLOUD_COVER', '').strip()
            if pid and url:
                product_to_url[pid] = url
                try:
                    product_to_cloud[pid] = float(cc)
                except ValueError:
                    product_to_cloud[pid] = 999.0
    print(f'Index loaded: {len(product_to_url):,} products.')
    return product_to_url, product_to_cloud


# ---------------------------------------------------------------------------
# fix_s2 (inline; mirrors fix_s2.py)
# GCP omits empty dirs; Sen2Cor requires AUX_DATA/ and HTML/
# ---------------------------------------------------------------------------

def fix_s2(safe_dir):
    for sub in ('AUX_DATA', 'HTML'):
        d = os.path.join(safe_dir, sub)
        if not exists(d):
            print('mkdir', d)
            os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Zip helpers — explicit paths only, no ambiguity
# ---------------------------------------------------------------------------

def safe_to_zip(safe_dir):
    '''
    Archive  <parent>/<stem>.SAFE/  ->  <parent>/<stem>.zip

    Zip root entry is  <stem>.SAFE/  (not double-nested, no loose files).
    Original .SAFE is NOT deleted.
    Returns path to zip (no-op if already present).
    '''
    safe_dir  = os.path.abspath(safe_dir)
    parent    = os.path.dirname(safe_dir)
    safe_name = os.path.basename(safe_dir)
    if not safe_name.endswith('.SAFE'):
        err(f'safe_to_zip: expected .SAFE path, got: {safe_dir}')
    stem     = safe_name[:-5]
    zip_stem = os.path.join(parent, stem)
    zip_path = zip_stem + '.zip'

    if exists(zip_path):
        print(f'Zip already present: {zip_path}')
        return zip_path

    print(f'Zipping {safe_name} -> {zip_path}')
    # make_archive(base_name, format, root_dir, base_dir)
    #   root_dir = parent     <- archive is created relative to here
    #   base_dir = safe_name  <- this subtree becomes the zip root entry
    shutil.make_archive(zip_stem, 'zip', parent, safe_name)
    print(f'Created: {zip_path}')
    return zip_path


def extract_safe_from_zip(zip_path, target_parent):
    '''
    Extract  <stem>.zip  ->  <target_parent>/<stem>.SAFE/

    Zip was created by safe_to_zip so every member starts with <stem>.SAFE/.
    We extract directly into target_parent: result is target_parent/<stem>.SAFE/...
    No double-nesting, no loose files in cwd.
    Returns path to extracted .SAFE dir.
    '''
    zip_path      = os.path.abspath(zip_path)
    target_parent = os.path.abspath(target_parent)
    stem          = os.path.basename(zip_path)[:-4]   # strip .zip
    safe_name     = stem + '.SAFE'
    out_safe      = os.path.join(target_parent, safe_name)

    if exists(out_safe):
        print(f'Already extracted: {out_safe}')
        return out_safe

    print(f'Extracting {zip_path} -> {target_parent}/')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        bad = [m for m in members
               if not (m == safe_name + '/' or m.startswith(safe_name + '/'))]
        if bad:
            print(f'WARNING: unexpected zip members (first 5): {bad[:5]}')
        # extractall with target_parent -> target_parent/<stem>.SAFE/...
        zf.extractall(target_parent)

    if not exists(out_safe):
        err(f'Expected {out_safe} after extraction — not found.')
    print(f'Extracted: {out_safe}')
    return out_safe


# ---------------------------------------------------------------------------
# Step 5: per-dataset gsutil rsync worker
# ---------------------------------------------------------------------------

def download_safe(item, gsutil_path):
    '''
    Download one .SAFE folder via gsutil rsync -r.
    Resumable by design: reruns skip files already matching checksum.
    Calls fix_s2 on success.
    Returns item dict (pass-through for parfor).
    '''
    global _files_completed, _bytes_completed

    pid       = item['pid']
    base_url  = item['base_url']
    out_dir   = item['out_dir']
    safe_name = item['safe_name']
    safe_path = item['safe_path']

    md(out_dir)

    stdout_log = safe_path + '_stdout.txt'
    stderr_log = safe_path + '_stderr.txt'

    # Trailing slashes on both src and dst are required by gsutil rsync:
    # without them gsutil may see a same-named object and refuse to treat
    # the destination as a directory (CommandException: does not name a directory).
    src = base_url.rstrip('/') + '/'
    dst = safe_path.rstrip('/') + '/'
    cmd = f'{gsutil_path} -m rsync -r {src} {dst}'
    print(f'\n[gsutil rsync] {cmd}')
    t0  = time.time()
    ret = os.system(f'{cmd} > {stdout_log} 2> {stderr_log}')
    dt  = time.time() - t0

    if ret != 0:
        print(f'WARNING: gsutil rsync returned {ret} for {pid} — check {stderr_log}')
    else:
        fix_s2(safe_path)

    safe_bytes = (
        sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(safe_path) for f in fs)
        if exists(safe_path) else 0
    )

    _files_completed += 1
    _bytes_completed += safe_bytes
    print_status_update(safe_name, safe_bytes, dt)
    return item


# ---------------------------------------------------------------------------
# Sen2Cor worker
# ---------------------------------------------------------------------------

def run_sen2cor_job(job, l2a_process):
    '''
    job: (l1c_safe_path, l2a_safe_path, l2a_zip_path, out_dir)

    Runs fix_s2 then L2A_Process on the L1C .SAFE.
    Sen2Cor writes L2A .SAFE alongside L1C .SAFE (same directory).
    If it lands in cwd instead, we relocate it.
    Zips the L2A .SAFE; neither .SAFE is deleted.
    '''
    safe_path, l2a_safe_path, l2a_zip_path, out_dir = job

    if exists(l2a_zip_path):
        print(f'L2A zip already present — skipping: {l2a_zip_path}')
        return

    if not exists(safe_path):
        print(f'L1C SAFE missing — cannot run sen2cor: {safe_path}')
        return

    fix_s2(safe_path)

    cmd = f'{l2a_process} {safe_path}'
    print(f'[sen2cor] {cmd}')
    ret = os.system(cmd)
    if ret != 0:
        print(f'WARNING: sen2cor returned {ret} for {safe_path}')
        return

    # Sen2Cor should write L2A .SAFE into the same directory as L1C .SAFE
    if not exists(l2a_safe_path):
        # Some sen2cor versions write to cwd — check and relocate
        cwd_l2a = os.path.join(os.getcwd(), os.path.basename(l2a_safe_path))
        if exists(cwd_l2a):
            print(f'Relocating sen2cor output: {cwd_l2a} -> {l2a_safe_path}')
            shutil.move(cwd_l2a, l2a_safe_path)
        else:
            print(f'WARNING: L2A SAFE not found after sen2cor: {l2a_safe_path}')
            return

    fix_s2(l2a_safe_path)           # belt-and-suspenders on L2A output
    safe_to_zip(l2a_safe_path)      # .SAFE retained — not deleted


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def download_by_gids(gids, yyyymmdd, yyyymmdd2,
                     use_L2, use_parallel, force_listing,
                     gsutil_path, l2a_process):
    global _files_completed, _total_files, _bytes_completed, _total_bytes, _download_start_time

    _files_completed = 0; _total_files = 0
    _bytes_completed = 0; _total_bytes = 0

    if len(yyyymmdd) != 8 or len(yyyymmdd2) != 8:
        err('expected date in format yyyymmdd')

    start_d = datetime.datetime(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:]))
    end_d   = datetime.datetime(int(yyyymmdd2[:4]), int(yyyymmdd2[4:6]), int(yyyymmdd2[6:]))
    print(f'Date range: {start_d.date()} -> {end_d.date()}')

    date_range = set()
    cur = start_d
    while cur <= end_d:
        date_range.add(f'{cur.year:04d}{cur.month:02d}{cur.day:02d}')
        cur += datetime.timedelta(days=1)

    listing_path = select_listing(yyyymmdd2, force_listing)
    product_to_url, _ = load_index(listing_path)

    # ------------------------------------------------------------------
    # Filter: L1C products matching date range and GID list
    # GCP stores only L1C; we always download L1C regardless of --L2
    # ------------------------------------------------------------------
    out_prefix = 'L2_' if use_L2 else 'L1_'
    work_items = []
    skipped    = 0

    for pid, base_url in product_to_url.items():
        parts = pid.split('_')
        # S2A _ MSIL1C _ 20230525T... _ N0xxx _ R0xx _ T10UGU _ ...
        if len(parts) < 6:
            continue
        if parts[1] != 'MSIL1C':
            continue
        ts_raw = parts[2].split('T')[0]   # '20230525'
        if ts_raw not in date_range:
            continue
        gid = parts[5]                    # e.g. T10UGU
        if gids is not None and gid not in gids:
            continue

        out_dir   = out_prefix + gid
        safe_name = pid + '.SAFE'
        zip_name  = pid + '.zip'
        safe_path = os.path.join(out_dir, safe_name)
        zip_path  = os.path.join(out_dir, zip_name)

        if use_L2:
            l2a_pid      = pid.replace('MSIL1C', 'MSIL2A')
            l2a_zip_path = os.path.join(out_dir, l2a_pid + '.zip')
            if exists(l2a_zip_path):
                print(f'L2A zip present — skipping: {l2a_zip_path}')
                skipped += 1
                continue
        else:
            if exists(zip_path):
                print(f'L1C zip present — skipping: {zip_path}')
                skipped += 1
                continue

        work_items.append({
            'pid':      pid,
            'base_url': base_url,
            'gid':      gid,
            'out_dir':  out_dir,
            'safe_name':safe_name,
            'safe_path':safe_path,
            'zip_path': zip_path,
        })

    print(f'\n{"="*60}')
    print(f'MATCH SUMMARY')
    print(f'{"="*60}')
    print(f'Products to download/process : {len(work_items)}')
    print(f'Already complete (skipped)   : {skipped}')
    print(f'{"="*60}\n')

    if not work_items:
        print('Nothing to do.')
        return

    disk = shutil.disk_usage(os.getcwd())
    rough_gb = len(work_items) * 1.0
    if disk.free < rough_gb * (1024**3):
        print(f'WARNING: {disk.free/(1024**3):.1f} GB free may be insufficient '
              f'for ~{rough_gb:.0f} GB estimated download.')

    _total_files         = len(work_items)
    _download_start_time = time.time()

    # ------------------------------------------------------------------
    # Step 5: download .SAFE folders via gsutil rsync
    # ------------------------------------------------------------------
    print(f'\n--- Downloading {_total_files} .SAFE folder(s) '
          f'[{"parallel" if use_parallel else "serial"}] ---')

    if use_parallel:
        def _dl(item):
            return download_safe(item, gsutil_path)
        parfor(_dl, work_items, N_WORKERS)
    else:
        for item in work_items:
            download_safe(item, gsutil_path)

    # ------------------------------------------------------------------
    # Step 6: zip each successfully downloaded L1C .SAFE
    # fix_s2 was already called inside download_safe on success
    # .SAFE folders are NEVER deleted
    # ------------------------------------------------------------------
    print('\n--- Zipping L1C .SAFE folders ---')
    for item in work_items:
        safe_path = item['safe_path']
        if not exists(safe_path):
            print(f'SAFE not found (download failed?): {safe_path}')
            continue
        safe_to_zip(safe_path)   # no-op if zip already present

    # ------------------------------------------------------------------
    # Sen2Cor (--L2 only)
    # ------------------------------------------------------------------
    if use_L2:
        print('\n--- Running Sen2Cor for L2A ---')
        sen2cor_jobs = []
        for item in work_items:
            pid       = item['pid']
            out_dir   = item['out_dir']
            safe_path = item['safe_path']
            l2a_pid       = pid.replace('MSIL1C', 'MSIL2A')
            l2a_safe_path = os.path.join(out_dir, l2a_pid + '.SAFE')
            l2a_zip_path  = os.path.join(out_dir, l2a_pid + '.zip')
            if exists(l2a_zip_path):
                print(f'L2A zip already present — skipping: {l2a_zip_path}')
                continue
            if not exists(safe_path):
                print(f'L1C SAFE missing — skipping sen2cor: {safe_path}')
                continue
            sen2cor_jobs.append((safe_path, l2a_safe_path, l2a_zip_path, out_dir))

        print(f'Sen2Cor jobs: {len(sen2cor_jobs)} '
              f'[{"parallel" if use_parallel else "serial"}]')

        if use_parallel:
            def _s2c(job):
                return run_sen2cor_job(job, l2a_process)
            parfor(_s2c, sen2cor_jobs, N_WORKERS)
        else:
            for job in sen2cor_jobs:
                run_sen2cor_job(job, l2a_process)

    print('\n=== Done ===')


# ---------------------------------------------------------------------------
# Argument parsing  (mirrors sync_daterange_gid_zip.py style)
# ---------------------------------------------------------------------------

use_L2        = '--L2'            in args
use_parallel  = '--parallel'      in args
force_listing = '--force-listing' in args

if '--L2' in args and '--L1' in args:
    err('Specify --L2 or --L1, not both.')
if '--L1' in args:
    use_L2 = False

clean_args = [a for a in args if not a.startswith('--')]

if len(clean_args) < 2:
    print(__doc__)
    sys.exit(0)

yyyymmdd = clean_args[1]

if len(clean_args) > 2 and is_date_format(clean_args[2]):
    yyyymmdd2     = clean_args[2]
    gid_start_idx = 3
else:
    yyyymmdd2     = yyyymmdd
    gid_start_idx = 2

raw_gids = set(clean_args[gid_start_idx:]) if len(clean_args) > gid_start_idx else set()

if not raw_gids:
    from gid import bc
    gids = bc()
    print('Pulling BC GIDs (default)...')
elif 'all' in raw_gids:
    gids = None
    print('Pulling all Canada GIDs...')
else:
    gids = raw_gids

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'{"="*60}')
    print(f'sync_daterange_gid_zip_gcp.py')
    print(f'Mode      : {"L2 (L1C + Sen2Cor)" if use_L2 else "L1C"}')
    print(f'Parallel  : {use_parallel}  (workers={N_WORKERS})')
    print(f'Date range: {yyyymmdd} -> {yyyymmdd2}')
    print(f'GIDs      : {gids if gids is not None else "ALL"}')
    print(f'Install   : {GCP_INSTALL_DIR}')
    print(f'Listings  : {LISTING_DIR}')
    print(f'{"="*60}')

    gsutil_path = ensure_gsutil()

    l2a_process = None
    if use_L2:
        l2a_process = ensure_sen2cor()

    download_by_gids(
        gids          = gids,
        yyyymmdd      = yyyymmdd,
        yyyymmdd2     = yyyymmdd2,
        use_L2        = use_L2,
        use_parallel  = use_parallel,
        force_listing = force_listing,
        gsutil_path   = gsutil_path,
        l2a_process   = l2a_process,
    )
