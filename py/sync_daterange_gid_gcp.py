'''20260603: GCP variant of sync_daterange_gid_zip.py

Downloads Sentinel-2 L1C data from Google Cloud Platform (public bucket,
no credentials required). If --L2 is requested, runs Sen2Cor locally to
produce L2A, since GCP only stores L1C.

Usage:
    python3 sync_daterange_gid_zip_gcp.py [yyyymmdd] [yyyymmdd2] [GID ...] [flags]

Flags:
    --L1            L1C mode (default)
    --L2            download L1C + run Sen2Cor -> L2A zip
    --parallel      enable parallel downloads (N_DL_WORKERS=4); sen2cor is always parallel (N_S2C_WORKERS=64)
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

N_DL_WORKERS  = 4    # parallel workers for gsutil downloads (--parallel flag)
N_S2C_WORKERS = 64   # parallel workers for sen2cor — always parallel, flag ignored

# ---------------------------------------------------------------------------
# Listing directory: shared with the AWS sync script
# ---------------------------------------------------------------------------
_base0 = '/data/'
_base1 = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
_listing_base = _base1
try:
    if not exists(_base0):
        print(f'[{__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] mkdir {_base0}')
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
        log(f'mkdir {path}')
        os.makedirs(path, exist_ok=True)


def today_str():
    td = datetime.date.today()
    return f'{td.year:04d}{td.month:02d}{td.day:02d}'


def is_date_format(s):
    return len(s) == 8 and s.isdigit()


def log(msg, prefix=''):
    """Timestamped print — all script output goes through here."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tag = f'[{prefix}] ' if prefix else ''
    print(f'[{ts}] {tag}{msg}', flush=True)


def print_status_update(file_name, file_size, file_dl_time):
    global _files_completed, _total_files, _bytes_completed, _total_bytes, _download_start_time
    elapsed  = time.time() - _download_start_time
    pct_f    = (_files_completed / _total_files * 100) if _total_files > 0 else 0
    eta      = (elapsed / _files_completed * (_total_files - _files_completed)
                if _files_completed > 0 else 0)

    # Speed: MB/s for this file, and overall average MB/s
    speed_now = file_size / (1024**2) / file_dl_time if file_dl_time > 0 else 0
    speed_avg = _bytes_completed / (1024**2) / elapsed if elapsed > 0 else 0

    # Bytes: we don't know total upfront (index has no sizes), so show
    # cumulative downloaded + a rolling estimate of total based on avg file size.
    avg_bytes  = _bytes_completed / _files_completed if _files_completed > 0 else 0
    est_total  = avg_bytes * _total_files
    pct_b      = (_bytes_completed / est_total * 100) if est_total > 0 else 0

    remaining_files = _total_files - _files_completed
    est_remaining_bytes = avg_bytes * remaining_files

    print(f"\n{'='*60}")
    print(f"DOWNLOAD STATUS UPDATE")
    print(f"{'='*60}")
    print(f"File completed: {file_name}")
    print(f"File size: {file_size/(1024**2):.2f} MB | Download time: {file_dl_time:.1f}s | Speed: {speed_now:.1f} MB/s")
    print(f"{'='*60}")
    print(f"Files:   {_files_completed} / {_total_files} ({pct_f:.1f}%)")
    print(f"Bytes:   {_bytes_completed/(1024**3):.2f} GB downloaded"
          f"  (~{est_total/(1024**3):.1f} GB est. total, {pct_b:.1f}%)")
    print(f"Avg size/file: {avg_bytes/(1024**2):.0f} MB  |  Avg speed: {speed_avg:.1f} MB/s")
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
    log('STEP 1: Locating gsutil', 'gsutil')
    gsutil = find_gsutil()
    if gsutil:
        log(f'gsutil found: {gsutil}', 'gsutil')
        _ensure_crcmod()
        return gsutil

    log('gsutil not found — installing Google Cloud SDK...', 'gsutil')
    log(f'  install dir : {GCP_INSTALL_DIR}', 'gsutil')
    log(f'  SDK version : {SDK_VERSION}', 'gsutil')
    log(f'  SDK URL     : {SDK_URL}', 'gsutil')
    md(GCP_INSTALL_DIR)

    local_tar = os.path.join(GCP_INSTALL_DIR, SDK_TARBALL)
    if not exists(local_tar):
        log(f'  Downloading SDK tarball -> {local_tar}', 'gsutil')
        run(f'wget -q --show-progress -O {local_tar} {SDK_URL}')
        log(f'  Download complete: {local_tar}', 'gsutil')
    else:
        log(f'  SDK tarball already cached: {local_tar}', 'gsutil')

    sdk_dir = os.path.join(GCP_INSTALL_DIR, 'google-cloud-sdk')
    if not exists(sdk_dir):
        log(f'  Extracting SDK to {GCP_INSTALL_DIR}...', 'gsutil')
        run(f'tar -xzf {local_tar} -C {GCP_INSTALL_DIR}')
        log(f'  Extraction complete: {sdk_dir}', 'gsutil')
    else:
        log(f'  SDK already extracted: {sdk_dir}', 'gsutil')

    install_sh = os.path.join(sdk_dir, 'install.sh')
    log(f'  Running install.sh...', 'gsutil')
    run(f'bash {install_sh} --quiet --usage-reporting=false --path-update=true')

    gsutil = os.path.join(sdk_dir, 'bin', 'gsutil')
    if not exists(gsutil):
        err(f'gsutil not found after install — expected: {gsutil}')

    _ensure_crcmod()
    log(f'gsutil ready: {gsutil}', 'gsutil')

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
        log('crcmod already installed (gsutil CRC32c acceleration active)', 'gsutil')
    except ImportError:
        log('crcmod not found — installing for gsutil CRC32c performance...', 'gsutil')
        run('sudo apt-get install -y gcc python3-dev python3-setuptools 2>/dev/null || true')
        run('pip3 install --no-cache-dir -U crcmod --break-system-packages 2>/dev/null '
            '|| pip3 install --no-cache-dir -U crcmod')
        log('crcmod install complete', 'gsutil')


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
    log('STEP 2: Locating Sen2Cor (L2A_Process)', 'sen2cor')
    log(f'  version   : {SEN2COR_VERSION}', 'sen2cor')
    log(f'  installer : {SEN2COR_INSTALLER}', 'sen2cor')
    l2a = find_sen2cor()
    if l2a:
        log(f'L2A_Process found: {l2a}', 'sen2cor')
        return l2a

    log('Sen2Cor not found — installing...', 'sen2cor')
    log(f'  URL: {SEN2COR_URL}', 'sen2cor')
    md(GCP_INSTALL_DIR)

    local_installer = os.path.join(GCP_INSTALL_DIR, SEN2COR_INSTALLER)
    if not exists(local_installer):
        log(f'  Downloading Sen2Cor installer -> {local_installer}', 'sen2cor')
        run(f'wget -q --show-progress -O {local_installer} {SEN2COR_URL}')
        log(f'  Download complete: {local_installer}', 'sen2cor')
    else:
        log(f'  Sen2Cor installer already cached: {local_installer}', 'sen2cor')

    log(f'  Running installer (target={GCP_INSTALL_DIR})...', 'sen2cor')
    run(f'bash {local_installer} --target {GCP_INSTALL_DIR}')

    l2a = find_sen2cor()
    if not l2a:
        err(
            f'L2A_Process not found after Sen2Cor install under {GCP_INSTALL_DIR}.\n'
            f'If the ESA URL has changed, update SEN2COR_VERSION/SEN2COR_URL and retry.\n'
            f'Manual: https://step.esa.int/main/snap-supported-plugins/sen2cor/'
        )
    log(f'Sen2Cor ready: {l2a}', 'sen2cor')
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

    log('STEP 3: Selecting GCP index listing', 'listing')
    log(f'  listing dir : {LISTING_DIR}', 'listing')
    log(f'  newest file : {newest or "none"}', 'listing')
    log(f'  newest date : {newest_date or "n/a"}', 'listing')
    log(f'  today       : {today}', 'listing')
    log(f'  end_date    : {end_date_str}', 'listing')

    need_dl = False
    if force_update:
        log('--force-listing set: will refresh GCP index', 'listing')
        need_dl = True
    elif newest is None:
        log('No GCP listing found — will download fresh copy', 'listing')
        need_dl = True
    elif end_date_str <= newest_date:
        log(f'Historical query (end={end_date_str} <= listing={newest_date}): reusing {newest}', 'listing')
    elif newest_date < today:
        log(f'Listing from {newest_date} is stale (today={today}): will refresh', 'listing')
        need_dl = True
    else:
        log(f'Listing already fresh from today ({newest_date}): reusing {newest}', 'listing')

    if need_dl:
        ts       = timestamp()
        new_fn   = GCP_INDEX_PREFIX + ts + '.csv.gz'
        new_path = os.path.join(LISTING_DIR, new_fn)
        log(f'  Downloading GCP index from {GCP_INDEX_URL}', 'listing')
        log(f'  -> {new_path}', 'listing')
        run(f'wget -q --show-progress -O {new_path} {GCP_INDEX_URL}')
        newest_path = new_path
        log(f'  Download complete: {new_path}', 'listing')

    log(f'Using listing: {newest_path}', 'listing')
    return newest_path


def load_index(listing_gz_path):
    '''
    Parse GCP index CSV. Returns (product_to_url, product_to_cloud).

    Decompressed CSV is cached as <listing_gz_path without .gz>.
    On subsequent runs the .csv is read directly — no gunzip overhead.
    Each new .gz gets its own sibling .csv on first use; if you delete
    or replace the .gz its .csv won't exist yet so decompression reruns once.
    '''
    # Cache path: strip .gz to get the plain .csv sibling
    if listing_gz_path.endswith('.gz'):
        listing_csv_path = listing_gz_path[:-3]   # e.g. gcp_index_20260603_120000.csv
    else:
        listing_csv_path = listing_gz_path         # already uncompressed somehow

    log('STEP 4: Loading GCP product index', 'index')
    log(f'  gz  : {listing_gz_path}', 'index')
    log(f'  csv : {listing_csv_path}', 'index')

    if not exists(listing_csv_path):
        log(f'  Decompressing gz -> csv (first use of this listing)...', 'index')
        t0 = time.time()
        with gzip.open(listing_gz_path, 'rb') as src_fh, \
             open(listing_csv_path, 'wb') as dst_fh:
            shutil.copyfileobj(src_fh, dst_fh)
        csv_size = os.path.getsize(listing_csv_path)
        log(f'  Decompressed in {time.time()-t0:.1f}s  ({csv_size/(1024**2):.1f} MB)', 'index')
    else:
        csv_size = os.path.getsize(listing_csv_path)
        log(f'  Using cached CSV ({csv_size/(1024**2):.1f} MB): {listing_csv_path}', 'index')

    log(f'  Parsing CSV...', 'index')
    t1 = time.time()
    product_to_url   = {}
    product_to_cloud = {}
    with open(listing_csv_path, 'r', encoding='utf-8', errors='replace') as fh:
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
    log(f'  Index parsed in {time.time()-t1:.1f}s: {len(product_to_url):,} products', 'index')
    return product_to_url, product_to_cloud


# ---------------------------------------------------------------------------
# fix_s2 (inline; mirrors fix_s2.py)
# GCP omits empty dirs; Sen2Cor requires AUX_DATA/ and HTML/
# ---------------------------------------------------------------------------

def fix_s2(safe_dir):
    for sub in ('AUX_DATA', 'HTML'):
        d = os.path.join(safe_dir, sub)
        if not exists(d):
            log(f'fix_s2: creating missing dir: {d}', 'fix_s2')
            os.makedirs(d, exist_ok=True)
        else:
            log(f'fix_s2: {sub} already exists', 'fix_s2')


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
        log(f'Zip already present: {zip_path}', 'zip')
        return zip_path

    log(f'Zipping {safe_name} -> {zip_path}', 'zip')
    # make_archive(base_name, format, root_dir, base_dir)
    #   root_dir = parent     <- archive is created relative to here
    #   base_dir = safe_name  <- this subtree becomes the zip root entry
    shutil.make_archive(zip_stem, 'zip', parent, safe_name)
    zip_size = os.path.getsize(zip_path) if exists(zip_path) else 0
    log(f'Created: {zip_path}  ({zip_size/(1024**2):.0f} MB)', 'zip')
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
        log(f'Already extracted: {out_safe}', 'zip')
        return out_safe

    log(f'Extracting {zip_path} -> {target_parent}/', 'zip')
    t0 = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        log(f'  zip contains {len(members)} members', 'zip')
        bad = [m for m in members
               if not (m == safe_name + '/' or m.startswith(safe_name + '/'))]
        if bad:
            log(f'  WARNING: unexpected zip members (first 5): {bad[:5]}', 'zip')
        zf.extractall(target_parent)

    if not exists(out_safe):
        err(f'Expected {out_safe} after extraction — not found.')
    log(f'Extracted in {time.time()-t0:.1f}s: {out_safe}', 'zip')
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

    log(f'DOWNLOAD START: {pid}', 'rsync')
    log(f'  src  : {base_url}', 'rsync')
    log(f'  dst  : {safe_path}', 'rsync')

    # Create output dirs — gsutil rsync requires destination to exist
    md(out_dir)
    os.makedirs(safe_path, exist_ok=True)

    stdout_log = safe_path + '_stdout.txt'
    stderr_log = safe_path + '_stderr.txt'
    log(f'  logs : {stderr_log}', 'rsync')

    # Trailing slashes on both src and dst required by gsutil rsync
    src = base_url.rstrip('/') + '/'
    dst = safe_path.rstrip('/') + '/'
    cmd = f'{gsutil_path} -m rsync -r {src} {dst}'
    log(f'  cmd  : {cmd}', 'rsync')

    t0  = time.time()
    ret = os.system(f'{cmd} > {stdout_log} 2> {stderr_log}')
    dt  = time.time() - t0

    if ret != 0:
        log(f'FAILED (exit {ret}): {pid}', 'rsync')
        # Print stderr inline so failures are visible in the main log
        if exists(stderr_log):
            try:
                stderr_text = open(stderr_log).read().strip()
                if stderr_text:
                    log(f'  stderr:\n{stderr_text}', 'rsync')
            except Exception:
                pass
        if exists(stdout_log):
            try:
                stdout_text = open(stdout_log).read().strip()
                if stdout_text:
                    log(f'  stdout:\n{stdout_text}', 'rsync')
            except Exception:
                pass
    else:
        log(f'DOWNLOAD OK in {dt:.1f}s: {pid}', 'rsync')
        log(f'  Running fix_s2 on {safe_path}', 'rsync')
        fix_s2(safe_path)
        log(f'  fix_s2 done', 'rsync')

    # Count bytes on disk (works even for partial downloads)
    safe_bytes = (
        sum(os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(safe_path) for f in fs)
        if exists(safe_path) else 0
    )
    log(f'  on-disk size: {safe_bytes/(1024**2):.1f} MB', 'rsync')

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

    log(f'SEN2COR START: {os.path.basename(safe_path)}', 'sen2cor')
    log(f'  L1C  : {safe_path}', 'sen2cor')
    log(f'  L2A  : {l2a_safe_path}', 'sen2cor')
    log(f'  zip  : {l2a_zip_path}', 'sen2cor')

    if exists(l2a_zip_path):
        log(f'L2A zip already present — skipping: {l2a_zip_path}', 'sen2cor')
        return

    if not exists(safe_path):
        log(f'ERROR: L1C SAFE missing — cannot run sen2cor: {safe_path}', 'sen2cor')
        return

    log(f'  Running fix_s2 on L1C...', 'sen2cor')
    fix_s2(safe_path)

    cmd = f'{l2a_process} {safe_path}'
    log(f'  cmd  : {cmd}', 'sen2cor')
    t0  = time.time()
    ret = os.system(cmd)
    dt  = time.time() - t0

    if ret != 0:
        log(f'ERROR: sen2cor returned {ret} after {dt:.1f}s for {safe_path}', 'sen2cor')
        return

    log(f'  sen2cor finished in {dt:.1f}s', 'sen2cor')

    # Sen2Cor should write L2A .SAFE into the same directory as L1C .SAFE
    if not exists(l2a_safe_path):
        # Some sen2cor versions write to cwd — check and relocate
        cwd_l2a = os.path.join(os.getcwd(), os.path.basename(l2a_safe_path))
        if exists(cwd_l2a):
            log(f'  Relocating sen2cor output: {cwd_l2a} -> {l2a_safe_path}', 'sen2cor')
            shutil.move(cwd_l2a, l2a_safe_path)
        else:
            log(f'ERROR: L2A SAFE not found after sen2cor at either:', 'sen2cor')
            log(f'    {l2a_safe_path}', 'sen2cor')
            log(f'    {cwd_l2a}', 'sen2cor')
            return

    l2a_bytes = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(l2a_safe_path) for f in fs)
    log(f'  L2A on-disk size: {l2a_bytes/(1024**2):.1f} MB', 'sen2cor')

    log(f'  Running fix_s2 on L2A...', 'sen2cor')
    fix_s2(l2a_safe_path)

    log(f'  Zipping L2A: {l2a_safe_path}', 'sen2cor')
    t1 = time.time()
    safe_to_zip(l2a_safe_path)
    log(f'  Zip done in {time.time()-t1:.1f}s: {l2a_zip_path}', 'sen2cor')

    log(f'SEN2COR DONE: {os.path.basename(l2a_safe_path)}', 'sen2cor')


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
    log(f'Date range: {start_d.date()} -> {end_d.date()}')

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
                log(f'SKIP (L2A zip present): {l2a_zip_path}')
                skipped += 1
                continue
        else:
            if exists(zip_path):
                log(f'SKIP (L1C zip present): {zip_path}')
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

    log(f'{"="*60}')
    log(f'MATCH SUMMARY')
    log(f'  Products to download/process : {len(work_items)}')
    log(f'  Already complete (skipped)   : {skipped}')
    log(f'  Output prefix                : {out_prefix}')
    log(f'  Mode                         : {"L2 (L1C + Sen2Cor)" if use_L2 else "L1C only"}')
    log(f'{"="*60}')

    if not work_items:
        log('Nothing to do — all products already complete.')
        return

    disk = shutil.disk_usage(os.getcwd())
    rough_gb = len(work_items) * 1.0
    log(f'Disk free: {disk.free/(1024**3):.1f} GB  |  rough estimate: ~{rough_gb:.0f} GB needed')
    if disk.free < rough_gb * (1024**3):
        log(f'WARNING: disk may be insufficient — {disk.free/(1024**3):.1f} GB free, '
            f'~{rough_gb:.0f} GB estimated')

    _total_files         = len(work_items)
    _download_start_time = time.time()

    # ------------------------------------------------------------------
    # Step 5: download .SAFE folders via gsutil rsync
    # ------------------------------------------------------------------
    log(f'STEP 5: Downloading {_total_files} .SAFE folder(s) '
        f'[{"parallel" if use_parallel else "serial"}, workers={N_DL_WORKERS if use_parallel else 1}]')

    t_dl0 = time.time()
    if use_parallel:
        def _dl(item):
            return download_safe(item, gsutil_path)
        parfor(_dl, work_items, N_DL_WORKERS)
    else:
        for item in work_items:
            download_safe(item, gsutil_path)
    t_dl = time.time() - t_dl0

    dl_ok  = sum(1 for item in work_items if exists(item['safe_path']))
    dl_bad = _total_files - dl_ok
    log(f'STEP 5 DONE in {t_dl/60:.1f} min: {dl_ok} succeeded, {dl_bad} failed')

    # ------------------------------------------------------------------
    # Step 6: zip each successfully downloaded L1C .SAFE
    # fix_s2 was already called inside download_safe on success
    # .SAFE folders are NEVER deleted
    # ------------------------------------------------------------------
    log(f'STEP 6: Zipping {dl_ok} L1C .SAFE folder(s)')
    t_zip0   = time.time()
    zip_ok   = 0
    zip_skip = 0
    zip_fail = 0
    for item in work_items:
        safe_path = item['safe_path']
        zip_path  = item['zip_path']
        if not exists(safe_path):
            log(f'  SKIP (no SAFE): {safe_path}', 'zip')
            zip_fail += 1
            continue
        safe_size = sum(os.path.getsize(os.path.join(r, f))
                        for r, _, fs in os.walk(safe_path) for f in fs)
        if exists(zip_path):
            log(f'  ALREADY ZIPPED: {zip_path}', 'zip')
            zip_skip += 1
            continue
        log(f'  Zipping {os.path.basename(safe_path)} ({safe_size/(1024**2):.0f} MB) -> {zip_path}', 'zip')
        t1 = time.time()
        safe_to_zip(safe_path)
        zip_size = os.path.getsize(zip_path) if exists(zip_path) else 0
        log(f'  Done in {time.time()-t1:.1f}s  zip size: {zip_size/(1024**2):.0f} MB', 'zip')
        zip_ok += 1
    log(f'STEP 6 DONE in {(time.time()-t_zip0)/60:.1f} min: '
        f'{zip_ok} zipped, {zip_skip} already existed, {zip_fail} failed')

    # ------------------------------------------------------------------
    # Sen2Cor (--L2 only)
    # ------------------------------------------------------------------
    if use_L2:
        sen2cor_jobs = []
        for item in work_items:
            pid       = item['pid']
            out_dir   = item['out_dir']
            safe_path = item['safe_path']
            l2a_pid       = pid.replace('MSIL1C', 'MSIL2A')
            l2a_safe_path = os.path.join(out_dir, l2a_pid + '.SAFE')
            l2a_zip_path  = os.path.join(out_dir, l2a_pid + '.zip')
            if exists(l2a_zip_path):
                log(f'  L2A zip already present — skipping: {l2a_zip_path}', 'sen2cor')
                continue
            if not exists(safe_path):
                log(f'  L1C SAFE missing — skipping sen2cor: {safe_path}', 'sen2cor')
                continue
            sen2cor_jobs.append((safe_path, l2a_safe_path, l2a_zip_path, out_dir))

        log(f'STEP 7: Sen2Cor on {len(sen2cor_jobs)} tile(s) '
            f'[always parallel, workers={N_S2C_WORKERS}]')
        t_s2c0 = time.time()
        def _s2c(job):
            return run_sen2cor_job(job, l2a_process)
        parfor(_s2c, sen2cor_jobs, N_S2C_WORKERS)

        s2c_ok = sum(1 for _, l2a_safe, l2a_zip, _ in sen2cor_jobs if exists(l2a_zip))
        log(f'STEP 7 DONE in {(time.time()-t_s2c0)/60:.1f} min: '
            f'{s2c_ok}/{len(sen2cor_jobs)} L2A zips produced')

    total_elapsed = time.time() - _download_start_time
    log(f'ALL DONE — total elapsed: {total_elapsed/60:.1f} min  ({total_elapsed/3600:.2f} hr)')
    log(f'  Downloaded : {_bytes_completed/(1024**3):.2f} GB across {_files_completed} files')


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
    log('Pulling BC GIDs (default)...')
elif 'all' in raw_gids:
    gids = None
    log('Pulling all Canada GIDs...')
else:
    gids = raw_gids

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    log(f'{"="*60}')
    log(f'sync_daterange_gid_zip_gcp.py')
    log(f'Started   : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'Mode      : {"L2 (L1C + Sen2Cor)" if use_L2 else "L1C"}')
    log(f'Parallel  : {use_parallel}  (dl_workers={N_DL_WORKERS}, s2c_workers={N_S2C_WORKERS})')
    log(f'Date range: {yyyymmdd} -> {yyyymmdd2}')
    log(f'GIDs      : {gids if gids is not None else "ALL"}')
    log(f'Install   : {GCP_INSTALL_DIR}')
    log(f'Listings  : {LISTING_DIR}')
    log(f'CWD       : {os.getcwd()}')
    log(f'{"="*60}')

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
