#!/usr/bin/env python3
"""
20260915 estimate cloud cover of BC MRAP (most recent available pixel) products,
and optionally delete the cloudy ones.

Method:
  1. Search MRAP_DIR (default /data/mrap_bc/, recursively) for yyyymmdd_mrap.bin
     files that have an ENVI .hdr sidecar (yyyymmdd_mrap.hdr or yyyymmdd_mrap.bin.hdr).
     Collect and sort their dates.
  2. Get the BC Sentinel-2 tile IDs from bc_gid.py (last line of its output).
  3. Run sentinel2_extract_cloud_cover_tiles.py (--L2) for all BC tiles over the
     date range [earliest MRAP date - lookback, latest MRAP date], and read its CSV.
  4. For each MRAP date D and each tile, take the cloud cover of that tile's most
     recent acquisition ON OR BEFORE D (never after D). Multiple acquisitions of
     the same tile on the same day are averaged. The MRAP cloud cover estimate is
     the mean of those per-tile values over all tiles that have one.

Output (stdout): one line per yyyymmdd_mrap.bin:
    <path to yyyymmdd_mrap.bin> <estimated cloud cover %> [(*)]
(*) marks entries whose estimate is strictly greater than the threshold.
Progress, warnings and the output of the child scripts go to stderr, so stdout
stays a clean list.

Usage:
    sentinel2_mrap_delete_cloudy.py [THRESHOLD] [--delete] [options]

Arguments:
    THRESHOLD        cloud cover threshold in percent, 0 to 100 (optional)

Options:
    --delete         delete the yyyymmdd_mrap.bin and its .hdr for every (*) entry.
                     Requires THRESHOLD.
    --mrap_dir=DIR   where to search for yyyymmdd_mrap.bin (default: /data/mrap_bc/)
    --lookback=N     start the cloud cover query N days before the earliest MRAP date,
                     so every tile already has a "most recent" value on the earliest
                     MRAP dates (default: 0, i.e. exactly the MRAP date range)
    --workers=N      parallel workers passed to sentinel2_extract_cloud_cover_tiles.py
                     (default: 16)
    --workdir=DIR    working directory for the query cache, CSV and plot
                     (default: ~/.sentinel2_mrap_cloud). Reusing it keeps the
                     per-product cloud cover cache, so repeat runs are fast.
    --csv=FILE       skip the AWS query and use an existing *_L2A.csv written by
                     sentinel2_extract_cloud_cover_tiles.py

Examples:
    sentinel2_mrap_delete_cloudy.py
    sentinel2_mrap_delete_cloudy.py 40
    sentinel2_mrap_delete_cloudy.py 40 --lookback=10 --delete
"""

import os
import re
import sys
import csv
import bisect
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

MRAP_RE = re.compile(r'^(\d{8})_mrap\.bin$')
TILE_RE = re.compile(r'^T\d{2}[A-Z]{3}$')
PROD_TILE_RE = re.compile(r'_(T\d{2}[A-Z]{3})_')
PROD_DATE_RE = re.compile(r'_(\d{8})T\d{6}_')   # first timestamp = sensing time

CLOUD_SCRIPT = 'sentinel2_extract_cloud_cover_tiles.py'
GID_SCRIPT = 'bc_gid.py'


def err(msg):
    print(msg, file=sys.stderr, flush=True)


def find_mrap(mrap_dir):
    """Return sorted list of (date, bin_path, [hdr_paths])."""
    found = []
    for root, dirs, files in os.walk(mrap_dir):
        dirs.sort()
        for fn in files:
            m = MRAP_RE.match(fn)
            if not m:
                continue
            try:
                d = datetime.strptime(m.group(1), '%Y%m%d').date()
            except ValueError:
                err(f"WARNING: bad date in filename, skipping: {os.path.join(root, fn)}")
                continue
            bin_path = os.path.join(root, fn)
            hdrs = [p for p in (bin_path[:-4] + '.hdr', bin_path + '.hdr') if os.path.isfile(p)]
            if not hdrs:
                err(f"WARNING: no .hdr sidecar for {bin_path}, skipping")
                continue
            found.append((d, bin_path, hdrs))
    found.sort()
    return found


def get_bc_tiles():
    """Run bc_gid.py and parse the space-separated tile ID list it prints last."""
    r = subprocess.run([GID_SCRIPT], capture_output=True, text=True)
    if r.returncode != 0:
        err(r.stderr)
        sys.exit(f"ERROR: {GID_SCRIPT} failed (exit code {r.returncode})")
    for line in reversed(r.stdout.splitlines()):
        toks = line.split()
        if toks and all(TILE_RE.match(t) for t in toks):
            return sorted(set(toks))
    sys.exit(f"ERROR: could not find the tile ID list in {GID_SCRIPT} output")


def query_cloud_cover(start, end, tiles, workers, workdir):
    """Run the cloud cover extraction script; return path of the L2A CSV."""
    os.makedirs(workdir, exist_ok=True)
    prefix = os.path.join(workdir, f"mrap_cloud_{start:%Y%m%d}_{end:%Y%m%d}")
    csv_path = prefix + '_L2A.csv'
    if os.path.exists(csv_path):
        os.remove(csv_path)   # never read back a stale CSV

    cmd = ([CLOUD_SCRIPT, f"{start:%Y%m%d}", f"{end:%Y%m%d}"] + tiles +
           ['--L2', f'--workers={workers}', f'--output={prefix}'])
    err(f"Running {CLOUD_SCRIPT} {start:%Y%m%d} {end:%Y%m%d} <{len(tiles)} tiles> "
        f"--L2 --workers={workers} (cwd={workdir})")
    r = subprocess.run(cmd, cwd=workdir, stdout=sys.stderr, stderr=sys.stderr)
    if r.returncode != 0:
        sys.exit(f"ERROR: {CLOUD_SCRIPT} failed (exit code {r.returncode})")
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {CLOUD_SCRIPT} wrote no CSV ({csv_path}); no products found?")
    return csv_path


def load_cloud_cover(csv_path, tiles):
    """Return {tile: (sorted_dates, cloud_pct_per_date)}, same-day values averaged."""
    tile_set = set(tiles)
    per_tile_day = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            pid = row.get('Product', '')
            mt, md = PROD_TILE_RE.search(pid), PROD_DATE_RE.search(pid)
            if not mt or not md or mt.group(1) not in tile_set:
                continue
            try:
                cc = float(row['CloudPercentage'])
                d = datetime.strptime(md.group(1), '%Y%m%d').date()
            except (KeyError, ValueError):
                continue
            per_tile_day[mt.group(1)][d].append(cc)

    series = {}
    for tile, days in per_tile_day.items():
        ds = sorted(days)
        series[tile] = (ds, [sum(days[d]) / len(days[d]) for d in ds])
    return series


def estimate_mrap_cloud(d, series):
    """Mean over tiles of each tile's most recent cloud cover on or before d."""
    vals = []
    for ds, ccs in series.values():
        i = bisect.bisect_right(ds, d) - 1   # last index with ds[i] <= d
        if i >= 0:
            vals.append(ccs[i])
    return (sum(vals) / len(vals) if vals else None), len(vals)


def main():
    threshold = None
    delete = False
    mrap_dir = '/data/mrap_bc/'
    lookback = 0
    workers = 16
    workdir = os.path.expanduser('~/.sentinel2_mrap_cloud')
    csv_file = None

    for arg in sys.argv[1:]:
        if arg in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)
        elif arg == '--delete':
            delete = True
        elif arg.startswith('--mrap_dir='):
            mrap_dir = arg.split('=', 1)[1]
        elif arg.startswith('--lookback='):
            lookback = int(arg.split('=', 1)[1])
        elif arg.startswith('--workers='):
            workers = int(arg.split('=', 1)[1])
        elif arg.startswith('--workdir='):
            workdir = os.path.abspath(os.path.expanduser(arg.split('=', 1)[1]))
        elif arg.startswith('--csv='):
            csv_file = arg.split('=', 1)[1]
        elif threshold is None and not arg.startswith('--'):
            try:
                threshold = float(arg)
            except ValueError:
                sys.exit(f"ERROR: threshold must be a number from 0 to 100, got: {arg}")
            if not 0.0 <= threshold <= 100.0:
                sys.exit(f"ERROR: threshold must be from 0 to 100, got: {threshold}")
        else:
            sys.exit(f"ERROR: unrecognized argument: {arg}\n{__doc__}")

    if delete and threshold is None:
        sys.exit("ERROR: --delete requires a threshold")
    if lookback < 0:
        sys.exit("ERROR: --lookback must be >= 0")

    # 1. MRAP products
    mrap = find_mrap(mrap_dir)
    if not mrap:
        sys.exit(f"ERROR: no yyyymmdd_mrap.bin (+ .hdr) found under {mrap_dir}")
    first, last = mrap[0][0], mrap[-1][0]
    err(f"Found {len(mrap)} MRAP products, {first} to {last}")

    # 2. BC tiles
    tiles = get_bc_tiles()
    err(f"BC tiles from {GID_SCRIPT}: {len(tiles)}")

    # 3. Cloud cover per tile per acquisition date
    if csv_file is None:
        csv_file = query_cloud_cover(first - timedelta(days=lookback), last,
                                     tiles, workers, workdir)
    series = load_cloud_cover(csv_file, tiles)
    err(f"Cloud cover series loaded for {len(series)} of {len(tiles)} tiles from {csv_file}")

    no_data = [t for t in tiles if t not in series]
    if no_data:
        err(f"WARNING: {len(no_data)} tile(s) have no cloud cover data in range "
            f"and are left out of every average: {' '.join(no_data)}")

    # 4. Estimate, report, flag
    flagged = []
    partial = []
    for d, bin_path, hdrs in mrap:
        cc, n = estimate_mrap_cloud(d, series)
        if cc is None:
            print(f"{bin_path} NA")
            continue
        if n < len(series):
            partial.append(f"{d:%Y%m%d}({n}/{len(series)})")
        star = threshold is not None and cc > threshold
        print(f"{bin_path} {cc:.2f}" + (" (*)" if star else ""), flush=True)
        if star:
            flagged.append((bin_path, hdrs))

    if partial:
        err(f"WARNING: on {len(partial)} MRAP date(s) some tiles had no acquisition yet "
            f"within the queried range, so the average uses fewer tiles "
            f"(consider --lookback=N): {' '.join(partial)}")

    if threshold is not None:
        err(f"{len(flagged)} of {len(mrap)} MRAP products exceed {threshold:g}% cloud cover")

    # 5. Delete
    if delete:
        for bin_path, hdrs in flagged:
            for p in [bin_path] + hdrs:
                try:
                    os.remove(p)
                    err(f"deleted {p}")
                except OSError as e:
                    err(f"ERROR: could not delete {p}: {e}")


if __name__ == '__main__':
    main()
