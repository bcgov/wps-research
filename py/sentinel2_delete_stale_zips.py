#!/usr/bin/env python3
"""
20260812: List Sentinel-2 product zips that are older than a cutoff age.

Age is taken from the *sensing* timestamp in the filename -- the first
YYYYMMDDTHHMMSS field, e.g. for

    S2C_MSIL2A_20260811T184921_N0512_R113_T11UNS_20260812T000210.zip
                ^^^^^^^^^^^^^^^

the sensing time is 2026-08-11 18:49:21 UTC.  The trailing timestamp
(processing/baseline time) is ignored.

Usage:
    ./old_s2_zips.py                       # default: /data/mrap_bc/, 14 days
    ./old_s2_zips.py --days 30
    ./old_s2_zips.py --root /data/mrap_bc/L2_T11UNS
    ./old_s2_zips.py --paths-only | xargs -d '\n' ls -lh
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

# First _YYYYMMDDTHHMMSS_ group in the basename = sensing start time.
STAMP_RE = re.compile(r"_(\d{8}T\d{6})_")


def find_files(root, pattern):
    """Run: find <root> -name <pattern>   and return the hits."""
    try:
        proc = subprocess.run(
            ["find", root, "-name", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("error: 'find' not available on PATH")

    if proc.stderr.strip():
        # e.g. permission-denied on some subtree; report but keep going
        for line in proc.stderr.strip().splitlines():
            print(f"find: {line}", file=sys.stderr)
    if proc.returncode != 0 and not proc.stdout.strip():
        sys.exit(f"error: find failed on {root!r} (exit {proc.returncode})")

    return [p for p in proc.stdout.splitlines() if p.strip()]


def sensing_time(path):
    """Parse the sensing timestamp from a filename. Returns aware UTC datetime or None."""
    m = STAMP_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/data/mrap_bc/",
                    help="directory to search (default: %(default)s)")
    ap.add_argument("--pattern", default="S2*.zip",
                    help="find -name pattern (default: %(default)s)")
    ap.add_argument("--days", type=float, default=14,
                    help="age cutoff in days (default: %(default)s)")
    ap.add_argument("--paths-only", action="store_true",
                    help="print bare paths only, for piping into xargs")
    ap.add_argument("--newer", action="store_true",
                    help="invert: list files NEWER than the cutoff instead")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=args.days)

    hits, unparsed = [], []
    for path in find_files(args.root, args.pattern):
        stamp = sensing_time(path)
        if stamp is None:
            unparsed.append(path)
            continue
        is_old = stamp < cutoff
        if is_old != args.newer:          # XOR against --newer
            hits.append((stamp, path))

    hits.sort()                            # oldest first

    if args.paths_only:
        for _, path in hits:
            print(path)
    else:
        for stamp, path in hits:
            age = (now - stamp).total_seconds() / 86400.0
            print(f"{stamp:%Y-%m-%d %H:%M}  {age:6.1f}d  {path}")

    if unparsed:
        print(f"\n{len(unparsed)} file(s) with no parsable timestamp:", file=sys.stderr)
        for path in unparsed:
            print(f"  {path}", file=sys.stderr)

    if not args.paths_only:
        word = "newer" if args.newer else "older"
        print(f"\n{len(hits)} file(s) {word} than {args.days:g} days "
              f"(cutoff {cutoff:%Y-%m-%d %H:%M} UTC)", file=sys.stderr)


if __name__ == "__main__":
    main()


