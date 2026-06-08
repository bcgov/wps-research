'''20260608 sentinel2_mrap_retain_full_frames.py:

    delete any data ( at the beginning ) where we don't have a full frame of data 

    I.e., find the first date that has no no-data ( NAN ) in it, and 

    prompt if we want to delete dates before that.
'''
#!/usr/bin/env python3

import os
import sys
from osgeo import gdal
import numpy as np


def get_filename_lines(folder):
    lines = [x.strip() for x in os.popen(f"ls -1r {folder}/S2*_cloudfree.bin").readlines()]

    if not lines:
        lines = [x.strip() for x in os.popen(f"ls -1r {folder}/S2*.bin").readlines()]
        lines = [x for x in lines if "MRAP" not in x]

    lines = [os.path.basename(x).split("_") for x in lines]
    lines = [[x[2], x] for x in lines]  # acquisition timestamp
    lines.sort()

    return [[x[0], "_".join(x[1])] for x in lines]


def has_nan(ds):
    for b in range(1, ds.RasterCount + 1):
        arr = ds.GetRasterBand(b).ReadAsArray()
        if np.isnan(arr).any():
            return True
    return False


def delete_older_files(folder, cutoff_date, lines):
    deleted = 0

    for line_date, filename in lines:
        if line_date >= cutoff_date:
            break

        path = os.path.join(folder, filename)

        print("DELETE", path)
        os.remove(path)

        hdr = path + ".hdr"
        if os.path.exists(hdr):
            os.remove(hdr)

        deleted += 1

    print(f"Deleted {deleted} date(s)")


def main(folder):

    lines = get_filename_lines(folder)

    if not lines:
        print("No Sentinel-2 files found.")
        return

    print(f"Found {len(lines)} dates")

    for idx, (line_date, filename) in enumerate(lines):

        path = os.path.join(folder, filename)

        print(f"[{idx+1}/{len(lines)}] checking {line_date}")

        ds = gdal.Open(path)

        if ds is None:
            print("Failed to open:", path)
            continue

        if not has_nan(ds):
            print()
            print("FIRST COMPLETE DATE FOUND")
            print(line_date)
            print(path)

            answer = input(
                f"\nDelete all dates before {line_date}? [y/N]: "
            ).strip().lower()

            if answer == "y":
                delete_older_files(folder, line_date, lines)

            return

        ds = None

    print("\nNo fully-populated date found.")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("    python3 find_first_complete_date.py L2_10UFA")
        sys.exit(1)

    main(sys.argv[1])


