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


def get_filename_lines(search_cmd):
    lines = [x.strip() for x in os.popen(search_cmd).readlines()]
    lines = [os.path.basename(x).split('_') for x in lines]
    lines = [[x[2], x] for x in lines]
    lines.sort()
    return [[x[0], '_'.join(x[1])] for x in lines]


def contains_nan(filename):
    d = gdal.Open(filename)

    if d is None:
        raise RuntimeError(f"Could not open {filename}")

    for b in range(1, d.RasterCount + 1):

        band = d.GetRasterBand(b)

        xsize = band.XSize
        ysize = band.YSize

        # scan in blocks so we don't load entire bands
        block_x = 1024
        block_y = 1024

        for y in range(0, ysize, block_y):

            rows = min(block_y, ysize - y)

            for x in range(0, xsize, block_x):

                cols = min(block_x, xsize - x)

                arr = band.ReadAsArray(x, y, cols, rows)

                if np.isnan(arr).any():
                    d = None
                    return True

    d = None
    return False


def remove_file_and_hdr(fn):

    if os.path.exists(fn):
        print("DELETE", fn)
        os.remove(fn)

    hdr = fn + ".hdr"

    if os.path.exists(hdr):
        print("DELETE", hdr)
        os.remove(hdr)


def main():

    search_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    search_cmd = f'ls -1r "{search_dir}"/*MRAP.bin'

    lines = get_filename_lines(search_cmd)

    if not lines:
        print("No *MRAP.bin files found.")
        return

    print("Found", len(lines), "MRAP files")

    first_complete_date = None

    for i, (line_date, filename) in enumerate(lines):

        full_path = os.path.join(search_dir, filename)

        print(f"[{i+1}/{len(lines)}] {line_date} {filename}")

        if not contains_nan(full_path):

            first_complete_date = line_date

            print()
            print("FIRST COMPLETE MRAP FOUND")
            print("Date:", line_date)
            print("File:", filename)
            print()

            break

    if first_complete_date is None:
        print("No complete MRAP found.")
        return

    response = input(
        f"Delete all MRAP dates before {first_complete_date}? [y/N] "
    ).strip().lower()

    if response != "y":
        print("Aborted.")
        return

    deleted = 0

    for line_date, filename in lines:

        if line_date >= first_complete_date:
            break

        full_path = os.path.join(search_dir, filename)

        remove_file_and_hdr(full_path)

        deleted += 1

    print()
    print("Deleted", deleted, "MRAP files")


if __name__ == "__main__":
    main()
