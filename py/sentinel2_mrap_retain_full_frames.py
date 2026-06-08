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


def nan_percent(filename):
    """
    Return percentage of pixels that are NaN across all bands.

    Uses block reads so memory usage remains low.
    """

    d = gdal.Open(filename)

    if d is None:
        raise RuntimeError(f"Could not open {filename}")

    total_pixels = 0
    nan_pixels = 0

    for b in range(1, d.RasterCount + 1):

        band = d.GetRasterBand(b)

        xsize = band.XSize
        ysize = band.YSize

        block_x = 1024
        block_y = 1024

        for y in range(0, ysize, block_y):

            rows = min(block_y, ysize - y)

            for x in range(0, xsize, block_x):

                cols = min(block_x, xsize - x)

                arr = band.ReadAsArray(x, y, cols, rows)

                total_pixels += arr.size
                nan_pixels += np.count_nonzero(np.isnan(arr))

    d = None

    return 100.0 * nan_pixels / total_pixels


def remove_file_and_hdr(fn):

    if os.path.exists(fn):
        print("DELETE", fn)
        os.remove(fn)

    hdr = fn + ".hdr"

    if os.path.exists(hdr):
        print("DELETE", hdr)
        os.remove(hdr)


def main():

    percent_threshold = 1.0
    search_dir = "."

    for arg in sys.argv[1:]:

        if arg.startswith("--percent="):
            percent_threshold = float(arg.split("=", 1)[1])

        else:
            search_dir = arg

    search_cmd = f'ls -1r "{search_dir}"/*MRAP.bin'

    lines = get_filename_lines(search_cmd)

    if not lines:
        print("No *MRAP.bin files found.")
        return

    print("Directory:", os.path.abspath(search_dir))
    print("Threshold:", percent_threshold, "% NaN")
    print("Found", len(lines), "MRAP files")
    print()

    first_acceptable_date = None
    first_acceptable_file = None
    first_acceptable_pct = None

    for i, (line_date, filename) in enumerate(lines):

        full_path = os.path.join(search_dir, filename)

        print(f"[{i+1}/{len(lines)}] {line_date} {filename}")

        pct_nan = nan_percent(full_path)

        print(f"    NaN = {pct_nan:.4f}%")

        if pct_nan <= percent_threshold:

            first_acceptable_date = line_date
            first_acceptable_file = filename
            first_acceptable_pct = pct_nan

            print()
            print("FIRST ACCEPTABLE MRAP FOUND")
            print("Date:", line_date)
            print("File:", filename)
            print(f"NaN percent: {pct_nan:.4f}%")
            print(f"Threshold: {percent_threshold:.4f}%")
            print()

            break

    if first_acceptable_date is None:
        print()
        print("No MRAP met the threshold.")
        return

    response = input(
        f"Delete all MRAP dates before {first_acceptable_date} "
        f"(NaN={first_acceptable_pct:.4f}%)? [y/N] "
    ).strip().lower()

    if response != "y":
        print("Aborted.")
        return

    deleted = 0

    for line_date, filename in lines:

        if line_date >= first_acceptable_date:
            break

        full_path = os.path.join(search_dir, filename)

        remove_file_and_hdr(full_path)

        deleted += 1

    print()
    print("Deleted", deleted, "MRAP files")


if __name__ == "__main__":
    main()

