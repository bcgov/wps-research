'''
20250226: Added --L1 flag to optionally process L1_ folders instead of L2_ folders.

20230727 extract most recent available pixel (MRAP) from sentinel2 series.. assume tile based
(*) Difference from sentinel2_extract_swir.py:
    can't process in parallel (By tile, at least)
NOTE: assumes sentinel2_extract_cloudfree.py has been run.

20240618 NOTE: need to make this fully incremental. I.e., initialize buffer (per tile) with last avaiable MRAP date

20250602: note: the program can't be run in parallel currently ( will try scoping the extract() function inside the run_mrap function)
20250602: need to do the above before ____ ( DONE )

20250602: NB need to run sentinel2_extract_cloudfree_swir_nir.py BEFORE running this script ( if we've updated with sync_recent.py )

20250128: Added handling for duplicate acquisition timestamps - when multiple files have the same
          acquisition timestamp (3rd field), only the one with the latest processing timestamp
          (7th field) is used. This prevents NaN reversion errors in MRAP products.
'''

import warnings
warnings.filterwarnings("ignore")

from misc import args, run, hdr_fn, err, parfor
from envi import envi_update_band_names
from aws_download import aws_download
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import copy
import sys
import os

# Parse --L1 flag
L1_mode = '--L1' in sys.argv
if L1_mode:
    sys.argv.remove('--L1')
L_prefix = 'L1_' if L1_mode else 'L2_'


def run_mrap(gid):  # run MRAP on one tile
    print(gid, '-----------------------------------------------------------------------')
    my_bands, my_proj, my_geo, my_xsize, my_ysize, nbands, file_name  = {}, None, None, None, None, None, None

    def extract(file_name):
        nonlocal my_proj, my_geo, my_bands, my_xsize, my_ysize, nbands
        print("+r", file_name)

        d = gdal.Open(file_name)  # open the file brought in for this update step
        for i in range(1, d.RasterCount + 1):
            band = d.GetRasterBand(i)
            new_data = band.ReadAsArray().astype(np.float32)
            if i not in my_bands:  #initialize the band / first time
                my_bands[i] = new_data # new data for this band becomes the band
            else:
                nans, update = np.isnan(new_data), copy.deepcopy(my_bands[i])  # forgot copy before?
                update[~nans] = new_data[~nans]
                my_bands[i] = update
                # my_bands[i][~nans] = new_data[~nans]
        my_proj = d.GetProjection() if my_proj is None else my_proj
        my_geo = d.GetGeoTransform() if my_geo is None else my_geo
        if my_xsize is None:
            my_xsize, my_ysize, nbands = d.RasterXSize, d.RasterYSize, d.RasterCount
        d = None  # close input file brought in for this update step

        # write output file
        out_file_name, driver = file_name + '_MRAP.bin', gdal.GetDriverByName('ENVI')
        print("+w", out_file_name)
        if True: # not os.path.exists(out_file_name):  # skip files that already exist
            if os.path.exists(out_file_name):
                os.remove(out_file_name)
            print(out_file_name, my_xsize, my_ysize, nbands, gdal.GDT_Float32)
            stack_ds = driver.Create(out_file_name, my_xsize, my_ysize, nbands, gdal.GDT_Float32)
            stack_ds.SetProjection(my_proj)
            stack_ds.SetGeoTransform(my_geo)

            for i in range(1, nbands + 1):
                stack_ds.GetRasterBand(i).WriteArray(my_bands[i])
            stack_ds = None
            run('fh ' + out_file_name)  # fix envi header, then reproduce the band names
            envi_update_band_names(['envi_update_band_names.py',
                                    hdr_fn(file_name),
                                    hdr_fn(out_file_name)])
        else:
            pass
            # print(out_file_name, 'exists [SKIP WRITE]')

    # look for all the dates in this tile's folder and sort them in aquisition time

    def get_filename_lines(search_cmd):
        lines = [x.strip() for x in os.popen(search_cmd).readlines()]
        lines = [x.split(os.path.sep)[-1].split('_') for x in lines]
        lines = [[x[2], x] for x in lines]
        lines.sort()
        return [[x[0], '_'.join(x[1])] for x in lines]  # [date string, full S2 data filename]

    def deduplicate_by_processing_time(lines):
        '''
        When multiple files have the same acquisition timestamp (3rd field, index 2),
        keep only the one with the latest processing timestamp (7th field, index 6).

        Input: list of [acquisition_timestamp, full_filename] pairs
        Output: deduplicated list with same format
        '''
        if not lines:
            return lines

        # Group files by acquisition timestamp
        by_acq_time = {}
        for [acq_ts, filename] in lines:
            if acq_ts not in by_acq_time:
                by_acq_time[acq_ts] = []
            by_acq_time[acq_ts].append(filename)

        # For each acquisition timestamp, select the file with latest processing timestamp
        result = []
        for acq_ts in sorted(by_acq_time.keys()):
            files = by_acq_time[acq_ts]
            if len(files) == 1:
                result.append([acq_ts, files[0]])
            else:
                # Multiple files with same acquisition timestamp
                # Sort by processing timestamp (7th field, index 6) and pick the latest
                def get_processing_ts(filename):
                    parts = filename.split('_')
                    if len(parts) >= 7:
                        return parts[6]  # 7th field (0-indexed: 6)
                    return ''

                files_sorted = sorted(files, key=get_processing_ts)
                selected = files_sorted[-1]  # Latest processing timestamp

                print(f"  Duplicate acquisition timestamp {acq_ts}:")
                for f in files:
                    marker = " [SELECTED]" if f == selected else " [SKIPPED]"
                    print(f"    {f}{marker}")

                result.append([acq_ts, selected])

        return result

    data_lines = get_filename_lines("ls -1r " + L_prefix + gid + os.path.sep + "S2*_cloudfree.bin")
    mrap_lines = get_filename_lines("ls -1r " + L_prefix + gid + os.path.sep + "S2*_cloudfree.bin_MRAP.bin")

    # Deduplicate data_lines - keep only files with latest processing timestamp for each acquisition timestamp
    original_count = len(data_lines)
    data_lines = deduplicate_by_processing_time(data_lines)
    if len(data_lines) < original_count:
        print(f"Deduplicated data files: {original_count} -> {len(data_lines)}")

    last_good_mrap_date = None   # last MRAP date that's still good.. i.e. before the first data file that doesn't have an MRAP file
    last_good_mrap_file = None
    data_dates_set = set([line[0] for line in data_lines])
    mrap_dates_set = set([line[0] for line in mrap_lines])
    mrap_date_lookup = {line[0]: line[1] for line in mrap_lines}
    # print(data_dates_set)  # have to match MRAP files and data files by collection timestamp in case original data reprocessed.
    for [mrap_date, line] in mrap_lines:
        if mrap_date not in data_dates_set:
            print(line)
            err("found MRAP date: S2x_cloudfree.bin_MRAP.bin without S2x_cloudfree.bin file")

    print("DATA lines", len(data_lines))
    for [line_date, line] in data_lines:
        if gid != line.split("_")[5]:
            err('gid sanity check failed')
        extract_path = L_prefix +  gid + os.path.sep + line
        print('**' + line_date + " " + extract_path)

    print("MRAP lines", len(mrap_lines))
    for [mrap_date, line] in mrap_lines: # check for latest MRAP file and "seed" with that
        gid = line.split("_")[5]
        extract_path = L_prefix +  gid + os.path.sep + line
        print('**' + mrap_date + " " + extract_path)

    # find the last mrap date ( if applicable ) that's still good ( before first data file without MRAP file )
    for [line_date, line] in data_lines:
        if last_good_mrap_date is None or line_date in mrap_dates_set:
                try:
                    last_good_mrap_file = mrap_date_lookup[line_date]
                    last_good_mrap_date = line_date
                except:
                    pass # might not have any mrap files generated.

        else:
            # print("line_date", line_date)
            # print("mrap_dates_set", mrap_dates_set)
            break

    print("last_good_mrap_date", last_good_mrap_date)

    last_mrap_date = mrap_lines[-1][0] if len(mrap_lines) > 0 else None
    last_mrap_file = L_prefix + gid + os.path.sep + mrap_lines[-1][1] if len(mrap_lines) > 0 else None
    print("last_mrap_date", last_mrap_date)
    last_data_date = data_lines[-1][0] if len(data_lines) > 0 else None

    # last MRAP date string, that also has a data file with the same date string
    # load a SEED if there are MRAP files, but data files without corresponding MRAP file
    if last_mrap_date is not None and last_data_date is not None and last_data_date > last_good_mrap_date:
        print("load SEED")
        print("+r", L_prefix +  gid + os.path.sep + last_good_mrap_file)  # load / seed from "most recent" MRAP file
        d = gdal.Open(L_prefix +  gid + os.path.sep + last_good_mrap_file)  # open the file brought in for this update step
        my_bands = {i: d.GetRasterBand(i).ReadAsArray().astype(np.float32) for i in range(1, d.RasterCount + 1)}
        my_proj, my_geo = d.GetProjection(), d.GetGeoTransform()
        my_xsize, my_ysize, nbands = d.RasterXSize, d.RasterYSize, d.RasterCount
        # print(my_proj, my_geo, my_xsize, my_ysize, nbands)

    print("run extract:")  # run extract() on data files later than the last MRAP date
    for [line_date, line] in data_lines:
        gid = line.split("_")[5]
        extract_path = L_prefix +  gid + os.path.sep + line
        if last_good_mrap_date is None or ( last_good_mrap_date is not None and line_date > last_good_mrap_date):
            print('extract(' + extract_path + ')')
            extract(extract_path)

    # THIS PART SHOULD MARK ( NEEDING REFRESHING ) MRAP COMPOSITES ( MERGED PRODUCTS ) THAT NEED REFRESHING / BY DELETING THEM !!!!

    '''
    print("check sorting order")
    for [line_date, line] in lines:
        print("mrap " + line)
    '''
    print("done " + gid + "------------------------------------")

if __name__ == "__main__":
    if len(args) < 2:
        gids = []
        dirs = [x.strip() for x in os.popen('ls -1d ' + L_prefix + '*').readlines()]
        for d in dirs:
            print(d)
            w = d.split('_')
            if len(w) != 2:
                err('unexpected folder name')

            gid = w[1]
            gids += [gid]
            # should run on this frame here.
            # but check for cloud-free data first
        # err("python3 sentinel2_mrap.py [sentinel-2 gid] # [optional: yyyymmdd 'maxdate' parameter] ")
        def f(fn):
            run_mrap(fn)
        parfor(f, gids, 1) #  int(mp.cpu_count()))
    else:
        run_mrap(args[1])  # single tile mode: no mosaicing
