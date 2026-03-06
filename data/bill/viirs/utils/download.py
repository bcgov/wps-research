# Thanks Tianran for this amazing file.
# Author: Tianran Zhang
# Contact: tianran.zhang@kcl.ac.uk
# Date: 2019-07-03 17:01:30
# Last Modified by: Matthew Ansell
# Date: 2023-10-20


###################################################################################################
# Description
###################################################################################################

"""
viirs/utils/download_vnp14.py

This Python script automatically downloads VIIRS data from the LAADS DAAC portal using the
portal's V2 API.

A multiprocessing approach is used to download the files; this means that if multiple
days are specified by the user, the script will launch multiple processes for simultaneous
downloading of days' worth of VIIRS data.

The primary advantage of using the V2 API is that it gives the user more control and
flexibility over the filtering criteria that can be applied to the desired VIIRS data; in
particular, restricting downloaded data by geographical region.

The RESTful API used in this code ("details") returns a JSON object given a query provided
through the URL that is built from user-provided variables. These variables are defined and
described in the "Variables" section of this script.

Currently, the code supports regions specified by Administrative Area name (such as "Canada").
Although the V2 API also supports specifying regions in other ways, such as by bounding box
(lat/lon) or closed convex polygon (cartesian coordinate pairs), these other ways are NOT
currently implemented.

The script uses a slightly modified version of the original VIIRS Python data download
script, "laads_data_download.py". This version has been modified to always assume that the
URL will return a JSON object, and that this JSON object will specify the download URL
of each VIIRS data file using the "downloadsLink" field.

Since this script uses the "sync()" method in "laads_data_download.py" to do the actual
downloading of the VIIRS data, any data files that already exist in the download path
specified by the "dirGFAS" folder will be skipped.

NOTE: In rare occasions the script will fail to download a file due to network errors, but will
still write a 0 KB size file to the download folder. A workaround for this is to delete the
0 KB files after the script has finished executing, and then re-run the script. Existing data
files will be skipped, and the deleted 0 KB files will be re-downloaded.
"""

###################################################################################################
# Import statements
###################################################################################################

import os
import datetime
import numpy as np
from multiprocessing import Pool
from viirs.utils.laads_data_download_v2 import sync
# import earthaccess


###################################################################################################
# Variables
###################################################################################################


global downloadStartDay, downloadEndDay

# These integer variables define the date ranges for the downloads, in YYYYMMDD format.
# NOTE: downloadEndDay is EXCLUSIVE.
# These variables are used for VIIRS LAADS DAAC downloads.
downloadStartDay  = 20230301
downloadEndDay    = 20251031

# This string variable specifies the LAADS DAAC authentication token to use for the download.
# Used for VIIRS data from LAADS DAAC.
# TODO: Add LAADS DAAC token here.
with open('/data/.tokens/laads', 'r') as fh:
    laadsToken = fh.read().strip()
# This string variable specifies the path to the folder to use for the downloaded VIIRS data.
# Subdirectories will be created within this folder using the following structure:\
#
# [dirGFAS]/[product]/YYYY/DDD/
#
# ... where: [product] is the VIIRS product name specified in the "product" variable below;
#            YYYY is the string formatted four-digit year of the current download day; and
#            DDD is the string formatted three-digit Julian day of the year.
# Used for VIIRS L1B data from LAADS DAAC.
dirGFAS = '/data/bill/viirs_aoi/'

# This string specifies the Administrative Area region name (such as "Canada") to restrict
# the downloaded VIIRS data to.
# Used for VIIRS data from LAADS DAAC.
regions = "Canada"


# This string specifies the VIIRS product name to download (such as VNP02IMG, VNP02MOD, VNP03IMG or
# VNP14IMG).
product = "VNP14IMG"


###################################################################################################
# Methods
###################################################################################################

# This method parses an integer in the form YYYYMMDD and returns a datetime.datetime object based
# on this representation.
def ymd_to_datetime(ymd):
    if isinstance(ymd, int):
        ymd = str(ymd)
    return datetime.datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))


# This method returns a half-open datetime interval specified by the downloadStartDay and
# downloadEndDay variable (exclusive on downloadEndDay).
def get_download_day_list():
    global downloadStartDay, downloadEndDay
    startdt  = ymd_to_datetime(downloadStartDay)
    enddt    = ymd_to_datetime(downloadEndDay)
    interval = datetime.timedelta(days=1)
    return np.arange(startdt, enddt, interval, dtype=datetime.datetime)


# This method builds the V2 API URL to use based on the current "downloadDay", as well as based
# on the "product" and "regions" parameters.
# It then calls the "sync()" method in "laads_data_download_v2.py" to perform the actual
# downloading/syncing of VIIRS data.
def loop_through_download(downloadDay):
    
    print("loop_through_download(): Retrieving data for %s on %s . . ." % (product, downloadDay.strftime("%m/%d/%Y")))
    

    downloadUrl = (
        f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?"
        f"products={product}&"
        f"temporalRanges={downloadDay.year}-{downloadDay.timetuple().tm_yday}&"
        "regions=%5BBBOX%5DN59.944776%20S51.666160%20E-113.803377%20W-130.518259"
    )

    
    downloadPath = (dirGFAS +
                    product +
                    "/"
                    + "%04d" % (downloadDay.year) +
                    "/" + "%03d" % (downloadDay.timetuple().tm_yday)
                    )
    
    print()
    print(f'loop_through_download(): downloadUrl is:')
    print(downloadUrl)
    print()
    print(f'loop_through_download(): downloadPath is:')
    print(downloadPath)
    print()
    
    if not os.path.exists(downloadPath):
        os.makedirs(downloadPath)
    sync(downloadUrl, downloadPath, laadsToken)

if __name__ == "__main__":

    downloadDayList = get_download_day_list()
    Nday = len(downloadDayList)

    with Pool(processes=min(Nday, 16)) as p:   # cap workers; adjust 4 as needed
        list(p.map(loop_through_download, downloadDayList))