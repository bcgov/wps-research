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
viirs/utils/download.py

This Python script automatically downloads VIIRS data from the LAADS DAAC portal using the
portal's V2 API.
"""

###################################################################################################
# Import statements
###################################################################################################

import os
import datetime
import numpy as np
from multiprocessing import get_context
from viirs.utils.laads_data_download_v2 import sync


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

    with get_context('spawn').Pool(processes=min(Nday, 16)) as p:
        list(p.map(loop_through_download, downloadDayList))