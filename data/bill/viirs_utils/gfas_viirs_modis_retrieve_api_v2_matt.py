# Thanks Tianran for this amazing file.
# Author: Tianran Zhang
# Contact: tianran.zhang@kcl.ac.uk
# Date: 2019-07-03 17:01:30
# Last Modified by: Matthew Ansell
# Date: 2023-10-20

                                  #########
                                 ##       ###
                        #########      ######
                    ####             ##
                  ###                 ##
               ###              ########
            ####    #######    ##
       #####   #####     ##  ###
       ########          ########


###################################################################################################
# Description
###################################################################################################

"""
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
from laads_data_download_v2 import sync
# import earthaccess


###################################################################################################
# Variables
###################################################################################################


global downloadStartDay, downloadEndDay

# These integer variables define the date ranges for the downloads, in YYYYMMDD format.
# NOTE: downloadEndDay is EXCLUSIVE.
# These variables are used for VIIRS LAADS DAAC downloads.
downloadStartDay  = 20230101
downloadEndDay    = 20260101

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
dirGFAS = '/data/bill/viirs/'

# This string specifies the Administrative Area region name (such as "Canada") to restrict
# the downloaded VIIRS data to.
# Used for VIIRS data from LAADS DAAC.
regions = "Canada"


# This string specifies the VIIRS product name to download (such as VNP02IMG, VNP02MOD, VNP03IMG or
# VNP14IMG).
product = "VNP14IMG"

# These string variables specify temporal ranges.
# Used for VIIRS L2 Earthdata downloads.
#start_date_l2 = '2021-04'
#end_date_l2 = '2021-10'
#start_date_l2 = '2021-07-14'
#end_date_l2 = '2021-07-14'

# This tuple defines a spatial boundary in the form (lower_left_lon, lower_left_lat,
# upper_right_lon, upper_right_lat). Used for VIIRS L2 Earthdata downloads.
# Let's define one for Canada.
# Used for VIIRS L2 Earthdata downloads.

# This Boolean specifies whether to use cloud-hosted data or not.
# Used for VIIRS L2 Earthdata downloads.
# cloud_hosted = True

# This string specifies an identifier for the VIIRS data type to be downloaded.
# Used for VIIRS L2 Earthdata downloads.
# doi = '10.5067/VIIRS/VNP14IMG.002'

# This int specifies the number of download threads to use to download the data.
# Used for VIIRS L2 Earthdata downloads.
# threads = 1


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
        f"regions=%5BBBOX%5DN53.2120%20S52.1755%20E-124.3658%20W-126.0722"
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

    print(f"DEBUG URL: {downloadUrl}")
    sync(downloadUrl, downloadPath, laadsToken)

if __name__ == "__main__":
    
    # Use the old LAADS DAAC way if we are not downloading VIIRS L1B Collection 2 data.
    #if product != 'VNP14IMG':
    downloadDayList = get_download_day_list()
    Nday = len(downloadDayList)
    # p = Pool()
    # p.imap(loop_through_download, downloadDayList)
    # p.close()
    # p.join()

    with Pool(processes=min(Nday, 64)) as p:   # cap workers; adjust 4 as needed
        list(p.map(loop_through_download, downloadDayList))
    
    # This code was used to experiment with trying to download VNP14IMG C002 data from
    # the NASA Earthdata portal.
    #else:
    #    earthaccess_download(bounding_box=bounding_box,
    #                         cloud_hosted=cloud_hosted,
    #                         doi=doi,
    #                         threads=threads
    #                         )



"""
def earthaccess_download(bounding_box: tuple,
                         cloud_hosted: bool,
                         doi: str,
                         threads: int
                         ):
    # This method handles downloading of data from the NASA Earthdata portal.
    
    # Log in to the Earthdata portal interactively; do not persist login information.
    
    print('Logging in to the Earthdata portal . . .')
    auth = earthaccess.login(persist=False,
                             strategy='interactive'
                             )
    
    # Let's grab some search results based on our parameters.
    print('Grabbing search results based on our parameters . . .')
    results = earthaccess.search_data(
        bounding_box=bounding_box,
        cloud_hosted=cloud_hosted,
        doi=doi,
        temporal=(start_date_l2,
                  end_date_l2
                  ),
        threads=threads
    )
    
    # Let's download the search results to a local folder.
    print(f'Downloading data to {dirGFAS} . . .')
    files = earthaccess.download(granules=results,
                                 local_path=dirGFAS
                                 )
    
    # Print the files to the console and exit.
    print(files)
"""