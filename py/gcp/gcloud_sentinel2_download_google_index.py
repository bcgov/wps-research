'''DEPRECATED:

download based on a selection of rows from the google cloud index file..
previous version searched that file based on entries in our esa-format download script

1. perform csv_split on the index file subselection to obtain:

-rw-rw-r-- 1      11K Jul 18 11:29 index.csv
drwxrwxr-x 2     4.0K Jul 18 11:30 .
-rw-rw-r-- 1      517 Jul 18 11:30 index.csv_WEST_LON
-rw-rw-r-- 1      270 Jul 18 11:30 index.csv_TOTAL_SIZE
-rw-rw-r-- 1      478 Jul 18 11:30 index.csv_SOUTH_LAT
-rw-rw-r-- 1      740 Jul 18 11:30 index.csv_SENSING_TIME
-rw-rw-r-- 1     1.6K Jul 18 11:30 index.csv_PRODUCT_ID
-rw-rw-r-- 1      479 Jul 18 11:30 index.csv_NORTH_LAT
-rw-rw-r-- 1      165 Jul 18 11:30 index.csv_MGRS_TILE
-rw-rw-r-- 1      920 Jul 18 11:30 index.csv_GRANULE_ID
-rw-rw-r-- 1       48 Jul 18 11:30 index.csv_GEOMETRIC_QUALITY_FLAG
-rw-rw-r-- 1      743 Jul 18 11:30 index.csv_GENERATION_TIME
-rw-rw-r-- 1      516 Jul 18 11:30 index.csv_EAST_LON
-rw-rw-r-- 1      929 Jul 18 11:30 index.csv_DATATAKE_IDENTIFIER
-rw-rw-r-- 1      195 Jul 18 11:30 index.csv_CLOUD_COVER
-rw-rw-r-- 1     2.9K Jul 18 11:30 index.csv_BASE_URL
'''
import os
import sys
def err(m):
    print("Error: " + str(m)); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0: err("command failed: " + c)

print("read google data files..")
g_base = [x.strip() for x in open("index.csv_BASE_URL").readlines()]
if g_base[0] != "BASE_URL":
    err("expected: BASE_URL")
g_base = g_base[1:]

g_prod = [x.strip() for x in open("index.csv_PRODUCT_ID").readlines()]
if g_prod[0] != "PRODUCT_ID":
    err("expected: PRODUCT_ID")
g_prod = g_prod[1:]

print("creating lookup..")
g_prod_to_url = {g_prod[i]: g_base[i] for i in range(len(g_base))}

f = open("gcloud_download.sh", "wb")

for p in g_prod_to_url:
    print(g_prod_to_url[p]) 
    f.write(("test ! -f " + p + ".SAFE && " +
             "gsutil -m cp -r " + g_prod_to_url[p] + " ./ " +
             "> " + p + ".SAFE_stdout.txt 2> " + p + ".SAFE_stderr.txt" + "\n").encode())
f.close()

# step 2: cross-reference with index file..
# create a list of download commands e.g.:
# gsutil cp -r gs://gcp-public-data-sentinel-2/tiles/44/X/MQ/S2B_MSIL1C_20200526T101559_N0209_R065_T44XMQ_20200526T123451.SAFE ./
