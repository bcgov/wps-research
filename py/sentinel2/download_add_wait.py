'''input: .sentinel2_download.sh (etc)..
...from py/find_sentinel2.py

add a wait interval between downloads / polls

was running this on .sentinel2_downloads.sh, followed by work_queue.py with two threads as two simultaneous downloads are allowed from copernicus'''
import os
import sys

lines = open("download.sh").readlines()
lines = [x.strip() for x in lines]

for line in lines:
    line = line.replace("&&", " && sleep 30s &&").replace(" & ", "; ")
    print(line)
