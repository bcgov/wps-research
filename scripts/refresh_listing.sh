#!/usr/bin/bash
# refresh sentinel-2 data listing ( run this in home folder using Cron ) 
export PATH=/home/ash/GitHub/wps-research/cpp:/home/ash/GitHub/bin/bin:$PATH

cd /data/.listing/

timestamp=$(date +"%Y%m%d_%H%M%S")

# Define a hidden log file name (starts with a dot)
log_file=".log_$timestamp.txt"

# Example: write to the hidden log
update_listing.py >> "$log_file"
