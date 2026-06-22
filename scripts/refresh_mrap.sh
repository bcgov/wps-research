#!/usr/bin/bash
# refresh MRAP product ( run this in home folder using Cron ) 
export PATH=/usr/local/bin:/home/ash/GitHub/wps-research/cpp:/home/ash/GitHub/bin/bin:$PATH

cd /data/mrap_bc/

timestamp=$(date +"%Y%m%d_%H%M%S")

# Define a hidden log file name (starts with a dot)
log_file=".log_$timestamp.txt"

# Example: write to the hidden log
sentinel2_mrap_update.py  >> "$log_file" 2>&1

# stack the new data and reboot the server!

fire_mapping_build_and_serve_stack.py >> ".log_fire_mapping_build_and_serve_$(date +%Y%m%d_%H%M%S).txt" 2>&1
