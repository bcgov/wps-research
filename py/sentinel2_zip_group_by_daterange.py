#!/usr/bin/env python3
"""
sentinel2_zip_group_by_daterange.py [yyyymmdd1] [yyyymmdd2]

Recursively finds all Sentinel-2 zip files (S2*.zip) and moves those
where the timestamp in the third underscore-separated field falls within
the specified date range (inclusive) to a new folder named yyyymmdd1_yyyymmdd2.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime


def parse_date(date_str):
    """Parse a date string in yyyymmdd format."""
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None


def extract_date_from_filename(filename):
    """
    Extract the date from a Sentinel-2 filename.
    
    The filename format has underscore-separated fields, where the third field
    is a timestamp. The timestamp is split on 'T', and the first part is the date.
    
    Example: S2A_MSIL1C_20230415T103021_N0509_R108_T32TQM_20230415T141051.zip
    Third field: 20230415T103021
    Date part: 20230415
    """
    # Remove .zip extension
    basename = filename[:-4] if filename.lower().endswith('.zip') else filename
    
    # Split by underscore
    parts = basename.split('_')
    
    if len(parts) < 3:
        return None
    
    # Get the third field (index 2)
    timestamp_field = parts[2]
    
    # Split on 'T' to get the date part
    timestamp_parts = timestamp_field.split('T')
    
    if len(timestamp_parts) < 1:
        return None
    
    date_str = timestamp_parts[0]
    
    # Validate it's a proper date
    return parse_date(date_str)


def find_sentinel2_zips(start_path):
    """Recursively find all S2*.zip files."""
    matches = []
    for root, dirs, files in os.walk(start_path):
        for filename in files:
            if filename.startswith('S2') and filename.lower().endswith('.zip'):
                matches.append(Path(root) / filename)
    return matches


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} yyyymmdd1 yyyymmdd2")
        print("  yyyymmdd1: Start date (inclusive)")
        print("  yyyymmdd2: End date (inclusive)")
        sys.exit(1)
    
    date1_str = sys.argv[1]
    date2_str = sys.argv[2]
    
    # Parse the date arguments
    start_date = parse_date(date1_str)
    end_date = parse_date(date2_str)
    
    if start_date is None:
        print(f"Error: Invalid start date format '{date1_str}'. Expected yyyymmdd.")
        sys.exit(1)
    
    if end_date is None:
        print(f"Error: Invalid end date format '{date2_str}'. Expected yyyymmdd.")
        sys.exit(1)
    
    # Ensure start_date <= end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date
        date1_str, date2_str = date2_str, date1_str
    
    # Create output directory name
    output_dir_name = f"{date1_str}_{date2_str}"
    output_dir = Path.cwd() / output_dir_name
    
    # Find all Sentinel-2 zip files
    print(f"Searching for Sentinel-2 zip files...")
    all_zips = find_sentinel2_zips(Path.cwd())
    print(f"Found {len(all_zips)} Sentinel-2 zip file(s)")
    
    # Filter by date range
    matching_files = []
    for zip_path in all_zips:
        file_date = extract_date_from_filename(zip_path.name)
        if file_date is not None:
            if start_date <= file_date <= end_date:
                matching_files.append(zip_path)
    
    print(f"Found {len(matching_files)} file(s) within date range {date1_str} to {date2_str}")
    
    if not matching_files:
        print("No files to move.")
        sys.exit(0)
    
    # Create output directory if needed
    output_dir.mkdir(exist_ok=True)
    print(f"Moving files to: {output_dir}")
    
    # Move matching files
    moved_count = 0
    for zip_path in matching_files:
        dest_path = output_dir / zip_path.name
        
        # Handle case where file might already be in the destination
        if zip_path.resolve() == dest_path.resolve():
            print(f"  Skipping (already in destination): {zip_path.name}")
            continue
        
        # Handle duplicate filenames
        if dest_path.exists():
            print(f"  Warning: {zip_path.name} already exists in destination, skipping")
            continue
        
        try:
            shutil.move(str(zip_path), str(dest_path))
            print(f"  Moved: {zip_path} -> {dest_path}")
            moved_count += 1
        except Exception as e:
            print(f"  Error moving {zip_path}: {e}")
    
    print(f"\nDone. Moved {moved_count} file(s) to {output_dir}")


if __name__ == "__main__":
    main()

