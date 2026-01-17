#!/usr/bin/env python3
'''20260116: unstage.py - Move directories from /ram/ (ramdisk) to /data/ram/
Usage: python unstage.py [directory] [destination]
       directory: defaults to current working directory, must be under /ram/
       destination: defaults to /data/ram/, alternative persistent storage location'''
import os
import sys
import subprocess
import shutil


def get_dir_size(path):
    """Calculate total size of directory in bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating directory size: {e}", file=sys.stderr)
        sys.exit(1)
    return total


def get_available_space(path):
    """Get available space on filesystem in bytes"""
    try:
        # Ensure parent directory exists for statvfs
        parent = path
        while not os.path.exists(parent):
            parent = os.path.dirname(parent)
            if parent == '/' or parent == '':
                parent = '/'
                break
        
        stat = os.statvfs(parent)
        return stat.f_bavail * stat.f_frsize
    except Exception as e:
        print(f"Error getting available space: {e}", file=sys.stderr)
        sys.exit(1)


def check_ramdisk_mounted():
    """Check if /ram/ is mounted"""
    if not os.path.ismount('/ram'):
        print("Error: /ram/ is not mounted", file=sys.stderr)
        sys.exit(1)


def main():
    # Get source directory
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    else:
        source_dir = os.getcwd()
    
    # Get destination base (optional second argument)
    if len(sys.argv) > 2:
        dest_base = sys.argv[2]
    else:
        dest_base = '/data/ram'
    
    # Normalize paths
    source_dir = os.path.abspath(source_dir)
    dest_base = os.path.abspath(dest_base)
    
    # Validate source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(source_dir):
        print(f"Error: Source path is not a directory: {source_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Check source directory is under /ram/
    if not source_dir.startswith('/ram/'):
        print(f"Error: Source directory must be under /ram/, got: {source_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Check ramdisk is mounted
    check_ramdisk_mounted()
    
    # Get base name (Y)
    # Remove /ram/ prefix and get first component
    relative_path = source_dir[len('/ram/'):]
    base_name = relative_path.split(os.path.sep)[0]
    
    # Destination directory
    dest_dir = os.path.join(dest_base, base_name)
    
    print(f"Unstaging: {source_dir} -> {dest_dir}")
    
    # Check available space on destination
    source_size = get_dir_size(source_dir)
    available_space = get_available_space(dest_base)
    
    print(f"Source size: {source_size / (1024**3):.2f} GB")
    print(f"Available space on {dest_base}: {available_space / (1024**3):.2f} GB")
    
    if source_size > available_space:
        print(f"Error: Insufficient space on {dest_base}", file=sys.stderr)
        print(f"Need: {source_size / (1024**3):.2f} GB, Available: {available_space / (1024**3):.2f} GB", file=sys.stderr)
        sys.exit(1)
    
    # Ensure destination parent directory exists
    os.makedirs(dest_base, exist_ok=True)
    
    # Use rsync to copy
    print("Copying files with rsync...")
    rsync_cmd = ['rsync', '-av', '--delete', source_dir + '/', dest_dir + '/']
    
    try:
        result = subprocess.run(rsync_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during rsync: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    
    # Verify sync was successful by comparing sizes
    dest_size = get_dir_size(dest_dir)
    if abs(source_size - dest_size) > source_size * 0.01:  # Allow 1% tolerance
        print(f"Warning: Size mismatch after sync. Source: {source_size}, Dest: {dest_size}", file=sys.stderr)
        print("Not deleting source directory due to verification failure", file=sys.stderr)
        sys.exit(1)
    
    # Delete source directory
    print(f"Removing source directory: {source_dir}")
    try:
        shutil.rmtree(source_dir)
    except Exception as e:
        print(f"Error removing source directory: {e}", file=sys.stderr)
        print(f"Data has been copied to {dest_dir} but source was not deleted", file=sys.stderr)
        sys.exit(1)
    
    print(f"Successfully unstaged {base_name} to {dest_base}")


if __name__ == '__main__':
    main()


