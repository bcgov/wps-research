'''20260116'''
#!/usr/bin/env python3
"""
unstage.py - Move directories from /ram/ (ramdisk) to /data/ram/ with parallel rsync
Usage: python unstage.py [directory] [destination] [--threads N]
       directory: defaults to current working directory, must be under /ram/
       destination: defaults to /data/ram/, alternative persistent storage location
       --threads: Override automatic thread detection
"""

import os
import sys
import subprocess
import shutil
import multiprocessing
import argparse
import time
import re
from threading import Thread, Lock


class ProgressMonitor:
    """Monitor and display transfer progress with ETA and speed"""

    def __init__(self, total_size, total_files):
        self.total_size = total_size
        self.total_files = total_files
        self.transferred_size = 0
        self.transferred_files = 0
        self.start_time = time.time()
        self.lock = Lock()
        self.last_update = 0

    def update(self, bytes_transferred, files_transferred=0):
        """Update progress with new bytes transferred"""
        with self.lock:
            self.transferred_size += bytes_transferred
            self.transferred_files += files_transferred

            # Throttle updates to once per second
            current_time = time.time()
            if current_time - self.last_update < 1.0:
                return

            self.last_update = current_time
            self.display()

    def set_transferred(self, bytes_transferred):
        """Set absolute transferred amount"""
        with self.lock:
            self.transferred_size = bytes_transferred
            current_time = time.time()
            if current_time - self.last_update < 1.0:
                return
            self.last_update = current_time
            self.display()

    def display(self):
        """Display current progress"""
        elapsed = time.time() - self.start_time
        if elapsed < 0.1:
            return

        # Calculate speed (MB/s)
        speed_mbps = (self.transferred_size / (1024 * 1024)) / elapsed

        # Calculate percentage
        if self.total_size > 0:
            percent = (self.transferred_size / self.total_size) * 100
        else:
            percent = 0

        # Calculate ETA
        if self.transferred_size > 0 and speed_mbps > 0:
            remaining_size = self.total_size - self.transferred_size
            eta_seconds = remaining_size / (speed_mbps * 1024 * 1024)
            eta_str = self.format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # Format output
        transferred_mb = self.transferred_size / (1024 * 1024)
        total_mb = self.total_size / (1024 * 1024)

        print(f"\rProgress: {percent:5.1f}% | "
              f"{transferred_mb:8.1f}/{total_mb:8.1f} MB | "
              f"{speed_mbps:6.1f} MB/s | "
              f"ETA: {eta_str:>10s}",
              end='', flush=True)

    def format_time(self, seconds):
        """Format seconds into human readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def finish(self):
        """Display final statistics"""
        elapsed = time.time() - self.start_time
        speed_mbps = (self.transferred_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0

        print(f"\n\nTransfer complete!")
        print(f"Total: {self.transferred_size / (1024**3):.2f} GB")
        print(f"Time: {self.format_time(elapsed)}")
        print(f"Average speed: {speed_mbps:.1f} MB/s")


def get_storage_type(path):
    """Detect storage device type and optimal thread count"""
    try:
        # Get the actual device
        result = subprocess.run(['df', path], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return 'unknown', 4

        device = lines[1].split()[0]

        # Check if it's a ramdisk/tmpfs
        if 'tmpfs' in device or 'ramfs' in device or path.startswith('/ram'):
            return 'ramdisk', multiprocessing.cpu_count()

        # Get the base device name (remove partition numbers)
        base_device = device.rstrip('0123456789').split('/')[-1]

        # Check rotational flag (0 = SSD, 1 = HDD)
        rotational_path = f'/sys/block/{base_device}/queue/rotational'
        if os.path.exists(rotational_path):
            with open(rotational_path, 'r') as f:
                is_rotational = f.read().strip() == '1'

            if is_rotational:
                # HDD: limited parallelization due to seek times
                return 'hdd', 2
            else:
                # SSD: benefits from parallelization
                # Check queue depth for better estimation
                nr_requests_path = f'/sys/block/{base_device}/queue/nr_requests'
                try:
                    with open(nr_requests_path, 'r') as f:
                        queue_depth = int(f.read().strip())
                        # Use queue depth as hint, cap at CPU count
                        threads = min(queue_depth // 32, multiprocessing.cpu_count(), 16)
                        return 'ssd', max(threads, 4)
                except:
                    return 'ssd', min(8, multiprocessing.cpu_count())

        # Check if network filesystem
        if device.startswith('//') or ':' in device:
            return 'network', 6

        return 'unknown', 4
    except Exception as e:
        print(f"Warning: Could not detect storage type: {e}", file=sys.stderr)
        return 'unknown', 4


def get_optimal_threads(source_path, dest_path):
    """Determine optimal number of threads based on source and destination storage"""
    source_type, source_threads = get_storage_type(source_path)
    dest_type, dest_threads = get_storage_type(dest_path)

    print(f"Source storage: {source_type} (recommended threads: {source_threads})")
    print(f"Destination storage: {dest_type} (recommended threads: {dest_threads})")

    # The bottleneck determines parallelism
    if source_type == 'ramdisk':
        # Reading from ramdisk is extremely fast, limited by destination
        optimal = dest_threads
    elif dest_type == 'ramdisk':
        # Writing to ramdisk is extremely fast, limited by source
        optimal = source_threads
    else:
        # Both are disks, use the minimum
        optimal = min(source_threads, dest_threads)

    return max(optimal, 1)


def get_dir_size_and_count(path):
    """Calculate total size and file count of directory"""
    total_size = 0
    file_count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath) and not os.path.islink(filepath):
                    try:
                        total_size += os.path.getsize(filepath)
                        file_count += 1
                    except (OSError, FileNotFoundError):
                        pass
    except Exception as e:
        print(f"Error calculating directory size: {e}", file=sys.stderr)
        sys.exit(1)
    return total_size, file_count


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


def parse_rsync_progress(line, monitor):
    """Parse rsync progress output and update monitor"""
    # Look for patterns like "1.23G  45%  123.45MB/s"
    # rsync --info=progress2 outputs: transferred size, percentage, speed
    match = re.search(r'(\d+(?:,\d+)*)\s+(\d+)%\s+([\d.]+[KMG]B/s)', line)
    if match:
        size_str = match.group(1).replace(',', '')
        try:
            transferred = int(size_str)
            monitor.set_transferred(transferred)
        except ValueError:
            pass


def rsync_with_progress(source_dir, dest_dir, monitor, use_parallel=False, num_threads=1):
    """Perform rsync with progress monitoring"""

    if use_parallel and num_threads > 1:
        # Use parallel rsync for better throughput
        print(f"Using parallel rsync with {num_threads} threads\n")
        return parallel_rsync(source_dir, dest_dir, monitor, num_threads)
    else:
        # Single-threaded rsync with progress
        print("Using single-threaded rsync with progress monitoring\n")
        rsync_cmd = [
            'rsync', '-av', '--info=progress2', '--no-inc-recursive',
            source_dir.rstrip('/') + '/',
            dest_dir.rstrip('/') + '/'
        ]

        try:
            process = subprocess.Popen(
                rsync_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                parse_rsync_progress(line, monitor)

            return_code = process.wait()

            if return_code != 0:
                stderr = process.stderr.read()
                print(f"\n\nError during rsync: {stderr}", file=sys.stderr)
                return False

            return True

        except Exception as e:
            print(f"\n\nError during rsync: {e}", file=sys.stderr)
            return False


def parallel_rsync(source_dir, dest_dir, monitor, num_threads):
    """Perform parallel rsync using multiple processes"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Get list of top-level items to process in parallel
    try:
        items = sorted(os.listdir(source_dir))
    except Exception as e:
        print(f"Error listing directory: {e}", file=sys.stderr)
        return False

    if not items:
        return True

    # Separate files and directories
    files = []
    dirs = []

    for item in items:
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            dirs.append(item)
        elif os.path.isfile(item_path):
            files.append(item)

    # Process directories in parallel, files in one batch
    tasks = []

    # Add directory tasks
    for d in dirs:
        tasks.append(('dir', d))

    # Add files as single task if any exist
    if files:
        tasks.append(('files', files))

    # Limit threads to number of tasks
    actual_threads = min(num_threads, len(tasks))

    print(f"Processing {len(dirs)} directories and {len(files)} files")

    # Track progress
    completed = 0
    total_tasks = len(tasks)

    with ProcessPoolExecutor(max_workers=actual_threads) as executor:
        futures = {}

        for task_type, task_data in tasks:
            if task_type == 'dir':
                future = executor.submit(
                    rsync_single_item,
                    os.path.join(source_dir, task_data),
                    os.path.join(dest_dir, task_data),
                    True
                )
                futures[future] = task_data
            else:
                # Process all files together
                future = executor.submit(
                    rsync_files_batch,
                    source_dir,
                    dest_dir,
                    task_data
                )
                futures[future] = 'files'

        # Monitor completion
        for future in as_completed(futures):
            completed += 1
            task_name = futures[future]

            try:
                success, size = future.result()
                if success:
                    monitor.update(size)
                else:
                    print(f"\nWarning: Failed to sync {task_name}")
            except Exception as e:
                print(f"\nError syncing {task_name}: {e}")
                return False

    return True


def rsync_single_item(source, dest, is_dir):
    """Rsync a single directory or file (for parallel processing)"""
    try:
        if is_dir:
            os.makedirs(dest, exist_ok=True)
            rsync_cmd = [
                'rsync', '-a', '--info=none',
                source.rstrip('/') + '/',
                dest.rstrip('/') + '/'
            ]
        else:
            rsync_cmd = [
                'rsync', '-a', '--info=none',
                source,
                dest
            ]

        result = subprocess.run(rsync_cmd, capture_output=True, check=True)

        # Calculate size transferred
        size = get_dir_size_and_count(source)[0] if is_dir else os.path.getsize(source)

        return True, size

    except Exception as e:
        return False, 0


def rsync_files_batch(source_dir, dest_dir, file_list):
    """Rsync a batch of files"""
    try:
        total_size = 0
        for filename in file_list:
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)

            rsync_cmd = ['rsync', '-a', '--info=none', source_file, dest_file]
            subprocess.run(rsync_cmd, capture_output=True, check=True)

            total_size += os.path.getsize(source_file)

        return True, total_size

    except Exception as e:
        return False, 0


def main():
    parser = argparse.ArgumentParser(description='Unstage data from /ram/ to /data/ram/')
    parser.add_argument('directory', nargs='?', help='Directory to unstage (default: current directory)')
    parser.add_argument('destination', nargs='?', default='/data/ram',
                        help='Destination base directory (default: /data/ram)')
    parser.add_argument('--threads', type=int, help='Number of threads (overrides auto-detection)')

    args = parser.parse_args()

    # Get source directory
    if args.directory:
        source_dir = args.directory
    else:
        source_dir = os.getcwd()

    # Get destination base
    dest_base = args.destination

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

    # Get base name
    relative_path = source_dir[len('/ram/'):]
    base_name = relative_path.split(os.path.sep)[0]

    # Destination directory
    dest_dir = os.path.join(dest_base, base_name)

    print(f"Unstaging: {source_dir} -> {dest_dir}\n")

    # Calculate size and file count
    print("Analyzing directory structure...")
    source_size, file_count = get_dir_size_and_count(source_dir)
    available_space = get_available_space(dest_base)

    print(f"Source size: {source_size / (1024**3):.2f} GB ({file_count:,} files)")
    print(f"Available space on {dest_base}: {available_space / (1024**3):.2f} GB")

    if source_size > available_space:
        print(f"\nError: Insufficient space on {dest_base}", file=sys.stderr)
        print(f"Need: {source_size / (1024**3):.2f} GB, Available: {available_space / (1024**3):.2f} GB", file=sys.stderr)
        sys.exit(1)

    # Determine optimal thread count
    if args.threads:
        num_threads = args.threads
        print(f"\nUsing manual thread count: {num_threads}")
    else:
        num_threads = get_optimal_threads(source_dir, dest_base)
        print(f"Optimal thread count: {num_threads}")

    # Ensure destination directory exists (but not the base path)
    # Only create if base path already exists
    if not os.path.exists(dest_base):
        print(f"Error: Destination base path does not exist: {dest_base}", file=sys.stderr)
        print(f"Please create {dest_base} first or it must be a mounted filesystem", file=sys.stderr)
        sys.exit(1)

    # Create the specific destination directory
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Destination directory created/verified: {dest_dir}")

    # Create progress monitor
    monitor = ProgressMonitor(source_size, file_count)

    # Perform rsync
    print("\n" + "="*70)
    use_parallel = num_threads > 1 and file_count > 10
    success = rsync_with_progress(source_dir, dest_dir, monitor, use_parallel, num_threads)

    if not success:
        print("\n\nTransfer failed!", file=sys.stderr)
        sys.exit(1)

    monitor.finish()

    # Verify sync was successful
    print("\nVerifying transfer...")
    dest_size, dest_files = get_dir_size_and_count(dest_dir)

    size_diff = abs(source_size - dest_size)
    if size_diff > source_size * 0.01:  # Allow 1% tolerance
        print(f"Warning: Size mismatch after sync.", file=sys.stderr)
        print(f"Source: {source_size:,} bytes, Dest: {dest_size:,} bytes", file=sys.stderr)
        print("Not deleting source directory due to verification failure", file=sys.stderr)
        sys.exit(1)

    print(f"Verification successful: {dest_files:,} files, {dest_size / (1024**3):.2f} GB")

    # Delete source directory
    print(f"\nRemoving source directory: {source_dir}")
    try:
        shutil.rmtree(source_dir)
    except Exception as e:
        print(f"Error removing source directory: {e}", file=sys.stderr)
        print(f"Data has been copied to {dest_dir} but source was not deleted", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Successfully unstaged {base_name} to {dest_base}")


if __name__ == '__main__':
    main()
