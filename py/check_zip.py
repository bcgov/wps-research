'''20250601 check_zip.py: check integrity of zip archives in parallel
'''
import subprocess
import concurrent.futures
import threading
import os

print_lock = threading.Lock()
bad_files = []

def check_zip(filepath):
    result = subprocess.run(['zip', '-T', filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with print_lock:
        if result.returncode != 0:
            print(f"[BAD]  {filepath}")
            bad_files.append(filepath)
        else:
            print(f"[GOOD] {filepath}")

def check_tar_gz(filepath):
    result = subprocess.run(['gzip', '-t', filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with print_lock:
        if result.returncode != 0:
            print(f"[BAD]  {filepath}")
            bad_files.append(filepath)
        else:
            print(f"[GOOD] {filepath}")

def check_tar(filepath):
    result = subprocess.run(['tar', '-tf', filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with print_lock:
        if "error" in result.stderr.lower():
            print(f"[BAD]  {filepath}")
            bad_files.append(filepath)
        else:
            print(f"[GOOD] {filepath}")

def find_files(ext):
    result = subprocess.run(['find', '.', '-name', ext], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip().split('\n') if result.stdout else []

def main():
    cpu_count = os.cpu_count() or 4  # Fallback to 4 if cpu_count is None

    # Collect files
    zip_files = find_files("*.zip")
    tgz_files = find_files("*.tar.gz")
    tar_files = find_files("*.tar")

    # Build jobs
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        for f in zip_files:
            if f: executor.submit(check_zip, f)
        for f in tgz_files:
            if f: executor.submit(check_tar_gz, f)
        for f in tar_files:
            if f: executor.submit(check_tar, f)

    # Final summary
    print("\nSummary:")
    if bad_files:
        print("Bad files found:")
        for bad in bad_files:
            print(bad)
    else:
        print("All archive files passed integrity checks.")

if __name__ == "__main__":
    main()

