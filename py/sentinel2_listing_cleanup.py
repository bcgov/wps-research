#!/usr/bin/env python3
'''20261106 we pull listings of the NRCAN / Canada Sentinel products mirror ( hosted on AWS ). Sometimes these listings pile up, here's a script to remove most of them and free up space!
'''
import os
import re
from pathlib import Path

# Match filenames like: 20261106170002_objects.txt
PATTERN = re.compile(r"^(\d{14})_objects\.txt$")


def main():
    directory = Path.cwd()

    # Find matching files in the current directory only
    files_by_month = {}

    for path in directory.iterdir():
        if not path.is_file():
            continue

        match = PATTERN.match(path.name)
        if match:
            timestamp = match.group(1)
            month = timestamp[:6]

            files_by_month.setdefault(month, []).append(path)

    if not files_by_month:
        print("No matching files found.")
        return

    keep_files = []
    delete_files = []

    # Process each month
    for month, files in sorted(files_by_month.items()):
        files.sort(key=lambda p: p.name)

        # Keep the first (earliest) file in each month
        keep_files.append(files[0])
        delete_files.extend(files[1:])

    # Calculate space
    keep_size = sum(f.stat().st_size for f in keep_files)
    delete_size = sum(f.stat().st_size for f in delete_files)
    total_size = keep_size + delete_size

    # Display files being retained
    print("\n" + "=" * 65)
    print("FILES THAT WILL BE SAVED")
    print("=" * 65)

    for f in keep_files:
        print(f.name)

    # Display summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print(f"Matching files found : {len(keep_files) + len(delete_files):,}")
    print(f"Files to be saved    : {len(keep_files):,}")
    print(f"Files to be deleted  : {len(delete_files):,}")
    print()
    print(f"Space to be freed    : {delete_size:,} bytes ({delete_size / 1024**2:.2f} MB)")
    print(f"Space remaining      : {keep_size:,} bytes ({keep_size / 1024**2:.2f} MB)")
    print(f"Total current space  : {total_size:,} bytes ({total_size / 1024**2:.2f} MB)")

    if not delete_files:
        print("\nNothing to delete. All files are already the first capture of their month.")
        return

    # Confirm before deleting
    print("\n" + "=" * 65)
    answer = input(f"Delete {len(delete_files):,} files and free {delete_size / 1024**2:.2f} MB? [y/N]: ")

    if answer.strip().lower() != "y":
        print("\nDeletion cancelled. No files were deleted.")
        return

    # Perform deletion
    deleted = 0

    for f in delete_files:
        try:
            f.unlink()
            deleted += 1
        except OSError as e:
            print(f"ERROR deleting {f.name}: {e}")

    print(f"\nFinished. Successfully deleted {deleted:,} of {len(delete_files):,} files.")


if __name__ == "__main__":
    main()

