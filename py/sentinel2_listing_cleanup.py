#!/usr/bin/env python3
'''20261106 we pull listings of the NRCAN / Canada Sentinel products mirror ( hosted on AWS ). Sometimes these listings pile up, here's a script to remove most of them and free up space!
'''
import re
from pathlib import Path

# Match filenames like: 20261106170002_objects.txt
PATTERN = re.compile(r"^(\d{14})_objects\.txt$")


def format_size(size):
    """Return size in MB, GB and TB (using 1024-based units)."""
    mb = size / 1024**2
    gb = size / 1024**3
    tb = size / 1024**4

    return f"{mb:,.2f} MB | {gb:,.4f} GB | {tb:,.6f} TB"


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

    keep_files = set()
    delete_files = []

    # Keep the earliest file in every month
    for month, files in files_by_month.items():
        files.sort(key=lambda p: p.name)
        keep_files.add(files[0])

    # Also keep the most recent file overall
    all_files = [
        path
        for files in files_by_month.values()
        for path in files
    ]

    most_recent = max(all_files, key=lambda p: p.name)
    keep_files.add(most_recent)

    # Everything else is a deletion candidate
    delete_files = [
        path for path in all_files
        if path not in keep_files
    ]

    keep_files = sorted(keep_files, key=lambda p: p.name)
    delete_files.sort(key=lambda p: p.name)

    # Calculate sizes
    keep_size = sum(f.stat().st_size for f in keep_files)
    delete_size = sum(f.stat().st_size for f in delete_files)
    total_size = keep_size + delete_size

    total_files = len(all_files)

    # Display files being retained
    print("\n" + "=" * 75)
    print("FILES THAT WILL BE SAVED")
    print("=" * 75)

    for f in keep_files:
        print(f.name)

    print(f"\nMost recent overall: {most_recent.name}")

    # Display summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(f"Matching files found : {total_files:,}")
    print(f"Files to be saved    : {len(keep_files):,}")
    print(f"Files to be deleted  : {len(delete_files):,}")

    print("\nSPACE SUMMARY")
    print("-" * 75)

    print(f"Space to be freed    : {format_size(delete_size)}")
    print(f"Space remaining      : {format_size(keep_size)}")
    print(f"Total current space  : {format_size(total_size)}")

    if not delete_files:
        print("\nNothing to delete. All files are already retained.")
        return

    # Confirm before deleting
    print("\n" + "=" * 75)

    answer = input(
        f"Delete {len(delete_files):,} files "
        f"and free {format_size(delete_size)}? [y/N]: "
    )

    if answer.strip().lower() != "y":
        print("\nDeletion cancelled. No files were deleted.")
        return

    # Perform deletion
    deleted = 0
    errors = 0

    for f in delete_files:
        try:
            f.unlink()
            deleted += 1
        except OSError as e:
            errors += 1
            print(f"ERROR deleting {f.name}: {e}")

    print("\n" + "=" * 75)
    print("DELETION COMPLETE")
    print("=" * 75)

    print(f"Successfully deleted : {deleted:,}")
    print(f"Deletion errors      : {errors:,}")
    print(f"Files remaining      : {total_files - deleted:,}")
    print(f"Space freed          : {format_size(delete_size)}")
    print(f"Space retained       : {format_size(keep_size)}")


if __name__ == "__main__":
    main()

