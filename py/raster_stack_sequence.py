'''20260127 raster stack sequence: 

stack numbered bin files ( regular # of digits prefix ) in the present directory into one file.

'''
#!/usr/bin/env python3

import os
import re
import sys
import subprocess


def main():
    # Find all .bin files
    bin_files = [f for f in os.listdir(".") if f.endswith(".bin")]

    if not bin_files:
        print("ERROR: No .bin files found")
        sys.exit(1)

    # Regex: leading digits followed by anything, ending in .bin
    pattern = re.compile(r"^(\d+).*\.bin$")

    files_with_prefix = []
    prefix_lengths = set()

    for f in bin_files:
        m = pattern.match(f)
        if not m:
            continue
        prefix = m.group(1)
        prefix_lengths.add(len(prefix))
        files_with_prefix.append((int(prefix), f))

    if not files_with_prefix:
        print("ERROR: No .bin files with numeric prefix found")
        sys.exit(1)

    if len(prefix_lengths) != 1:
        print("ERROR: Files have different digit-length numeric prefixes:")
        print(sorted(prefix_lengths))
        sys.exit(1)

    # Sort by numeric prefix
    files_with_prefix.sort(key=lambda x: x[0])
    file_names = [f for _, f in files_with_prefix]

    print("Stacking files in order:")
    for f in file_names:
        print(" ", f)

    # Build and run command
    cmd = " ".join(["raster_stack.py"] + file_names + ["stack.bin"])
    print("\nRunning command:")
    print(cmd)

    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()

