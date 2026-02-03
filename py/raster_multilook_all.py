'''20260203 apply multilook.cpp to all .bin files in provided directory
'''
#!/usr/bin/env python3

import os
import sys
from misc import run, err


def main():
    if len(sys.argv) != 3:
        err("raster_multilook_all.py [input directory] [window size]")

    input_dir = sys.argv[1]
    window_size = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    bin_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".bin"))

    if not bin_files:
        print("No .bin files found.")
        return

    print(f"Found {len(bin_files)} .bin files")
    print(f"Multilook window size: {window_size}\n")

    for fname in bin_files:
        in_path = os.path.join(input_dir, fname)
        cmd = (f"multilook " f"\"{in_path}\" " f"{window_size} ")
        try:
            run(cmd)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print("\nDone.")

if __name__ == "__main__":
    main()
