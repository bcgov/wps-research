'''20260203 apply multilook.cpp to all .bin files in provided directory
'''
#!/usr/bin/env python3

import os
import sys
from misc import run


def usage():
    print(
        "Usage:\n"
        "  raster_multilook_all.py <input_dir> <output_dir> <window_size>\n\n"
        "Example:\n"
        "  raster_multilook_all.py ./bins ./multilooked 5"
    )
    sys.exit(1)


def main():
    if len(sys.argv) != 4:
        usage()

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    window_size = sys.argv[3]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    bin_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".bin")
    )

    if not bin_files:
        print("No .bin files found.")
        return

    print(f"Found {len(bin_files)} .bin files")
    print(f"Multilook window size: {window_size}\n")

    for fname in bin_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        cmd = (
            f"multilook "
            f"\"{in_path}\" "
            f"{window_size} "
            f"> \"{out_path}\""
        )

        print(f"Processing: {fname}")
        print(f"Running: {cmd}")

        try:
            run(cmd)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()

