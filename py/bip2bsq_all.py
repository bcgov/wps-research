#!/usr/bin/env python3

import os
import subprocess

from misc import parfor


def run_bip2bsq(bin_file):
    print(f"Running bip2bsq on {bin_file}")
    try:
        subprocess.run(
            ["bip2bsq", bin_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR processing {bin_file}: {e}")


def main():
    # Find all .bin files in the current directory
    bin_files = sorted(
        f for f in os.listdir(".") if f.lower().endswith(".bin")
    )

    if not bin_files:
        print("No .bin files found.")
        return

    print(f"Found {len(bin_files)} .bin files")

    # Run using exactly 32 threads
    parfor(run_bip2bsq, bin_files, n_thread=32)


if __name__ == "__main__":
    main()



