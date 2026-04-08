#!/usr/bin/env python3
"""
20260408 Run sentinel2_swir_cloudmask_refine.py --mrap on each L2_T* folder
in the current directory, one at a time.
"""

import os
import sys
from pathlib import Path

def main():
    dirs = sorted(p for p in Path(".").iterdir() if p.is_dir() and p.name.startswith("L2_"))

    if not dirs:
        print("No L2_* directories found in current directory.")
        sys.exit(1)

    print(f"Found {len(dirs)} tile directories.\n")

    for i, d in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] Processing {d.name} ...")
        a = os.system(f"cd {d.name}; sentinel2_swir_cloudmask_refine.py --mrap")
        if a != 0:
            print(f"  WARNING: {d.name} exited with code {a}\n")

    print("\nAll tiles done.")

if __name__ == "__main__":
    main()


