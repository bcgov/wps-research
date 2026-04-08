"""Flatten all files in the directory tree under CWD into CWD.
Abort without moving anything if any filename collision would occur.
Uses parallel workers for the move and rmdir phases."""

import os
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

def scan(base):
    """Walk the tree once, collecting (source_path, leaf_name) pairs."""
    count, moves = {}, []
    for root, _, files in os.walk(base):
        root = os.path.normpath(root)
        for f in files:
            src = os.path.join(root, f)
            count[f] = count.get(f, 0) + 1
            moves.append((src, f))
    return count, moves

def check_collisions(count):
    """Exit with an error if any leaf name appears more than once."""
    collisions = [name for name, n in count.items() if n > 1]
    if collisions:
        for name in collisions:
            print(f"error: flatten would collide files with name: {name}", file=sys.stderr)
        sys.exit(1)

def move_file(src, dst):
    """Move a single file; returns a status string."""
    if src == dst:
        return None
    shutil.move(src, dst)
    return f"mv {src} -> {dst}"

def remove_dir(d):
    """Remove a single empty directory; returns a status string."""
    try:
        os.rmdir(d)
        return f"rmdir {d}"
    except OSError:
        return None

def main():
    base = os.getcwd()
    workers = os.cpu_count() or 4

    # --- scan & validate ---
    count, moves = scan(base)
    check_collisions(count)

    # --- parallel moves ---
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for src, leaf in moves:
            dst = os.path.join(base, leaf)
            futures.append(pool.submit(move_file, src, dst))
        for f in as_completed(futures):
            msg = f.result()
            if msg:
                print(msg)

    # --- collect empty dirs bottom-up, then remove in parallel ---
    dirs = []
    for root, subdirs, _ in os.walk(base, topdown=False):
        root = os.path.normpath(root)
        if root != base:
            dirs.append(root)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(remove_dir, d) for d in dirs]
        for f in as_completed(futures):
            msg = f.result()
            if msg:
                print(msg)

if __name__ == "__main__":
    main()


