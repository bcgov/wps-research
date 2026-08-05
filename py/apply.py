#!/usr/bin/env python3
"""20260805: apply.py: help automate application of changes.
apply.py — drop a changes .zip into the repo and report what it touched.

Usage, from the wps-research repo root:

    python3 apply.py ~/Downloads/changes.zip

It moves the archive into the current directory, extracts it with
``unzip -o``, collects the extracted paths from unzip's own output, and
prints (but does NOT run):

    save <path> <path> ...

The paths come from unzip rather than from walking the tree afterwards,
so the list is exactly what the archive contained -- files that happened
to already exist in the repo are not swept in, and the relative paths
are reported verbatim.
"""

import os
import re
import shutil
import subprocess
import sys

# unzip -o emits lines like:
#   "  inflating: data/bill/.../persistence.py  "
#   "   creating: data/bill/.../handlers/"
#   " extracting: some/stored/file.bin"
_LINE_RE = re.compile(
    r'^\s*(?:inflating|extracting|creating|linking)\s*:\s*(.+?)\s*$')


def main():
    if len(sys.argv) < 2:
        print(f'usage: {os.path.basename(sys.argv[0])} <changes.zip>',
              file=sys.stderr)
        return 2

    src = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(src):
        print(f'ERROR: not a file: {src}', file=sys.stderr)
        return 1

    cwd = os.getcwd()
    dest = os.path.join(cwd, os.path.basename(src))

    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.move(src, dest)
        print(f'moved {src} -> {dest}', file=sys.stderr)
    else:
        print(f'{dest} is already here', file=sys.stderr)

    proc = subprocess.run(['unzip', '-o', dest],
                          capture_output=True, text=True, cwd=cwd)
    sys.stderr.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f'ERROR: unzip exited {proc.returncode}', file=sys.stderr)
        return proc.returncode

    source_files = []
    for line in proc.stdout.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        path = m.group(1)
        # Directory entries are structure, not changed files.
        if path.endswith('/'):
            continue
        # Archives built on macOS carry resource forks; they are not
        # part of the change set.
        base = os.path.basename(path)
        if base.startswith('._') or '__MACOSX' in path:
            continue
        if path not in source_files:
            source_files.append(path)

    if not source_files:
        print('ERROR: no files extracted -- nothing to report',
              file=sys.stderr)
        return 1

    print(' '.join(['save'] + source_files))
    return 0


if __name__ == '__main__':
    sys.exit(main())
