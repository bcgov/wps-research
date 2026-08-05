#!/usr/bin/env python3
"""20260805: apply.py: help automate application of changes
apply.py — drop a changes .zip into the repo and report what it touched.

Usage, from the wps-research repo root:

    python3 apply.py ~/Downloads/changes.zip

It refuses to run unless the current directory is named
``wps-research``, moves the archive here, extracts it with ``unzip -o``,
collects the extracted paths from unzip's own output, then prints and
runs:

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

    # Guard against running from the wrong directory: the archive's
    # paths are relative to the repo root, so extracting elsewhere would
    # scatter files into an unrelated tree.
    cwd_name = os.path.basename(os.getcwd().rstrip(os.sep))
    if cwd_name != 'wps-research':
        print(f'ERROR: must be run from the wps-research repo root '
              f'(current directory is {cwd_name!r})', file=sys.stderr)
        return 1

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

    cmd = ' '.join(['save'] + source_files)
    print(cmd)
    a = os.system(cmd)
    if a != 0:
        print(f'WARNING: save exited with status {a}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
