#!/usr/bin/env python3
"""Zip every .py file under py/ into py.zip, keeping the folder structure.

Run from the wps-research directory:

    cd wps-research
    python3 py/create_zip_of_py_files.py

The script refuses to run unless the current folder is named
wps-research. It searches py/ at full depth and writes wps-research/py.zip,
with paths inside the archive starting py/ (e.g. py/sub/tool.py), so it
unzips back into place inside wps-research/.
"""
import os
import sys
import zipfile

REPO_NAME = 'wps-research'
SRC_DIR = 'py'
OUT_NAME = 'py.zip'


def main():
    cwd = os.getcwd()
    if os.path.basename(os.path.normpath(cwd)) != REPO_NAME:
        sys.exit(f'Error: run this from the {REPO_NAME} folder '
                 f'(current folder: {cwd}).\n'
                 f'  cd {REPO_NAME}\n'
                 f'  python3 {SRC_DIR}/create_zip_of_py_files.py')

    src = os.path.join(cwd, SRC_DIR)
    if not os.path.isdir(src):
        sys.exit(f'Error: no {SRC_DIR}/ folder in {cwd}')

    # Full-depth search. Symlinked folders are not followed, so a link
    # pointing back up the tree cannot make the walk loop.
    found = []
    for root, dirs, files in os.walk(src):
        dirs.sort()
        for name in sorted(files):
            if name.endswith('.py'):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    found.append(path)

    if not found:
        sys.exit(f'No .py files found under {SRC_DIR}/; nothing written.')

    out = os.path.join(cwd, OUT_NAME)
    tmp = out + '.tmp'
    total = 0
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            for path in found:
                # Archive path relative to wps-research: py/...
                arc = os.path.relpath(path, cwd).replace(os.sep, '/')
                z.write(path, arc)
                total += os.path.getsize(path)
        # Replace any previous py.zip only once the new one is complete.
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    print(f'Wrote {out}: {len(found)} .py file(s), '
          f'{total / 1024:.1f} KB before compression, '
          f'{os.path.getsize(out) / 1024:.1f} KB zipped.')


if __name__ == '__main__':
    main()
