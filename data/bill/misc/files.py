'''
This misc file focuses on working with file management.
'''

from pathlib import Path


def iter_binary_files(folder):
    
    folder = Path(folder)

    for p in folder.iterdir():
        if p.is_file() and p.suffix == ".bin":
            yield str(p)


