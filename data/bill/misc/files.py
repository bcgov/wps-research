'''
This misc file focuses on working with file management.
'''

from pathlib import Path


def iter_files(
        folder_name,
        file_type
    ):
    '''
    Description
    -----------
    Retrieve all file names of format ... from a folder.

    
    Parameters
    ----------
    folder_name: folder name to iterate through

    file_type: e.g .bin, .hdr, .py...


    Returns
    -------
    An iterator of file names
    '''
    
    folder = Path(folder_name)

    for p in folder.iterdir():
        if p.is_file() and p.suffix == file_type:
            yield str(p)



