'''20230628 group sentinel2 frames by date:
create a separate folder by date. '''
import os
import glob

def group_by_date(pattern):
    dirs_created = set()
    for line in glob.glob(pattern):
        d = line.split("_")[2].split('T')[0]
        if d not in dirs_created:
            os.makedirs(d, exist_ok=True)
            dirs_created.add(d)
        os.rename(line, os.path.join(d, line))

group_by_date("S2*")
