'''20240531 group sentinel2 frames by gid "tile id"
create a separate folder for each one'''
import os
import glob

def group_by_gid(pattern, prefix):
    dirs_created = set()
    for line in glob.glob(pattern):
        d = prefix + line.split("_")[5]
        if d not in dirs_created:
            os.makedirs(d, exist_ok=True)
            dirs_created.add(d)
        os.rename(line, os.path.join(d, line))

group_by_gid("S2*MSIL2A*.*", "L2_")
group_by_gid("S2*MSIL1C*.*", "L1_")
