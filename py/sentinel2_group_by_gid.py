'''20240531 group sentinel2 frames by gid "tile id"
create a separate folder for each one'''

from misc import run, err, exists
import os

lines = [x.strip() for x in os.popen("ls -1 S2*MSIL2A*.").readlines()]
for line in lines:
    d = "L2_" + line.split("_")[4]
    if not exists(d):
        os.mkdir(d)
    run('mv -v ' + line + ' ' + d)


lines = [x.strip() for x in os.popen("ls -1 S2*MSIL1C*.").readlines()]
for line in lines:
    d = "L1_" + line.split("_")[4]
    if not exists(d):
        os.mkdir(d)
    run('mv -v ' + line + ' ' + d)
