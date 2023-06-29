'''20230628 group sentinel2 frames by date:
create a separate folder by date. '''

from misc import run, err, exists
import os

lines = [x.strip() for x in os.popen("ls -1").readlines()]

for line in lines:
    if len(line) == 8:
        try:
            N = int(line)
        except:
            continue

    files = [x.strip() for x in os.popen("ls -1 " + line + "/S2*").readlines()]

    for f in files:
        run('mv -v ' + f + ' .')
        



