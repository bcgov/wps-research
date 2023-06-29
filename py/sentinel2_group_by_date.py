'''20230628 group sentinel2 frames by date:
create a separate folder by date. '''

from misc import run, err, exists
import os

lines = [x.strip() for x in os.popen("ls -1 S2*").readlines()]

for line in lines:
    w = line.split("_")
    d = w[2].split('T')[0]
    
    if not exists(d):
        os.mkdir(d)
   
    run('mv -v ' + line + ' ' + d)




