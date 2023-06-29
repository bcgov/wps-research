'''20230629 assuming sentinel2 frames already grouped by date,
create a mosaic for each date and project?'''
from misc import run, err, exists
import os

lines = [x.strip() for x in os.popen("ls -1").readlines()]

f = open("sentinel2_merge_by_date.sh", "wb")
for line in lines:
    if len(line) == 8:
        try:
            N = int(line)
        except:
            continue


    f.write(("cd " + line + "; merge.py; cd ..\n").encode())
f.close()

print("+w sentinel2_merge_by_date.sh")
