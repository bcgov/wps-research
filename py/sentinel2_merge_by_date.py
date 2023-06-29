'''20230629 assuming sentinel2 frames already grouped by date,
create a mosaic for each date and project?'''
from misc import run, err, exists
import os

master_date = "20230607"

lines = [x.strip() for x in os.popen("ls -1").readlines()]

f = open("sentinel2_merge_by_date.sh", "wb")
for line in lines:
    if len(line) == 8:
        try:
            N = int(line)
            if not exists(line + "/merge.bin"):
                f.write(("cd " + line + "; merge.py; cd ..\n").encode())
        except:
            pass
f.close()

print("+w sentinel2_merge_by_date.sh")



sc = "raster_stack.py"

for line in lines:
    if len(line) == 8:
        if not exists(line + ".bin"):
            cmd = "po " + line + "/merge.bin " + master_date + "/merge.bin " + line + ".bin"
            run(cmd)

        
        sc += (" " + line + ".bin")

sc += " stack.bin"
print(sc)
