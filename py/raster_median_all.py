# 20260525 calculate median/mediod for all .bin files in present folder. Exclude tmp* and median.bin files
import os
import sys
import glob

# Get all .bin files excluding median.bin and tmp*
lines = [
    f for f in glob.glob("*.bin")
    if f != "median.bin" and not f.startswith("tmp")
]

cmd = 'raster_median ' + ' '.join(lines) + ' median.bin'
print(cmd)

a = os.system(cmd)
