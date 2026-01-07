'''20250106 run median filter using a 10 day window over a sentinel-2 time series
'''
import os
import sys
 
def run(c):
    print(c)
    return os.system(c)
 
lines, lines2 = [x.strip() for x in os.popen("ls -1 S2*.bin").readlines()], []
for x in lines:
    w = x.split('_')
    lines2 += [[w[2], x]]
lines2.sort()
 
for i in range(0, len(lines2)):
    if ( i + 10 >= len(lines2)):
        break
 
    c = "raster_median "
    for j in range(10):
        c += (lines2[i + j][1] + " ")
 
    c += "median_" + lines2[i + j][0] + ".bin"
    run(c)
    c = "raster_plot.py median_" + lines2[i + j][0] + ".bin 1 2 3 1 &"
    run(c)
