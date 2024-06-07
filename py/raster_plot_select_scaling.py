'''20240607 raster_plot_select_scaling

Select scaling factors for one date.

Apply these to the other dates to restore and plot the sequence e.g. with sentinel2_plot.py 

e.g. take the scaling factors like:
S2A_MSIL2A_20230523T151651_N0509_R025_T19TGJ_20230523T214357_cloudfree.bin_MRAP.bin_rgb_scaling_0.txt
S2A_MSIL2A_20230523T151651_N0509_R025_T19TGJ_20230523T214357_cloudfree.bin_MRAP.bin_rgb_scaling_1.txt
S2A_MSIL2A_20230523T151651_N0509_R025_T19TGJ_20230523T214357_cloudfree.bin_MRAP.bin_rgb_scaling_2.txt

and replace all other scaling_0.txt with them!'''

import os
import sys
from misc import args, run, err

if len(args) != 2:
    err('raster_plot_select_scaling [image date envi format .bin file]')


files = [args[1] + '_rgb_scaling_' + str(i) + '.txt' for i in [0,1,2]]

for f in files:
    print(f)    

for i in [0, 1, 2]:
    lines = [x.strip() for x in os.popen('ls -1 *_rgb_scaling_' + str(i) + '.txt').readlines()]
    for line in lines:
        if os.path.abspath(line) != files[i]:
            run('cp -v ' + files[i] + ' ' + line)

