'''20240527 plot sentinel2 products (filenames beginning with frame ID) 

20241220: added support for *mrap.bin files
'''
from misc import err, run, parfor, hdr_fn, read_hdr
import os

lines = [x.strip() for x in os.popen('ls -1 S*.bin *mrap*bin').readlines()]
# add BARC files
print(lines)

# process in order
#lines = [[line.split('_')[2], line] for line in lines]
lines.sort()
# lines = [line[1] for line in lines]


print(lines)
cmds = []

for line in lines:
    samples, lin, bands = read_hdr(hdr_fn(line))

    out_file = 'plot_1_' + line + '_1_2_3_rgb.png'

    if not os.path.exists(out_file):
        if int(bands) < 3:
            cmds += ["raster_plot.py " + line + " 1 1 1 1 " for line in lines]
        else:
            cmds += ["raster_plot.py " + line + " 1 2 3 1 " for line in lines]
    # plot_1_20230902_mrap.bin_1_2_3_rgb.png

def r(x):
    return os.system(x)

parfor(r, cmds, 4)

'''Now: prefix the S2.png files by date:
'''
lines = os.popen("ls -1 S2*.png *mrap*png *BARC*png").readlines()
lines = [x.strip() for x in lines]

for line in lines:
    T = line.split('_')[2].split('T')[0]
    run('mv -v ' + line + ' ' + 'plot_' + T + '_' + line)
