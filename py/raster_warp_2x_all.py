'''use GDAL to resample a folder of raster (ENVI format) 
by a factor of two in both dimensions
'''
import os
import sys
from misc import err, run, args, exists, pd, sep

message = "python3 raster_warp_2x_all.py [input raster folder] [output raster folder] # [extra arg: use NN resampling instead of bilinear (default)]"
if len(args) < 3:
    err(message)

use_bilinear = len(args) < 4
print("use_bilinear", use_bilinear)

in_dir, out_dir = os.path.abspath(args[1]), os.path.abspath(args[2])

if not exists(in_dir):
    err("please check input dir")
if not exists(out_dir):
    err("please check output dir")

files = [x.strip() for x in os.popen("ls -1 " + in_dir + sep + "*.bin").readlines()]

for f in files:
    print(f)
    of = out_dir + sep + f.split(sep)[-1]
    oh = of[:-3] + 'hdr'
    print(oh)
    print("  " + of)
    # -r {nearest (default),bilinear,cubic,cubicspline,lanczos,average,rms,mode}
    cmd = ' '.join(["gdal_translate -of ENVI -ot Float32 -outsize 50% 50%",
                    f,
                    of])
    run(cmd)
    run('envi_header_cleanup.py ' + oh)
