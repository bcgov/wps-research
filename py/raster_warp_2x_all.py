'''use GDAL to resample a folder of raster (ENVI format) 
by a factor of two in both dimensions

Note: from GDAL documentation:
-r {nearest (default),bilinear,cubic,cubicspline,lanczos,average,rms,mode}
    Select a resampling algorithm.
    nearest applies a nearest neighbour (simple sampling) resampler
    average computes the average of all non-NODATA contributing pixels. Starting with GDAL 3.1, this is a weighted average taking into account properly the weight of source pixels not contributing fully to the target pixel.
    rms computes the root mean squared / quadratic mean of all non-NODATA contributing pixels (GDAL >= 3.3)
    bilinear applies a bilinear convolution kernel.
    cubic applies a cubic convolution kernel.
    cubicspline applies a B-Spline convolution kernel.
    lanczos applies a Lanczos windowed sinc convolution kernel.
    mode selects the value which appears most often of all the sampled points.

'''
import os
import sys
from misc import err, run, args, exists, pd, sep

message = "python3 raster_warp_2x_all.py [input raster folder] [output raster folder] # [extra arg: use NN resampling instead of bilinear (default)]"
if len(args) < 3:
    err(message)

use_bilinear = len(args) < 4

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
                    '-r bilinear' if use_bilinear else '-r nearest',
                    f,
                    of])
    run(cmd)
    run('envi_header_cleanup.py ' + oh)
    run('rm ' + oh + '.bak')
    run('rm ' + of + '.aux.xml')
