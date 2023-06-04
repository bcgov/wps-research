'''use GDAL to resample a folder of raster (ENVI format) by a factor of X (default 2) in both dimensions

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
    mode selects the value which appears most often of all the sampled points.'''
import os
import sys
import argparse
from misc import err, run, args, exists, pd, sep, parfor
# message = "python3 raster_warp_all.py [input raster folder] [output raster folder] # [extra arg: use NN resampling instead of bilinear (default)]"

parser = argparse.ArgumentParser()
parser.add_argument("input_raster_folder", type=str, help="input raster folder")
parser.add_argument("output_raster_folder", type=str, help="output raster folder")
parser.add_argument("-n", "--nearest_neighbour", action="count", default=0, help="use nearest neighbour resampling instead of bilinear")
parser.add_argument("-s", "--scaling_factor", type=float, default=2, help="integer scaling factor")
args = parser.parse_args()
# print(args)
use_bilinear = args.nearest_neighbour == 0
in_dir, out_dir = os.path.abspath(args.input_raster_folder), os.path.abspath(args.output_raster_folder)

if not exists(in_dir):
    err("please check input dir")
if not exists(out_dir):
    err("please check output dir")

files = [x.strip() for x in os.popen("ls -1 " + in_dir + sep + "*.bin").readlines()]

def processing(c):
    of = out_dir + sep + f.split(sep)[-1]
    oh = of[:-3] + 'hdr'

    if exists(of):
        print("Warning: file exists (skipping):", of)
        continue
    # -r {nearest (default),bilinear,cubic,cubicspline,lanczos,average,rms,mode}
    s = 100. / args.scaling_factor
    cmd = ' '.join(['gdal_translate',
                    '-of ENVI',
                    '-ot Float32',
                    '-outsize ' + str(s) + '% ' + str(s) +'%',
                    '-r bilinear' if use_bilinear else '-r nearest',
                    f,
                    of])
    run(cmd)
    run('envi_header_cleanup.py ' + oh)
    run('rm ' + oh + '.bak')
    run('rm ' + of + '.aux.xml')

parfor(processing, files, 16)
