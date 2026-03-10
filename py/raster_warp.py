'''use GDAL to resample a single raster (ENVI format) by a factor of X (default 2) in both dimensions
Note: from GDAL documentation:
-r {nearest (default),bilinear,cubic,cubicspline,lanczos,average,rms,mode}
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
import argparse
from misc import err, run, exists, sep

parser = argparse.ArgumentParser()
parser.add_argument("input_raster", type=str, help="input raster file")
parser.add_argument("output_raster", type=str, help="output raster file")
parser.add_argument("-n", "--nearest_neighbour", action="count", default=0, help="use nearest neighbour resampling instead of bilinear")
parser.add_argument("-s", "--scaling_factor", type=float, default=2, help="integer scaling factor")
args = parser.parse_args()

use_bilinear = args.nearest_neighbour == 0
f, of = os.path.abspath(args.input_raster), os.path.abspath(args.output_raster)

if not exists(f):
    err("please check input file")

oh = of[:-3] + 'hdr'
if exists(of):
    print("Warning: file exists (skipping):", of)
else:
    s = 100. / args.scaling_factor
    cmd = ' '.join(['gdal_translate',
                    '-of ENVI',
                    '-ot Float32',
                    '-outsize ' + str(s) + '% ' + str(s) + '%',
                    '-r bilinear' if use_bilinear else '-r nearest',
                    f,
                    of])
    a = run(cmd, False)
    if a == 0:
        run('envi_header_cleanup.py ' + oh)
        run('rm ' + oh + '.bak')
        run('rm ' + of + '.aux.xml')

