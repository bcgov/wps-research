'''20220831
running https://github.com/bopen/elevation automatically

Deps:
    pip3 install elevation
    pip3 install rasterio    # for raster reference/footprint file (here we assume raster)
    pip3 install fiona       # for shapefile reference/footprint file

Steps:
    1. raster_pixels_loc.py # determine lat/lon for each pixel in image
    2. raster_quickstats    # determine min,max for both lat and lon
    3. # {increase, decrease} the {lat/lon}  {max, min} respectively, to make sure we get results for the data

'''
from misc import run, err, args

fn = args[1]
out_fn = fn + '_dem.tif'

run('eio --product SRTM3 clip -o ' + out_fn + ' --reference ' + fn)