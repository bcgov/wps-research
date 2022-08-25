'''20220824 run this from a dated active-fire mapping folder, that doesn't have an existing subarea defined for detection
'''
from misc import err, run, args

if len(args) < 2:
    err('raster_fpf.py [input raster name to extract from] # rasterize fpf and pad before extracting')
fn = args[1]

run('fpf2kml.py ../fpf')
run('kml2shp ../fpf.kml')
run('shapefile_rasterize_onto.py ../fpf.kml.shp ' + fn + ' foot.bin')
run('crop foot_0000__polygon.bin')
run('pad foot_0000__polygon.bin_crop.bin ')
run('po ' + fn + 'foot_0000__polygon.bin_crop.bin_pad.bin sub.bin')
