'''
run https://github.com/bopen/elevation

Deps:
    pip3 install elevation
    pip3 install rasterio
    pip3 install fiona

'''
from misc import run, err, args

fn = args[1]
out_fn = fn + '_dem.tif'

run('eio --product SRTM3 clip -o ' + out_fn + ' --reference ' + fn)
