'''20220808 from ENVI type-4 file, create a good geo-Tiff by histogram trimming, converting to byte, then converting to TIF

Input assumed to have geo-location in .hdr file'''
import os
from misc import args, err, run

if len(args) < 2:
    err('python3 envi2tif.py [input .bin file name]')

sep = os.path.sep
pd = sep.join(__file__.split(sep)[:-1]) + sep

fn = args[1]

run('htrim2.exe ' + fn + ' 2. 2.')

# copy map info
run('python3 ' + pd + 'envi_header_copy_mapinfo.py '  + fn[:-3] + 'hdr ' + fn + '_ht.hdr')
run('python3 ' + pd + 'envi_update_band_names.py '    + fn[:-3] + 'hdr ' + fn + '_ht.hdr')

# convert to TIF (byte format)
run('gdal_translate -of GTiff -ot Byte ' +
    (fn + '_ht.bin') +
    ' ' +  
    (fn + '_ht.tif'))
