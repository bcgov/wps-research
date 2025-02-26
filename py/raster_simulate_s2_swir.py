'''20220908: modified from raster_simulate_s2.py

20220324: interpolate hyperspectral image to the frequencies at right:
    band names = {20210719 60m: B9 945nm,
                  20210719 20m: B11 1610nm,
                  20210719 20m: B12 2190nm}'''
from misc import err, run, cd, args
if len(args) < 2:
    err('python3 raster_simulate_s2.py [input hyperspectral raster (ENVI format)]')
fn = args[1]

run([cd + 'raster_spectral_interp.exe',
     fn,
     945,
     1610,
     2190])

hfn = fn[:-3] + 'hdr'
ohn = fn +  '_spectral_interp.hdr'
run(' '.join(['envi_header_copy_mapinfo.py',
              hfn,
              ohn]))
