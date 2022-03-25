'''20220324: interpolate hyperspectral image to these frequencies:
band names = {20210719 60m: B1 443nm,
20210719 10m: B2 490nm,
20210719 10m: B3 560nm,
20210719 10m: B4 665nm,
20210719 20m: B5 705nm,
20210719 20m: B6 740nm,
20210719 20m: B7 783nm,
20210719 10m: B8 842nm,
20210719 20m: B8A 865nm,
20210719 60m: B9 945nm,
20210719 20m: B11 1610nm,
20210719 20m: B12 2190nm}'''
from misc import err, run, cd, args
if len(args) < 2:
    err('python3 raster_simulate_s2.py [input hyperspectral file name (ENVI format)]')
run([cd + 'raster_spectral_interp.exe',
     args[1],
     443, 
     490,
     560,
     665,
     705,
     740,
     783,
     842,
     865,
     945,
     1610,
     2190])
