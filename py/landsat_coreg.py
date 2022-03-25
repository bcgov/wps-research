from misc import *
X = ["./L8_vernon/LC08_L2SP_045025_20210803_20210811_02_T1/LC08_L2SP_045025_20210803_20210811_02_T1.bin_spectral_interp.bin_active.bin",
     "./L8_vernon/LC08_L2SP_045025_20210718_20210729_02_T1/LC08_L2SP_045025_20210718_20210729_02_T1.bin_spectral_interp.bin_active.bin",
     "./L8_vernon/LC08_L2SP_046025_20210725_20210803_02_T1/LC08_L2SP_046025_20210725_20210803_02_T1.bin_spectral_interp.bin_active.bin",
     "./L8_vernon/LC08_L2SP_046024_20210725_20210803_02_T1/LC08_L2SP_046024_20210725_20210803_02_T1.bin_spectral_interp.bin_active.bin",
     "./L8_vernon/LC08_L2SP_046024_20210810_20210819_02_T1/LC08_L2SP_046024_20210810_20210819_02_T1.bin_spectral_interp.bin_active.bin",
     "./L8_vernon/LC08_L2SP_046025_20210810_20210819_02_T1/LC08_L2SP_046025_20210810_20210819_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_045025_20210811_20210906_02_T1/LE07_L2SP_045025_20210811_20210906_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_046024_20210717_20210812_02_T1/LE07_L2SP_046024_20210717_20210812_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_046024_20210802_20210828_02_T1/LE07_L2SP_046024_20210802_20210828_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_045025_20210726_20210821_02_T1/LE07_L2SP_045025_20210726_20210821_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_046025_20210818_20210913_02_T1/LE07_L2SP_046025_20210818_20210913_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_046025_20210802_20210828_02_T1/LE07_L2SP_046025_20210802_20210828_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_046025_20210717_20210812_02_T1/LE07_L2SP_046025_20210717_20210812_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_046024_20210818_20210913_02_T1/LE07_L2SP_046024_20210818_20210913_02_T1.bin_spectral_interp.bin_active.bin"]

files = []
for x in X:
    cf = x + '_coreg.bin'
    #print(cf)
    files.append(cf)
    if not exists(cf):
        run(['python3 ' + pd + 'raster_project_onto.py',
              x,
              'footprint3.bin',
              cf])

Y = []
for f in files:
    w = f.split(sep)[-1].split('_')
    Y.append([w[3], f])

Y.sort()

cmd = 'cat'
for f in Y:
    cmd += (' ' + f[1])
cmd += ' > landsat.bin'

if not exists('landsat.bin'):
    run(cmd)

run('cp footprint3.hdr landsat.hdr')

cmd = 'python3 ' + pd + 'envi_header_modify.py'
