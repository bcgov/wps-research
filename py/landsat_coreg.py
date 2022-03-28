from misc import *
X = ["./L8_vernon/LC08_L2SP_045025_20210803_20210811_02_T1/LC08_L2SP_045025_20210803_20210811_02_T1.bin_spectral_interp.bin_active.bin",
     "./L8_vernon/LC08_L2SP_045025_20210718_20210729_02_T1/LC08_L2SP_045025_20210718_20210729_02_T1.bin_spectral_interp.bin_active.bin",
     "./L8_vernon/LC08_L2SP_046025_20210725_20210803_02_T1/LC08_L2SP_046025_20210725_20210803_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L8_vernon/LC08_L2SP_046024_20210725_20210803_02_T1/LC08_L2SP_046024_20210725_20210803_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L8_vernon/LC08_L2SP_046024_20210810_20210819_02_T1/LC08_L2SP_046024_20210810_20210819_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L8_vernon/LC08_L2SP_046025_20210810_20210819_02_T1/LC08_L2SP_046025_20210810_20210819_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_045025_20210811_20210906_02_T1/LE07_L2SP_045025_20210811_20210906_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L7_vernon/LE07_L2SP_046024_20210717_20210812_02_T1/LE07_L2SP_046024_20210717_20210812_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L7_vernon/LE07_L2SP_046024_20210802_20210828_02_T1/LE07_L2SP_046024_20210802_20210828_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_045025_20210726_20210821_02_T1/LE07_L2SP_045025_20210726_20210821_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L7_vernon/LE07_L2SP_046025_20210818_20210913_02_T1/LE07_L2SP_046025_20210818_20210913_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L7_vernon/LE07_L2SP_046025_20210802_20210828_02_T1/LE07_L2SP_046025_20210802_20210828_02_T1.bin_spectral_interp.bin_active.bin",
     "./L7_vernon/LE07_L2SP_046025_20210717_20210812_02_T1/LE07_L2SP_046025_20210717_20210812_02_T1.bin_spectral_interp.bin_active.bin",
     # "./L7_vernon/LE07_L2SP_046024_20210818_20210913_02_T1/LE07_L2SP_046024_20210818_20210913_02_T1.bin_spectral_interp.bin_active.bin"]
     ]

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


# add Landsat
Y = []
for f in files:
    w = f.split(sep)[-1].split('_')
    Y.append([w[3], f])

# add in Sentinel2

for f in ['T10UGA', 'T10UGB', 'T11ULR', 'T11ULS']:
    files = os.popen('find ' + f + ' -name "*active.bin"').readlines()
    for z in files:
        x = z.strip()
        cf = x + '_coreg.bin'
        if not exists(cf):
            run(['python3 ' + pd + 'raster_project_onto.py',
                 x,
                 'footprint3.bin',
                 cf])
        x = x.strip().split(sep)[1].split('_')[2][0: 8]
        print(x)
        Y.append([x, cf])

Y.sort()
for x in Y:
    print(x)

cmd = 'cat'
for f in Y:
    cmd += (' ' + f[1])
cmd += ' > landsat.bin'
if not exists('landsat.bin'):
    run(cmd)

if not exists('landsat.hdr'):
    run('cp footprint3.hdr landsat.hdr')
    samples, lines, bands = read_hdr('landsat.hdr')
    cmd = 'python3 ' + pd + 'envi_header_modify.py landsat.hdr ' + str(lines) + ' ' + str(samples) + ' ' + str(len(Y))
    for f in Y:
        cmd += ' ' + '"' + f[1] + '"'
    run(cmd)
    print("number of bands", len(Y))
# for x in Y: run('imv ' + x[1])

