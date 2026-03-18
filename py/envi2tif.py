'''20220808 from ENVI type-4 file, create a good geo-Tiff by histogram trimming, converting to byte, then converting to TIF
Input assumed to have geo-location in .hdr file

Optional args: three 1-based band indices to select for the output TIF.
  python3 envi2tif.py [input .bin file] [band1] [band2] [band3]
If provided, band extraction is performed first on the raw input so that
htrim2, raster_smult, etc. operate only on the selected bands.
If omitted, all bands are included (original behaviour).'''
import os
from misc import args, err, run
if len(args) < 2:
    err('python3 envi2tif.py [input .bin file name] [optional: band1 band2 band3]')
sep = os.path.sep
pd = sep.join(__file__.split(sep)[:-1]) + sep
fn = args[1]

# Optional band selection: three 1-based indices
band_select = None
if len(args) == 5:
    try:
        band_select = [int(args[2]), int(args[3]), int(args[4])]
    except ValueError:
        err('band indices must be integers, got: ' + ' '.join(args[2:5]))

# If band selection requested, extract those bands from the raw .bin first.
# gdal_translate writes an ENVI .bin + .hdr pair that the rest of the pipeline
# can process identically to the original full-band file.
# The extracted file is removed at the end.
if band_select:
    fn_sel = fn + '_sel.bin'
    b_flags = ' '.join('-b ' + str(b) for b in band_select)
    run('gdal_translate -of ENVI ' + b_flags + ' ' + fn + ' ' + fn_sel)
    work_fn = fn_sel
else:
    work_fn = fn

run('htrim2.exe ' + work_fn + ' 1. 1.')
# multiply by 255.
run('raster_smult.exe ' +
    work_fn + '_ht.bin ' +
    '255.')
# copy map info
run('python3 ' + pd + 'envi_header_copy_mapinfo.py ' + fn[:-3] + 'hdr ' + work_fn + '_ht.bin_smult.hdr')
run('python3 ' + pd + 'envi_update_band_names.py '   + fn[:-3] + 'hdr ' + work_fn + '_ht.bin_smult.hdr')
# convert to TIF (byte format)
run('gdal_translate -of GTiff -ot Byte ' +
    (work_fn + '_ht.bin_smult.bin') +
    ' ' +
    (fn + '_ht.bin_smult.tif'))

# clean up band-selection intermediate files
if band_select:
    run('rm -f ' + fn_sel + ' ' + fn_sel[:-3] + 'hdr')
    run('rm -f ' + fn_sel + '_ht.bin ' + fn_sel + '_ht.hdr')
    run('rm -f ' + fn_sel + '_ht.bin_smult.bin ' + fn_sel + '_ht.bin_smult.hdr')
