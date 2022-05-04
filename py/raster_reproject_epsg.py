from misc import args, err, run

if len(args) < 3:
    err('python3 raster_reproject_epsg.py [input raster file] ' +
        '[output EPSG number to warp to. eg 4326]')

fn, epsg = args[1: 3]

run(['gdalwarp',
    '-t_srs EPSG:' + str(epsg),  # EPSG number of CRS to warp to
    '-r bilinear',  # interpolation method
    '-of ENVI',  # envi format output
    '-ot Float32',  # 32-bit float
    fn,  # input raster file name, to be warped!
    fn + '_EPSG' + str(epsg) + '.bin'])  # output raster filename


