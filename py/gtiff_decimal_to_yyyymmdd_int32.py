'''20220125 gtiff_decimal_to_yyyymmdd.py: convert tiff with decimal year to yyyymmdd.
* No-data value=0. Leap years are taken into account'''
in_f = 'tbreak_1985_2024.tif' # input file
ou_f = 'tbreak_1985_2024_int32.tif'  # output file

import datetime
import numpy as np
from osgeo import gdal
ds = gdal.Open(in_f, gdal.GA_ReadOnly)  # open file
band = ds.GetRasterBand(1)  # band info
rb = band.ReadAsArray().ravel()  # read as linear array
src_proj, geo_xform, rows, cols = ds.GetProjection(),\
                                  ds.GetGeoTransform(),\
                                  band.YSize,\
                                  band.XSize
result = []
leap = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028, 2032]
for i in range(rb.shape[0]):
    try:
        d = rb[i]
        i_d = int(d)
        D = datetime.datetime(i_d, 1, 1) + datetime.timedelta(days=((d - i_d) * (366. if i_d in leap else 365.)))
        result += [str(D.year).zfill(4) + str(D.month).zfill(2) + str(D.day).zfill(2)]
    except:
        result += [0]

print(result)
dst_ds = gdal.GetDriverByName('GTiff').Create(ou_f,  # out file
                                              cols,  # cols
                                              rows,  # rows
                                              1,  # 1 band assumed
                                              gdal.GDT_UInt32)  # 32 bit int
dst_ds.SetGeoTransform(geo_xform)  # geolocation info
dst_ds.SetProjection(src_proj)  # geolocation info
dst_ds.GetRasterBand(1).WriteArray(np.array(result).reshape(rows, cols))  # write out
dst_ds.GetRasterBand(1).SetNoDataValue(0)  # redundant as we didn't write nan
dst_ds.FlushCache()  # possibly redundant but good form?
dst_ds = None  # possibly redundant but good form?
