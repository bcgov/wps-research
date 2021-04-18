# this script is not finished
'''

cd path/to/shapefiles
ogr2ogr -sql "SELECT ST_Centroid(geometry), * FROM countries" -dialect sqlite countries_centroid.shp countries.shp

'''

'''
# Not tested. from:
# https://gis.stackexchange.com/questions/216788/convert-polygon-feature-centroid-to-points-using-python

import os
from osgeo import ogr

ogr.UseExceptions()
os.chdir('path/to/shapefiles')

ds = ogr.Open('countries.shp')
ly = ds.ExecuteSQL('SELECT ST_Centroid(geometry), * FROM countries', dialect='sqlite')
drv = ogr.GetDriverByName('Esri shapefile')
ds2 = drv.CreateDataSource('countries_centroid.shp')
ds2.CopyLayer(ly, '')
ly = ds = ds2 = None  # save, close
''
