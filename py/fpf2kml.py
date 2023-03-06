'''20220824 convert "fpf" format (copernicus footprint string file) to kml. Example file "fpf":

Intersects(POLYGON((-124.79178589364845 50.836204617080405,-124.79178589364845 50.914095382919584,-124.66834810635157 50.914095382919584,-124.66834810635157 50.836204617080405,-124.79178589364845 50.836204617080405)))
'''
import os
import sys
from osgeo import ogr
args = sys.argv
fn = args[1] if len(args) > 1 else 'fpf'
ofn = fn + '.kml'
wkt = open(fn).read().strip()[11:-1]

# wkt = 'POLYGON((-124.79178589364845 50.836204617080405,-124.79178589364845 50.914095382919584,-124.66834810635157 50.914095382919584,-124.66834810635157 50.836204617080405,-124.79178589364845 50.836204617080405))'
poly = ogr.CreateGeometryFromWkt(wkt)
kml = poly.ExportToKML()

data = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Placemark>
    <name>polygon</name>
    <description>polygon</description>'''

data2 = '''</Placemark>
</kml>'''

data = data + kml + data2 # print(data)

print('+w', ofn)
open(ofn, 'wb').write(data.encode())
