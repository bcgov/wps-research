'''20230316 extract polygon coordinates for shapefile fields. Assumed application: the sentinel-2 tile outlines at py/sentinel2_bc_tiles_shp/

Write each feature to a separate "footprint" file (.fpf) in it's own folder.

From ESA website: 
    footprint: 	Geographical search of the products whose footprint intersects or is included in a specific geographic type.
    Syntax:

    footprint:"intersects(<geographic type>)"
    The <geographic type> value can be expressed as a polygon or as a point, according to the syntax described below:

    GEOGRAPHIC TYPE:
    
    <geographic type>=POLYGON((P1Lon P1Lat, P2Lon P2Lat, …, PnLon PnLat, P1Lon P1Lat))
    where P1Lon and P1Lat are the Longitude and Latitude coordinates of the first point of the polygon in decimal degrees (DDD) format (e.g. 2.17403, 41.40338) and so on.
    The coordinates of the last point of the polygon must coincide with the coordinates of the first point of the polygon.
    There is not a fixed limit for the maximum number of points to be used to define the Area of Interest (AOI). The limit is related to the maximum number of characters for the GUI search (i.e 8000). The maximum footprint extent possible is defined by the Mercator projection used on DHuS, i.e. LAT [-180, + 180] and LONG [-85.05, +85.05].

    Examples:
        The polygon of the example is a bounding box around the Mediterranean Sea:
            footprint:"Intersects(POLYGON((-4.53 29.85, 26.75 29.85, 26.75 46.80,-4.53 46.80,-4.53 29.85)))"

    POINT:
    <geographic type>=Lat, Lon
    where the Latitude (Lat) and Longitude (Lon) values are expressed in decimal degrees (DDD) format (e.g. 41.40338, 2.17403 ).
'''
import os
import sys
import json
from osgeo import ogr
from osgeo import gdal
from osgeo import gdalconst
args = sys.argv

abspath = os.path.abspath
sep = os.path.sep
def get_pd():
    return abspath(sep.join(abspath(__file__).split(sep)[:-1])) + sep  # python directory i.e. path to here

InputVector = get_pd() + "Sentinel_BC_Tiles.shp"
print_all = False

def err(m):
    print("Error:", m); sys.exit(1)

if len(args) < 2:
    pass # err("Error: shapefile_info.py [input shapefile .shp] [optional parameter: full]")
else:
    InputVector = args[1] # shapefile to rasterize

# Open Shapefile
Shapefile = ogr.Open(InputVector)
layer = Shapefile.GetLayer()
layerDefinition = layer.GetLayerDefn()
feature_count = layer.GetFeatureCount()
spatialRef = layer.GetSpatialRef()
print("CRS:", spatialRef)

def records(layer):
    # generator
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        yield json.loads(feature.ExportToJson())

print("feature_count," + str(feature_count))
print("feature","type", "x","y")
features = records(layer)
feature_names, feature_ids = [], []
for f in features:  
    geom = "" 
    if print_all:
        print(f.keys())
    for key in f.keys():
        if key=='properties':
            if print_all:
                print(f[key])
    try:
        geom = f['geometry']
    except Exception:
        pass

    feature_id = f['id']
    feature_ids.append(feature_id) # print(f['properties'].keys())
    feature_name = ''
    try:
        feature_name = f['properties']['Name']
        print("Name", feature_name)
    except Exception:
        pass # feature name not available
    feature_names.append(feature_name)

    if geom['type'] == 'Point':
        stuff = [feature_id,
                 geom['type'],
                 geom['coordinates'][0],
                 geom['coordinates'][1]]
        print(','.join([str(x) for x in stuff]))
    else:
        #print(geom['type'])
        for c in geom['coordinates']:
            coords = list(c)
            #print(coords)

            s = 'Intersects(POLYGON(('
            for i in range(len(coords)):
                if i > 0:
                    s += ','
                s += (str(coords[i][0]) + ' ' + str(coords[i][1]))
            s += ')))'

            print(feature_name, s)
            if not os.path.exists(feature_name):
                os.mkdir(feature_name)
            open(feature_name + sep + 'fpf', 'wb').write(s.encode())
