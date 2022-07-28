'''list active fires listed by BCWS over 100 ha '''
import math
import json
import shutil
import zipfile
import datetime
import webbrowser
import urllib.request
from osgeo import ogr
from bounding_box import bounding_box

MIN_FIRE_SIZE_HA = 100.
TOP_N = 5
selected = []

if __name__ == '__main__':
    fn = 'prot_current_fire_points.zip'  # download fire data
    dl_path = 'https://pub.data.gov.bc.ca/datasets/2790e3f7-6395-4230-8545-04efb5a18800/' + fn
    urllib.request.urlretrieve(dl_path, fn)

    t = datetime.datetime.now().strftime("%Y%m%d")  # %H%M%S")  # timestamped backup
    shutil.copyfile(fn, 'prot_current_fire_points_' + t + '.zip')
    zipfile.ZipFile(fn).extractall()

    # Open Shapefile
    Shapefile = ogr.Open('prot_current_fire_points.shp') # print(Shapefile)
    layer = Shapefile.GetLayer()
    layerDefinition = layer.GetLayerDefn()
    feature_count = layer.GetFeatureCount()
    spatialRef = layer.GetSpatialRef()

    def records(layer):
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            yield json.loads(feature.ExportToJson())

    features = records(layer)
    feature_names, feature_ids = [], []
    for f in features:
        for key in f.keys():
            if key == 'properties':
                fk = f[key]
                fire_size = float(fk['CURRENT_SI'])
                if fk['FIRE_STATU'].lower() != 'out' and fire_size >= MIN_FIRE_SIZE_HA:  # fire_size > biggest_size
                    selected.append([fk['CURRENT_SI'], fk])  # selected fires
                    #print(fk)

# sort the fires by curent size, largest first
ix = [[selected[i][0], i] for i in range(len(selected))]
ix.sort(reverse=True)
selected = [selected[i[1]] for i in ix]

browser = webbrowser.get('google-chrome')

# consider the top N
ci = 0
for s in selected:
    r = s[1] 
    lat, lon, size_ha = r['LATITUDE'], r['LONGITUDE'], r['CURRENT_SI']
    print(r['GEOGRAPHIC'])
    print('\t', type(s[0]), ci, s)

    view_str = ('https://apps.sentinel-hub.com/sentinel-playground/?source=S2L2A&lat=' +
                str(lat) + '&lng=' +
                str(lon) +
                '&zoom=11&preset=CUSTOM&layers=B12,B11,B8A&maxcc=100'
                + '&evalscript=cmV0dXJuIFtCMTIqMi41LEIxMSoyLjUsQjhBKjIuNV0%3D')

    print(view_str)
    # browser.open_new_tab(view_str)

    # A hectare is equal to 10,000 square meters
    # def bounding_box(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    sq_m = size_ha * 10000.  # square metres area!
    sq_len = math.sqrt(sq_m) # assuming (wrongly) the fire is square, the length/width of the fire
    print("length of square", sq_len)
    sq_len_km = sq_len / 1000. # length of square in km

    print(bounding_box(lat, lon, sq_len_km))


    ci += 1
    if ci >= TOP_N:
        break
