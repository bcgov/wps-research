'''list active fires listed by BCWS over 100 ha '''

if __name__ == '__main__':
    fn = 'prot_current_fire_points.zip'  # download fire data
    dl_path = 'https://pub.data.gov.bc.ca/datasets/2790e3f7-6395-4230-8545-04efb5a18800/' + fn
    urllib.request.urlretrieve(dl_path, fn)

    t = datetime.datetime.now().strftime("%Y%m%d")  # %H%M%S")  # timestamped backup
    shutil.copyfile(fn, 'prot_current_fire_points_' + t + '.zip')
    zipfile.ZipFile(fn).extractall()

    # Open Shapefile
    Shapefile = ogr.Open('prot_current_fire_points.shp')
    print(Shapefile)
    layer = Shapefile.GetLayer()
    layerDefinition = layer.GetLayerDefn()
    feature_count = layer.GetFeatureCount()
    spatialRef = layer.GetSpatialRef()

    def records(layer):
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            yield json.loads(feature.ExportToJson())

    features = records(layer)
    feature_names, feature_ids, records_use = [], [], []
    for f in features:
        for key in f.keys():
            if key == 'properties':
                fk = f[key]
                fire_size = float(fk['CURRENT_SI'])
                if fk['FIRE_STATU'].lower() != 'out' and fire_size > 100.:  # fire_size > biggest_size
                    records_use.append(fk)  # selected fires
                    print(fk)

