'''20230605 sentinel2_extract_swir.py'''
from envi import envi_header_cleanup
from osgeo import gdal
import sys
d = gdal.Open(sys.argv[1])
subdatasets =  d.GetSubDatasets()
'''
for subdataset in subdatasets:
    subdataset_path = subdataset[0]
    subdataset_dataset = gdal.Open(subdataset_path)
    num_bands = subdataset_dataset.RasterCount
    for i in range(1, num_bands + 1):
        #band_list.append(f"{subdataset_path}:Band {i}")
        band = subdataset_dataset.GetRasterBand(i)
        band_metadata = band.GetMetadata()
        #print(f"Metadata for {subdataset_path}: Band {i}")
        #print(band_metadata)
'''
# band metadata to match
desired_metadata = [{"BANDNAME": "B12"},
                    {"BANDNAME": "B11"},
                    {"BANDNAME": "B9"}]
# select bands
arrays = {}
selected_bands = []
for subdataset in d.GetSubDatasets():
    subdataset_path = subdataset[0]
    subdataset_dataset = gdal.Open(subdataset_path)

    for i in range(1, subdataset_dataset.RasterCount + 1):
        band = subdataset_dataset.GetRasterBand(i)
        band_metadata = band.GetMetadata()

        for k in band_metadata:
            for j in desired_metadata:
                try:
                    if band_metadata[k] == j[k]:
                        # print("Selected: ", band_metadata)
                        selected_bands += [[band, band_metadata, subdataset_dataset]]
                        #if j['BANDNAME'] == "B9":
                        arrays[str(band_metadata)] = band.ReadAsArray()
                except:
                    pass

#print(selected_bands)
#print(arrays)
# resample band if required
resampled_bands = []
for [band, m, sub_dataset] in selected_bands:
    geotransform = sub_dataset.GetGeoTransform()
    # Extract the pixel size (resolution)
    px_size_x = geotransform[1]
    px_size_y = geotransform[5]

    #print("band", m, band)
    print(m)
    #if m['BANDNAME'] == "B9":
    w = sys.argv[1].split('_')
    print(w)
    ds = w[2].split('T')[0]
    ofn = sys.argv[1][:-4] + "_" + m['BANDNAME'] + '.bin'
    hdr_f = ofn[:-3] + 'hdr'  # output header file name
    # print("+w", ofn)    
    print("+w", hdr_f)
    x_res, y_res = arrays[str(m)].shape
    driver = gdal.GetDriverByName('ENVI')
    output_dataset = driver.Create(ofn,
                                   x_res,
                                   y_res,
                                   1,
                                   gdal.GDT_Float32)    
    # Set the geotransform and projection from the input dataset
    output_dataset.SetGeoTransform(sub_dataset.GetGeoTransform())
    output_dataset.SetProjection(sub_dataset.GetProjection())
    rb = output_dataset.GetRasterBand(1)
    rb.WriteArray(arrays[str(m)])
    rb.SetDescription(' '.join([ds,
                                str(int(px_size_x)) + 'm:', 
                                m['BANDNAME'], 
                                str(m['WAVELENGTH']) + str(m['WAVELENGTH_UNIT'])]))
    
    # Close the datasets
    arrays[str(m)] = None
    output_dataset = None
    envi_header_cleanup([None, hdr_f])
