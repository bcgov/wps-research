'''20230605 sentinel2_extract_swir.py
'''
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
# Desired band metadata to match
desired_metadata = [{"BANDNAME": "B12"},
                    {"BANDNAME": "B11"},
                    {"BANDNAME": "B9"}]

# Band selection based on metadata
selected_bands = []
arrays = {}
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
                        print("Selected: ", band_metadata)
                        selected_bands += [[band, band_metadata, subdataset_dataset]]
                        #if j['BANDNAME'] == "B9":
                        arrays[str(band_metadata)] = band.ReadAsArray()
                except:
                    pass

print(selected_bands)
# Perform operations on the selected band
print(arrays)

# resample band if required
resampled_bands = []
for [band, m, sub_dataset] in selected_bands:
    i = 0    
    print("band", m, band)
    #if m['BANDNAME'] == "B9":
    ofn= sys.argv[1][:-4] + "_" + m['BANDNAME'] + '.bin'
    print("+w", ofn)    
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
    output_dataset.GetRasterBand(i + 1).WriteArray(arrays[str(m)])

    # Close the datasets
    arrays[str(m)] = None
    output_dataset = None
