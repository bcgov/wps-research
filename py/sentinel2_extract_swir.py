'''20230605 sentinel2_extract_swir.py
for subdataset in subdatasets:
    subdataset_path = subdataset[0]
    subdataset_dataset = gdal.Open(subdataset_path)
    num_bands = subdataset_dataset.RasterCount
    for i in range(1, num_bands + 1):
        #band_list.append(f"{subdataset_path}:Band {i}")
        band = subdataset_dataset.GetRasterBand(i)
        band_metadata = band.GetMetadata()
        #print(f"Metadata for {subdataset_path}: Band {i}")
        #print(band_metadata)'''
from envi import envi_header_cleanup
from osgeo import gdal
import numpy as np
import sys
d = gdal.Open(sys.argv[1])
subdatasets =  d.GetSubDatasets()
desired_metadata = [{"BANDNAME": "B12"},
                    {"BANDNAME": "B11"},
                    {"BANDNAME": "B9"}]
arrays = {}
selected_bands = []
for subdataset in d.GetSubDatasets():  # select bands
    subdataset_path = subdataset[0]
    subdataset_dataset = gdal.Open(subdataset_path)

    for i in range(1, subdataset_dataset.RasterCount + 1):
        band = subdataset_dataset.GetRasterBand(i)
        band_metadata = band.GetMetadata()

        for k in band_metadata:
            for j in desired_metadata:
                try:
                    if band_metadata[k] == j[k]:  # print("Selected: ", band_metadata)
                        selected_bands += [[band, band_metadata, subdataset_dataset]]
                        arrays[str(band_metadata)] = band.ReadAsArray().astype(np.float32)
                except: pass

resampled_bands = []
target_sub_ds = selected_bands[0][2]
geo_xform = target_sub_ds.GetGeoTransform()
target_xs, target_ys = geo_xform[1], geo_xform[5]

for [band, m, sub_dataset] in selected_bands:
    w = sys.argv[1].split('_')  # split filename on '_'
    ds = w[2].split('T')[0]  # date string
    ofn = sys.argv[1][:-4] + "_" + m['BANDNAME'] + '.bin'
    hdr_f = ofn[:-3] + 'hdr'  # output header file name

    band_name = m['BANDNAME']
    geotransform = sub_dataset.GetGeoTransform()
    px_sx, px_sy = geotransform[1], geotransform[5]
    nodata_val = band.GetNoDataValue()
    
    ix = arrays[str(m)] == nodata_val
    arrays[str(m)][ix] = float('nan')
    band.SetNoDataValue(float('nan'))

    if band_name == "B9":
        mem_driver = gdal.GetDriverByName('MEM')
        input_ds = mem_driver.Create('', band.XSize, band.YSize, 1, gdal.GDT_Float32)
        input_ds.SetGeoTransform(sub_dataset.GetGeoTransform())
        input_ds.SetProjection(sub_dataset.GetProjection())
        input_ds.GetRasterBand(1).SetNoDataValue(float('nan'))
        input_ds.GetRasterBand(1).WriteArray(arrays[str(m)]) 
        
        resampled_geotransform = list(input_ds.GetGeoTransform())
        resampled_geotransform[1] = target_xs
        resampled_geotransform[5] = target_ys 
        resampled_ds = mem_driver.Create('', target_sub_ds.RasterXSize, target_sub_ds.RasterYSize, 1, gdal.GDT_Float32) 
        resampled_ds.SetGeoTransform(resampled_geotransform)
        resampled_ds.SetProjection(input_ds.GetProjection())
        resampled_ds.GetRasterBand(1).SetNoDataValue(float('nan'))

        gdal.Warp(resampled_ds, input_ds, xRes=target_xs, yRes=target_ys, resampleAlg='bilinear')
        driver = gdal.GetDriverByName("ENVI")
        output_dataset = driver.CreateCopy(ofn, resampled_ds)
        resampled_ds = None
        output_ds = None
        input_ds = None
    else:
        x_res, y_res = arrays[str(m)].shape
        driver = gdal.GetDriverByName('ENVI')
        output_dataset = driver.Create(ofn, x_res, y_res, 1, gdal.GDT_Float32)    
        output_dataset.SetGeoTransform(sub_dataset.GetGeoTransform())
        output_dataset.SetProjection(sub_dataset.GetProjection())
        rb = output_dataset.GetRasterBand(1)
        rb.SetNoDataValue(float('nan'))
        rb.WriteArray(arrays[str(m)])
        rb.SetDescription(' '.join([ds,  # dates string
                                    str(int(px_sx)) + 'm:',  # resolution
                                    band_name,   # band name and wavelength
                                    str(m['WAVELENGTH']) + str(m['WAVELENGTH_UNIT'])]))
        output_dataset = None  
    # Close the datasets
    arrays[str(m)] = None
    os.remove(ofn + 'aux.xml')
    envi_header_cleanup([None, hdr_f])
