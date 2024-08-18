'''20230605 modified from sentinel2_extract_swir.py

This script takes sentinel-2 .zip files as input.

.bin files are produced "as usual", with the exception that NAN is included for "undesirable" data areas, according to the Sentinel-2 (level-2) class map. 
'''
from misc import err, args, exist, run, parfor
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import sys
import os

def extract_cloudfree(file_name):
    w = file_name.split('_')  # split filename on '_'
    ds = w[2].split('T')[0]  # date string
    stack_fn = '.'.join(file_name.split('.')[:-1]) + '_cloudfree.bin' # output stack filename

    if file_name.split('.')[-1] == 'SAFE':
        file_name = file_name + os.path.sep + 'MTD_MSIL2A.xml'   

    if exist(stack_fn):
        print("Exists:", stack_fn, "skipping..")
        return

    def ignore_warnings(x, y, z): pass
    gdal.PushErrorHandler(ignore_warnings)  # suppress warnings
    
    d = gdal.Open(file_name)
    subdatasets =  d.GetSubDatasets()
    '''
[('SENTINEL2_L2A:/vsizip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.zip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.SAFE/MTD_MSIL2A.xml:10m:EPSG_32610', 'Bands B2, B3, B4, B8 with 10m resolution, UTM 10N'),

('SENTINEL2_L2A:/vsizip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.zip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.SAFE/MTD_MSIL2A.xml:20m:EPSG_32610', 'Bands B5, B6, B7, B8A, B11, B12, AOT, CLD, SCL, SNW, WVP with 20m resolution, UTM 10N'),

('SENTINEL2_L2A:/vsizip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.zip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.SAFE/MTD_MSIL2A.xml:60m:EPSG_32610', 'Bands B1, B9, AOT, CLD, SCL, SNW, WVP with 60m resolution, UTM 10N'),

('SENTINEL2_L2A:/vsizip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.zip/S2B_MSIL2A_20230725T192909_N0509_R142_T10VEJ_20230725T231042.SAFE/MTD_MSIL2A.xml:TCI:EPSG_32610', 'True color image, UTM 10N')]
    '''

    desired_bands = ['B11', 'B12', 'B9', 'SCL'] # ['B5', 'B6', 'B7', 'B8A', 'B9', 
                    # 'B11', 'B12', 'B9'] # 'CLD']
    desired_metadata = [{'BANDNAME': db} for db in desired_bands]
                        # {"BANDNAME": "B12"},
                        # {"BANDNAME": "B11"},
                        # {"BANDNAME": "B9"}]
    sbs={}
    arrays = {}
    selected_bands = []
    for subdataset in d.GetSubDatasets():  # select bands
        subdataset_path = subdataset[0]
        subdataset_dataset = gdal.Open(subdataset_path)
    
        for i in range(1, subdataset_dataset.RasterCount + 1):
            band = subdataset_dataset.GetRasterBand(i)
            band_metadata = band.GetMetadata()
            print(band_metadata) 
            if str(band_metadata) in arrays:
                continue
   
            for k in band_metadata:
                for j in desired_metadata:
                    try:
                        if band_metadata[k] == j[k]:  # print("Selected: ", band_metadata)
                            selected_bands += [[band, band_metadata, subdataset_dataset]]
                            sbs[band_metadata['BANDNAME']] = selected_bands[-1]
                            arrays[str(band_metadata)] = band.ReadAsArray().astype(np.float32)
                    except: pass
    
    selected_bands = [sbs['B12'], sbs['B11'], sbs['B9'], sbs['SCL']]  # reorder band selection
    
    resampled_bands = []
    target_sub_ds = selected_bands[0][2]  # last selected band is the one whose coordinates we should match
    geo_xform = target_sub_ds.GetGeoTransform()
    target_xs, target_ys = geo_xform[1], geo_xform[5]
    driver = gdal.GetDriverByName('ENVI')
    
    stack_ds = driver.Create(stack_fn,
                             target_sub_ds.RasterXSize,
                             target_sub_ds.RasterYSize,
                             len(selected_bands) - 1,
                             gdal.GDT_Float32)

    stack_ds.SetProjection(target_sub_ds.GetProjection())
    stack_ds.SetGeoTransform(target_sub_ds.GetGeoTransform())
    
    bi = 1
    cl_i = None
    for [band, m, sub_dataset] in selected_bands:
        band_name = m['BANDNAME']
        geotransform = sub_dataset.GetGeoTransform()
        px_sx, px_sy = geotransform[1], geotransform[5]
    
        if band_name == "B9":
            mem_driver = gdal.GetDriverByName('MEM')
            input_ds = mem_driver.Create('', band.XSize, band.YSize, 1, gdal.GDT_Float32)
            input_ds.SetGeoTransform(sub_dataset.GetGeoTransform())
            input_ds.SetProjection(sub_dataset.GetProjection())
            input_ds.GetRasterBand(1).WriteArray(arrays[str(m)])
    
            resampled_geotransform = list(input_ds.GetGeoTransform())
            resampled_geotransform[1] = target_xs
            resampled_geotransform[5] = target_ys
            resampled_ds = mem_driver.Create('', target_sub_ds.RasterXSize, target_sub_ds.RasterYSize, 1, gdal.GDT_Float32)
            resampled_ds.SetGeoTransform(resampled_geotransform)
            resampled_ds.SetProjection(input_ds.GetProjection())
    
            gdal.Warp(resampled_ds, input_ds, xRes=target_xs, yRes=target_ys, resampleAlg='bilinear')
            arrays[str(m)] = resampled_ds.GetRasterBand(1).ReadAsArray()
            resampled_ds = None
            input_ds = None
        
        if band_name == 'SCL':
            cl_i = str(m) 

    '''We don't want:
    0: No data
    1: Saturated or defective
    2: Dark area pixels
    3: Cloud shadows
    8: Cloud medium probability
    9: Cloud high probability
    10: Thin cirrus
    We don't want:
    arr2 = arr[np.where((arr >5) & (arr 5) | (arr % 5 == 0))]
    '''

    # calculate the valid areas:
    scl_d = arrays[cl_i]
    bad_data = np.where((scl_d <= 3) |
                        (scl_d == 8) |
                        (scl_d == 9) |
                        (scl_d == 10))
    

    # apply valid areas to other bands:
    for [band, m, sub_dataset] in selected_bands:
        band_name = m['BANDNAME']
        arrays[str(m)][bad_data] = 0.
        

    for [band, m, sub_dataset] in selected_bands:
        band_name = m['BANDNAME']
        geotransform = sub_dataset.GetGeoTransform()
        px_sx, px_sy = geotransform[1], geotransform[5]

        if band_name == 'SCL':  # don't write this one out
           continue

        # resume..
        rb = stack_ds.GetRasterBand(bi)
        d_out = arrays[str(m)]
        print("****", str(m), d_out.shape)
        try:
            print(arrays["{'BANDNAME': 'CLD'}"])
        except:
            pass
        print(arrays.keys())
        print("d_out", d_out)
        rb.WriteArray(d_out) # arrays[str(m)])
        rb.SetDescription(' '.join([ds,  # dates string
                                    str(int(px_sx)) + 'm:',  # resolution
                                    band_name,   # band name and wavelength
                                    (str(m['WAVELENGTH']) + str(m['WAVELENGTH_UNIT'])) if 'WAVELENGTH' in m else '']))
        # arrays[str(m)] = None
        bi += 1
    
    stack_ds = None
    hdr_f =  stack_fn[:-4] + '.hdr'
    envi_header_cleanup([None, hdr_f])
    xml_f = stack_fn + '.aux.xml'
    hdr_b = hdr_f + '.bak'
    for f in [xml_f, hdr_b]:
        if os.path.exists(f):
            os.remove(f)
    run('raster_zero_to_nan ' + stack_fn)


if __name__ == "__main__":
    
    file_name = None
    if len(args) == 2:
        file_name = args[1]
        if not exist(file_name):
            err('could not open input file: ' + d)
        if not file_name[-4:] == '.zip':
            err('zip expected')
        extract_cloudfree(file_name)

    else:
        files = [x.strip() for x in os.popen("ls -1 S*MSIL2A*.zip").readlines()]
        files += [x.strip() for x in os.popen("ls -1d S2*MSIL2A*.SAFE").readlines()]


        dirs = [x.strip() for x in os.popen('ls -1d L2_*').readlines()]
        for d in dirs:
            print(d)
            files += [x.strip() for x in os.popen("ls -1 " + d + os.path.sep + "S*MSIL2A*.zip").readlines()]
            files += [x.strip() for x in os.popen("ls -1d " + d + os.path.sep + "S2*MSIL2A*.SAFE").readlines()]

        for f in files:
            print(f)
        parfor(extract_cloudfree, files, int(mp.cpu_count())) 


'''
Table 3: SCL bit values

Classification
0: No data
1: Saturated or defective
2: Dark area pixels
3: Cloud shadows
4: Vegetation
5: Bare soils
6: Water
7: Unclassified
8: Cloud medium probability
9: Cloud high probability
10: Thin cirrus
11: Snow or ice


We don't want:
0: No data
1: Saturated or defective
2: Dark area pixels
3: Cloud shadows
8: Cloud medium probability
9: Cloud high probability
10: Thin cirrus
'''
