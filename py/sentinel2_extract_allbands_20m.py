'''
20241027 this is now the default downloader script for all-bands research
20240209 revised from sentinel2_extract_cloudfree_allbands.py to extract regardless of cloudfree
20230731 revised from sentinel2_extract_cloudfree.py to include all bands.

20230605 sentinel2_extract_swir.py

Weird errors may result in needing to downgrade to numpy version <2.0

```
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.21.0
```

'''
from misc import err, args, exist, run, parfor, sep
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import sys
import os

def extract(file_name):
    w = file_name.split('_')  # split filename on '_'
    ds = w[2].split('T')[0]  # date string
    stack_fn = '.'.join(file_name.split('.')[:-1]) + '.bin' #  file_name[:-4] + '.bin' # output stack filename
    
    if exist(stack_fn):
        print("Exists:", stack_fn, "skipping..")
        return

    def ignore_warnings(x, y, z): pass
    gdal.PushErrorHandler(ignore_warnings)  # suppress warnings
 
    file_name = file_name.rstrip(sep) + sep + 'MTD_MSIL1C.xml'
    print(file_name)
    d = gdal.Open(file_name)
    subdatasets =  d.GetSubDatasets()

    desired_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    desired_metadata = [{'BANDNAME': db} for db in desired_bands]
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
    
    selected_bands = [sbs['B1'], sbs['B2'], sbs['B3'], sbs['B4'], sbs['B5'], sbs['B6'], sbs['B7'], sbs['B8'], sbs['B8A'], sbs['B9'], sbs['B11'], sbs['B12']]  # reorder band selection
    resampled_bands = []
    target_sub_ds = selected_bands[-2][2]  # last selected band is the one whose coordinates we should match
    geo_xform = target_sub_ds.GetGeoTransform()
    target_xs, target_ys = geo_xform[1], geo_xform[5]
    driver = gdal.GetDriverByName('ENVI')
    
    stack_ds = driver.Create(stack_fn,
                             target_sub_ds.RasterXSize,
                             target_sub_ds.RasterYSize,
                             len(selected_bands),
                             gdal.GDT_Float32)

    stack_ds.SetProjection(target_sub_ds.GetProjection())
    stack_ds.SetGeoTransform(target_sub_ds.GetGeoTransform())
    
    bi = 1
    cl_i = None
    for [band, m, sub_dataset] in selected_bands:
        band_name = m['BANDNAME']
        geotransform = sub_dataset.GetGeoTransform()
        px_sx, px_sy = geotransform[1], geotransform[5]
    
        if band_name not in ['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']: # != "B12":
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
        
        if band_name == 'B11':
            cl_i = str(m) 

    bi = 1
    for [band, m, sub_dataset] in selected_bands:
        band_name = m['BANDNAME']
        geotransform = sub_dataset.GetGeoTransform()
        px_sx, px_sy = geotransform[1], geotransform[5]

        rb = stack_ds.GetRasterBand(bi)
        print(band_name)
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
    hdr_f =  ('.'.join(file_name.split('.')[:-1])) + '.hdr' #  file_name[:-4] + '.hdr'
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
        ext = file_name.split('.')[-1]
        if not ((ext == '.zip') or (ext == '.SAFE')):
            err('.zip or SAFE file expected')
        extract(file_name)

    else:
        files = [x.strip() for x in os.popen("ls -1 S*MSIL2A*.zip").readlines()]
        files += [x.strip() for x in os.popen("ls -1 S*MSIL1C*.zip").readlines()]
        files += [x.strip() for x in os.popen("ls -1 | grep .SAFE").readlines()]
        parfor(extract, files, 2) # int(mp.cpu_count()))
