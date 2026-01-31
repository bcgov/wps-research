#!/usr/bin/env python3
'''
20250130 sentinel2_extract_cloudfree_allbands_10m.py

Combined from:
- sentinel2_extract_allbands_10m.py (all bands at 10m, no cloud masking)
- sentinel2_extract_cloudfree_swirnir.py (cloud masking with SCL)
- sentinel2_extract_cloudfree_allbands.py (all bands at 20m with cloud masking)

This script extracts all Sentinel-2 bands resampled to 10m resolution,
with cloud/bad-data masking using the SCL (Scene Classification Layer).

Options:
    --rgb   Extract only B2 (Blue), B3 (Green), B4 (Red) at 10m resolution

Weird errors may result in needing to downgrade to numpy version <2.0:
    python3 -m pip install numpy==1.21.0
'''
from misc import err, args, exist, run, parfor, sep
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import sys
import os

# Check for --rgb flag
RGB_ONLY = '--rgb' in sys.argv
if RGB_ONLY:
    sys.argv.remove('--rgb')
    from misc import args  # re-import after modifying sys.argv


def extract(file_name):
    ext = file_name.split('.')[-1]  # .zip or .SAFE
    w = file_name.split('_')  # split filename on '_'
    ds = w[2].split('T')[0]  # date string
    
    # Output filename
    stack_fn = '.'.join(file_name.split('.')[:-1]) + '_cloudfree.bin'
    
    if exist(stack_fn):
        print("Exists:", stack_fn, "skipping..")
        return

    def ignore_warnings(x, y, z): pass
    gdal.PushErrorHandler(ignore_warnings)  # suppress warnings

    if ext == 'SAFE':
        fn1 = file_name.rstrip(sep) + sep + 'MTD_MSIL1C.xml'
        fn2 = file_name.rstrip(sep) + sep + 'MTD_MSIL2A.xml'
        if exist(fn2):
            file_name = fn2
        elif exist(fn1):
            file_name = fn1
        else:
            err('found neither: ' + str(fn1) + ' nor ' + str(fn2))

    print("file_name", file_name)
    d = gdal.Open(file_name)
    
    if d is None:
        filepath = 'bad_files.txt'
        file_is_empty = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        with open(filepath, "a") as f:
            if not file_is_empty:
                f.write("\n")
            f.write(file_name)
        return

    subdatasets = d.GetSubDatasets()

    # Define bands based on mode
    if RGB_ONLY:
        desired_bands = ['B2', 'B3', 'B4', 'SCL']  # Blue, Green, Red + SCL for masking
        output_bands = ['B2', 'B3', 'B4']
    else:
        desired_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
        output_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

    desired_metadata = [{'BANDNAME': db} for db in desired_bands]
    
    sbs = {}
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
                        if band_metadata[k] == j[k]:
                            selected_bands += [[band, band_metadata, subdataset_dataset]]
                            sbs[band_metadata['BANDNAME']] = selected_bands[-1]
                            arrays[str(band_metadata)] = band.ReadAsArray().astype(np.float32)
                    except:
                        pass

    # Check if SCL is available (L2A data)
    has_scl = 'SCL' in sbs
    
    # Reorder band selection
    if RGB_ONLY:
        selected_bands = [sbs['B2'], sbs['B3'], sbs['B4']]
        if has_scl:
            selected_bands.append(sbs['SCL'])
    else:
        selected_bands = [sbs['B1'], sbs['B2'], sbs['B3'], sbs['B4'], sbs['B5'], sbs['B6'], 
                         sbs['B7'], sbs['B8'], sbs['B8A'], sbs['B9'], sbs['B11'], sbs['B12']]
        if has_scl:
            selected_bands.append(sbs['SCL'])

    # Use B3 (Green, 10m native) as target resolution reference
    target_sub_ds = sbs['B3'][2]
    geo_xform = target_sub_ds.GetGeoTransform()
    target_xs, target_ys = geo_xform[1], geo_xform[5]
    
    driver = gdal.GetDriverByName('ENVI')
    
    # Number of output bands (excluding SCL)
    n_output_bands = len(output_bands)
    
    stack_ds = driver.Create(stack_fn,
                             target_sub_ds.RasterXSize,
                             target_sub_ds.RasterYSize,
                             n_output_bands,
                             gdal.GDT_Float32)

    stack_ds.SetProjection(target_sub_ds.GetProjection())
    stack_ds.SetGeoTransform(target_sub_ds.GetGeoTransform())

    # Bands that are native 10m (no resampling needed): B2, B3, B4, B8
    native_10m_bands = ['B2', 'B3', 'B4', 'B8']
    
    cl_i = None
    for [band, m, sub_dataset] in selected_bands:
        band_name = m['BANDNAME']
        geotransform = sub_dataset.GetGeoTransform()
        px_sx, px_sy = geotransform[1], geotransform[5]

        # Resample if not native 10m
        if band_name not in native_10m_bands:
            mem_driver = gdal.GetDriverByName('MEM')
            input_ds = mem_driver.Create('', band.XSize, band.YSize, 1, gdal.GDT_Float32)
            input_ds.SetGeoTransform(sub_dataset.GetGeoTransform())
            input_ds.SetProjection(sub_dataset.GetProjection())
            input_ds.GetRasterBand(1).WriteArray(arrays[str(m)])

            resampled_ds = mem_driver.Create('', target_sub_ds.RasterXSize, target_sub_ds.RasterYSize, 1, gdal.GDT_Float32)
            resampled_ds.SetGeoTransform(target_sub_ds.GetGeoTransform())
            resampled_ds.SetProjection(target_sub_ds.GetProjection())

            # Use nearest neighbor for SCL (classification data), bilinear for spectral bands
            resample_alg = 'near' if band_name == 'SCL' else 'bilinear'
            gdal.Warp(resampled_ds, input_ds, xRes=target_xs, yRes=target_ys, resampleAlg=resample_alg)
            arrays[str(m)] = resampled_ds.GetRasterBand(1).ReadAsArray()
            resampled_ds = None
            input_ds = None

        if band_name == 'SCL':
            cl_i = str(m)

    # Calculate bad data mask from SCL if available
    '''
    SCL Classification values we don't want:
    0: No data
    1: Saturated or defective
    2: Dark area pixels
    3: Cloud shadows
    8: Cloud medium probability
    9: Cloud high probability
    10: Thin cirrus
    '''
    bad_data = None
    if has_scl and cl_i is not None:
        scl_d = arrays[cl_i]
        bad_data = np.where((scl_d <= 3) |
                            (scl_d == 8) |
                            (scl_d == 9) |
                            (scl_d == 10))

    # Write output bands
    bi = 1
    for [band, m, sub_dataset] in selected_bands:
        band_name = m['BANDNAME']
        
        if band_name == 'SCL':  # don't write SCL to output
            continue

        geotransform = sub_dataset.GetGeoTransform()
        px_sx, px_sy = geotransform[1], geotransform[5]

        rb = stack_ds.GetRasterBand(bi)
        d_out = arrays[str(m)].copy()
        
        # Apply cloud mask
        if bad_data is not None:
            d_out[bad_data] = float('nan')

        print("****", str(m), d_out.shape)
        print(arrays.keys())
        
        rb.WriteArray(d_out)
        rb.SetDescription(' '.join([ds,  # date string
                                    '10m:',  # all output is 10m
                                    band_name,
                                    (str(m['WAVELENGTH']) + str(m['WAVELENGTH_UNIT'])) if 'WAVELENGTH' in m else '']))
        bi += 1

    stack_ds = None
    hdr_f = stack_fn[:-4] + '.hdr'
    envi_header_cleanup([None, hdr_f])
    xml_f = stack_fn + '.aux.xml'
    hdr_b = hdr_f + '.bak'
    for f in [xml_f, hdr_b]:
        if os.path.exists(f):
            os.remove(f)
    run('raster_zero_to_nan ' + stack_fn)


if __name__ == "__main__":
    
    file_name = None
    # Filter out --rgb from args count check
    filtered_args = [a for a in args if a != '--rgb']
    
    if len(filtered_args) == 2:
        file_name = filtered_args[1]
        if not exist(file_name):
            err('could not open input file: ' + file_name)
        ext = file_name.split('.')[-1]
        if not (ext == 'zip' or ext == 'SAFE'):
            err('.zip or .SAFE file expected')
        extract(file_name)

    else:
        files = [x.strip() for x in os.popen("ls -1 S*MSIL2A*.zip 2>/dev/null").readlines()]
        files += [x.strip() for x in os.popen("ls -1 S*MSIL1C*.zip 2>/dev/null").readlines()]
        files += [x.strip() for x in os.popen("ls -1d S2*.SAFE 2>/dev/null").readlines()]
        
        # Also check L2_* directories
        dirs = [x.strip() for x in os.popen('ls -1d L2_* 2>/dev/null').readlines()]
        for d in dirs:
            files += [x.strip() for x in os.popen("ls -1 " + d + sep + "S*MSIL2A*.zip 2>/dev/null").readlines()]
            files += [x.strip() for x in os.popen("ls -1d " + d + sep + "S2*.SAFE 2>/dev/null").readlines()]
        
        # Clean up file list
        files = [f.strip().strip(':') for f in files if f.strip() != '']
        
        print("Mode:", "RGB only (B2, B3, B4)" if RGB_ONLY else "All bands")
        print("files", files)
        parfor(extract, files, min(8, int(mp.cpu_count())))


'''
Sentinel-2 Band Information:
- B1:  60m, Coastal aerosol (443nm)
- B2:  10m, Blue (490nm)
- B3:  10m, Green (560nm)
- B4:  10m, Red (665nm)
- B5:  20m, Vegetation Red Edge (705nm)
- B6:  20m, Vegetation Red Edge (740nm)
- B7:  20m, Vegetation Red Edge (783nm)
- B8:  10m, NIR (842nm)
- B8A: 20m, Vegetation Red Edge (865nm)
- B9:  60m, Water Vapour (945nm)
- B11: 20m, SWIR (1610nm)
- B12: 20m, SWIR (2190nm)

SCL Classification (L2A only):
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

Masked values (set to NaN):
0, 1, 2, 3, 8, 9, 10
'''
