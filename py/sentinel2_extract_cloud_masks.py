'''Extract cloud probability and cloud shadow mask from Sentinel-2 L2A .zip / .SAFE files.

Outputs (ENVI float32 format):
    <basename>_cloudp.bin / .hdr       - Cloud probability (CLD band)
    <basename>_cloudshadow.bin / .hdr   - Cloud shadow binary mask (from SCL band, class 3)

Based on sentinel2_extract_cloudfree.py (20230605).
'''
from misc import err, args, exist, run, parfor
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import sys
import os


def extract_cloud(file_name):
    w = file_name.split('_')
    ds = w[2].split('T')[0]  # date string

    base = '.'.join(file_name.split('.')[:-1])
    cloudp_fn = base + '_cloudp.bin'
    cloudshadow_fn = base + '_cloudshadow.bin'

    if file_name.split('.')[-1] == 'SAFE':
        file_name = file_name + os.path.sep + 'MTD_MSIL2A.xml'

    # Skip if both outputs already exist
    if exist(cloudp_fn) and exist(cloudshadow_fn):
        print("Exists:", cloudp_fn, "and", cloudshadow_fn, "skipping..")
        return

    def ignore_warnings(x, y, z): pass
    gdal.PushErrorHandler(ignore_warnings)

    d = gdal.Open(file_name)
    if d is None:
        print("ERROR: could not open", file_name)
        return

    # --------------- locate CLD and SCL bands ---------------
    # CLD (cloud probability) is in 20m subdataset
    # SCL (scene classification) is in 20m subdataset
    desired_bands = ['CLD', 'SCL']
    desired_metadata = [{'BANDNAME': db} for db in desired_bands]

    arrays = {}
    selected_bands = {}

    for subdataset in d.GetSubDatasets():
        subdataset_path = subdataset[0]
        subdataset_dataset = gdal.Open(subdataset_path)
        if subdataset_dataset is None:
            continue

        for i in range(1, subdataset_dataset.RasterCount + 1):
            band = subdataset_dataset.GetRasterBand(i)
            band_metadata = band.GetMetadata()

            if str(band_metadata) in arrays:
                continue

            for k in band_metadata:
                for j in desired_metadata:
                    try:
                        if band_metadata[k] == j[k]:
                            band_name = band_metadata['BANDNAME']
                            selected_bands[band_name] = [band, band_metadata, subdataset_dataset]
                            arrays[band_name] = band.ReadAsArray().astype(np.float32)
                    except:
                        pass

    if 'CLD' not in selected_bands:
        print("ERROR: CLD band not found in", file_name)
        return
    if 'SCL' not in selected_bands:
        print("ERROR: SCL band not found in", file_name)
        return

    # --------------- reference geometry from the 20m CLD band ---------------
    ref_band_name = 'CLD'
    ref_sub_ds = selected_bands[ref_band_name][2]
    ref_geo = ref_sub_ds.GetGeoTransform()
    ref_proj = ref_sub_ds.GetProjection()
    ref_xsize = ref_sub_ds.RasterXSize
    ref_ysize = ref_sub_ds.RasterYSize
    target_xs = ref_geo[1]
    target_ys = ref_geo[5]

    # If SCL is at a different resolution, resample to match CLD (both should be 20m, but just in case)
    scl_sub_ds = selected_bands['SCL'][2]
    scl_geo = scl_sub_ds.GetGeoTransform()
    if scl_geo[1] != target_xs or scl_geo[5] != target_ys:
        print("Resampling SCL to match CLD resolution...")
        mem_driver = gdal.GetDriverByName('MEM')
        scl_band = selected_bands['SCL'][0]
        input_ds = mem_driver.Create('', scl_band.XSize, scl_band.YSize, 1, gdal.GDT_Float32)
        input_ds.SetGeoTransform(scl_sub_ds.GetGeoTransform())
        input_ds.SetProjection(scl_sub_ds.GetProjection())
        input_ds.GetRasterBand(1).WriteArray(arrays['SCL'])

        resampled_geo = list(ref_geo)
        resampled_ds = mem_driver.Create('', ref_xsize, ref_ysize, 1, gdal.GDT_Float32)
        resampled_ds.SetGeoTransform(resampled_geo)
        resampled_ds.SetProjection(ref_proj)

        gdal.Warp(resampled_ds, input_ds, xRes=target_xs, yRes=target_ys, resampleAlg='near')
        arrays['SCL'] = resampled_ds.GetRasterBand(1).ReadAsArray()
        resampled_ds = None
        input_ds = None

    driver = gdal.GetDriverByName('ENVI')

    # --------------- Write cloud probability output ---------------
    if not exist(cloudp_fn):
        print("Writing:", cloudp_fn)
        cloudp_ds = driver.Create(cloudp_fn, ref_xsize, ref_ysize, 1, gdal.GDT_Float32)
        cloudp_ds.SetProjection(ref_proj)
        cloudp_ds.SetGeoTransform(ref_geo)

        rb = cloudp_ds.GetRasterBand(1)
        rb.WriteArray(arrays['CLD'])
        rb.SetDescription(' '.join([ds, str(int(target_xs)) + 'm:', 'CLD', 'cloud_probability']))
        cloudp_ds = None

        # Clean up header
        hdr_f = cloudp_fn[:-4] + '.hdr'
        envi_header_cleanup([None, hdr_f])
        for f in [cloudp_fn + '.aux.xml', hdr_f + '.bak']:
            if os.path.exists(f):
                os.remove(f)
    else:
        print("Exists:", cloudp_fn, "skipping..")

    # --------------- Write cloud shadow mask output ---------------
    if not exist(cloudshadow_fn):
        print("Writing:", cloudshadow_fn)
        # SCL class 3 = Cloud shadows  (binary mask: 1 = shadow, 0 = not shadow)
        scl_data = arrays['SCL']
        shadow_mask = np.zeros_like(scl_data, dtype=np.float32)
        shadow_mask[scl_data == 3] = 1.0

        cloudshadow_ds = driver.Create(cloudshadow_fn, ref_xsize, ref_ysize, 1, gdal.GDT_Float32)
        cloudshadow_ds.SetProjection(ref_proj)
        cloudshadow_ds.SetGeoTransform(ref_geo)

        rb = cloudshadow_ds.GetRasterBand(1)
        rb.WriteArray(shadow_mask)
        rb.SetDescription(' '.join([ds, str(int(target_xs)) + 'm:', 'SCL_class3', 'cloud_shadow_mask']))
        cloudshadow_ds = None

        # Clean up header
        hdr_f = cloudshadow_fn[:-4] + '.hdr'
        envi_header_cleanup([None, hdr_f])
        for f in [cloudshadow_fn + '.aux.xml', hdr_f + '.bak']:
            if os.path.exists(f):
                os.remove(f)
    else:
        print("Exists:", cloudshadow_fn, "skipping..")

    print("Done:", base)


if __name__ == "__main__":

    file_name = None
    if len(args) == 2:
        file_name = args[1]
        if not exist(file_name):
            err('could not open input file: ' + file_name)
        if not (file_name.endswith('.zip') or file_name.endswith('.SAFE')):
            err('.zip or .SAFE expected')
        extract_cloud(file_name)

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
        parfor(extract_cloud, files, int(mp.cpu_count()))


