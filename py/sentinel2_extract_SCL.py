'''Extract Scene Classification Layer (SCL) from Sentinel-2 L2A .zip / .SAFE files.

Outputs (ENVI float32 format):
    <basename>_SCL.bin / .hdr       - Scene Classification Layer (SCL band)

Based on sentinel2_extract_cloud.py.
'''
from misc import err, args, exist, run, parfor
from envi import envi_header_cleanup
import multiprocessing as mp
from osgeo import gdal
import numpy as np
import sys
import os


def extract_scl(file_name):
    w = file_name.split('_')
    ds = w[2].split('T')[0]  # date string

    base = '.'.join(file_name.split('.')[:-1])
    scl_fn = base + '_SCL.bin'

    if file_name.split('.')[-1] == 'SAFE':
        file_name = file_name + os.path.sep + 'MTD_MSIL2A.xml'

    # Skip if output already exists
    if exist(scl_fn):
        print("Exists:", scl_fn, "skipping..")
        return

    def ignore_warnings(x, y, z): pass
    gdal.PushErrorHandler(ignore_warnings)

    d = gdal.Open(file_name)
    if d is None:
        print("ERROR: could not open", file_name)
        return

    # --------------- locate SCL band ---------------
    scl_array = None
    scl_sub_ds = None

    for subdataset in d.GetSubDatasets():
        subdataset_path = subdataset[0]
        subdataset_dataset = gdal.Open(subdataset_path)
        if subdataset_dataset is None:
            continue

        for i in range(1, subdataset_dataset.RasterCount + 1):
            band = subdataset_dataset.GetRasterBand(i)
            band_metadata = band.GetMetadata()

            for k in band_metadata:
                if band_metadata[k] == 'SCL':
                    scl_array = band.ReadAsArray().astype(np.float32)
                    scl_sub_ds = subdataset_dataset
                    break
            if scl_array is not None:
                break
        if scl_array is not None:
            break

    if scl_array is None:
        print("ERROR: SCL band not found in", file_name)
        return

    # --------------- reference geometry from the SCL subdataset ---------------
    ref_geo = scl_sub_ds.GetGeoTransform()
    ref_proj = scl_sub_ds.GetProjection()
    ref_xsize = scl_sub_ds.RasterXSize
    ref_ysize = scl_sub_ds.RasterYSize

    # --------------- Resample to 20m if needed ---------------
    target_res = 20.0
    src_xs, src_ys = abs(ref_geo[1]), abs(ref_geo[5])

    if abs(src_xs - target_res) > 0.5 or abs(src_ys - target_res) > 0.5:
        print("Resampling SCL from", src_xs, "m to", target_res, "m ...")
        mem_driver = gdal.GetDriverByName('MEM')

        input_ds = mem_driver.Create('', scl_array.shape[1], scl_array.shape[0], 1, gdal.GDT_Float32)
        input_ds.SetGeoTransform(ref_geo)
        input_ds.SetProjection(ref_proj)
        input_ds.GetRasterBand(1).WriteArray(scl_array)

        extent_x = scl_array.shape[1] * abs(ref_geo[1])
        extent_y = scl_array.shape[0] * abs(ref_geo[5])
        out_xsize = int(round(extent_x / target_res))
        out_ysize = int(round(extent_y / target_res))

        resampled_geo = list(ref_geo)
        resampled_geo[1] = target_res
        resampled_geo[5] = -target_res

        resampled_ds = mem_driver.Create('', out_xsize, out_ysize, 1, gdal.GDT_Float32)
        resampled_ds.SetGeoTransform(resampled_geo)
        resampled_ds.SetProjection(ref_proj)

        gdal.Warp(resampled_ds, input_ds, xRes=target_res, yRes=target_res, resampleAlg='near')
        scl_array = resampled_ds.GetRasterBand(1).ReadAsArray()

        ref_xsize = out_xsize
        ref_ysize = out_ysize
        ref_geo = tuple(resampled_geo)

        resampled_ds = None
        input_ds = None

    # --------------- Write SCL output ---------------
    driver = gdal.GetDriverByName('ENVI')

    print("Writing:", scl_fn)
    scl_ds = driver.Create(scl_fn, ref_xsize, ref_ysize, 1, gdal.GDT_Float32)
    scl_ds.SetProjection(ref_proj)
    scl_ds.SetGeoTransform(ref_geo)

    rb = scl_ds.GetRasterBand(1)
    rb.WriteArray(scl_array)
    rb.SetDescription(' '.join([ds, '20m:', 'SCL', 'scene_classification']))
    scl_ds = None

    # Clean up header
    hdr_f = scl_fn[:-4] + '.hdr'
    envi_header_cleanup([None, hdr_f])
    for f in [scl_fn + '.aux.xml', hdr_f + '.bak']:
        if os.path.exists(f):
            os.remove(f)

    print("Done:", base)


if __name__ == "__main__":

    file_name = None
    if len(args) == 2:
        file_name = args[1]
        if not exist(file_name):
            err('could not open input file: ' + file_name)
        if not (file_name.endswith('.zip') or file_name.endswith('.SAFE')):
            err('.zip or .SAFE expected')
        extract_scl(file_name)

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
        parfor(extract_scl, files, int(mp.cpu_count()))


