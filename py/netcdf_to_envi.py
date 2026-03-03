#!/usr/bin/env python3
"""
Convert netCDF file with multiple subdatasets to a single ENVI format file.
All subdatasets are resampled to the finest resolution present.
"""

from osgeo import gdal, osr
import numpy as np
import sys
import os

def get_subdatasets(nc_file):
    """Get all subdatasets from a netCDF file."""
    dataset = gdal.Open(nc_file)
    if dataset is None:
        raise ValueError(f"Could not open {nc_file}")
    
    subdatasets = dataset.GetSubDatasets()
    dataset = None
    return subdatasets

def get_dataset_info(subdataset_name):
    """Get resolution and dimensions of a subdataset."""
    ds = gdal.Open(subdataset_name)
    if ds is None:
        return None
    
    geotransform = ds.GetGeoTransform()
    pixel_width = abs(geotransform[1])
    pixel_height = abs(geotransform[5])
    resolution = min(pixel_width, pixel_height)
    
    info = {
        'name': subdataset_name,
        'resolution': resolution,
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'geotransform': geotransform,
        'projection': ds.GetProjection(),
        'bands': ds.RasterCount
    }
    
    ds = None
    return info

def find_finest_resolution(subdatasets):
    """Find the subdataset with the finest (smallest) resolution."""
    finest = None
    finest_res = float('inf')
    
    for sd_name, sd_desc in subdatasets:
        info = get_dataset_info(sd_name)
        if info is None:
            continue
        
        if info['resolution'] < finest_res:
            finest_res = info['resolution']
            finest = info
    
    return finest

def resample_to_reference(subdataset_name, reference_info):
    """Resample a subdataset to match reference resolution and extent."""
    # Create in-memory VRT for resampling
    src_ds = gdal.Open(subdataset_name)
    if src_ds is None:
        return None
    
    # Use gdal.Warp to resample
    warp_options = gdal.WarpOptions(
        format='MEM',
        width=reference_info['width'],
        height=reference_info['height'],
        outputBounds=(
            reference_info['geotransform'][0],  # min_x
            reference_info['geotransform'][3] + reference_info['height'] * reference_info['geotransform'][5],  # min_y
            reference_info['geotransform'][0] + reference_info['width'] * reference_info['geotransform'][1],  # max_x
            reference_info['geotransform'][3]  # max_y
        ),
        dstSRS=reference_info['projection'],
        resampleAlg=gdal.GRA_Bilinear,
        outputType=gdal.GDT_Float32
    )
    
    resampled = gdal.Warp('', src_ds, options=warp_options)
    src_ds = None
    
    return resampled

def netcdf_to_envi(nc_file, output_file):
    """
    Convert netCDF with multiple subdatasets to single ENVI file.
    
    Parameters:
    -----------
    nc_file : str
        Path to input netCDF file
    output_file : str
        Path to output ENVI .bin file (will also create .hdr)
    """
    print(f"Processing: {nc_file}")
    
    # Get all subdatasets
    subdatasets = get_subdatasets(nc_file)
    print(f"Found {len(subdatasets)} subdatasets")
    
    if not subdatasets:
        print("No subdatasets found!")
        return
    
    # Find finest resolution
    reference = find_finest_resolution(subdatasets)
    if reference is None:
        print("Could not determine reference resolution!")
        return
    
    print(f"Reference resolution: {reference['resolution']} units")
    print(f"Reference dimensions: {reference['width']} x {reference['height']}")
    
    # Prepare to collect all bands
    all_bands = []
    band_names = []
    
    # Process each subdataset
    for sd_name, sd_desc in subdatasets:
        print(f"Processing: {sd_desc}")
        
        # Resample to reference
        resampled = resample_to_reference(sd_name, reference)
        if resampled is None:
            print(f"  Skipping (could not resample)")
            continue
        
        # Extract bands
        for band_idx in range(1, resampled.RasterCount + 1):
            band = resampled.GetRasterBand(band_idx)
            data = band.ReadAsArray()
            all_bands.append(data)
            
            # Create band name from subdataset description
            var_name = sd_name.split(':')[-1]
            if resampled.RasterCount > 1:
                band_names.append(f"{var_name}_band{band_idx}")
            else:
                band_names.append(var_name)
        
        resampled = None
    
    print(f"Total bands collected: {len(all_bands)}")
    
    if not all_bands:
        print("No bands to write!")
        return
    
    # Create output ENVI file
    driver = gdal.GetDriverByName('ENVI')
    
    out_ds = driver.Create(
        output_file,
        reference['width'],
        reference['height'],
        len(all_bands),
        gdal.GDT_Float32
    )
    
    # Set geotransform and projection
    out_ds.SetGeoTransform(reference['geotransform'])
    out_ds.SetProjection(reference['projection'])
    
    # Write bands
    for idx, (data, name) in enumerate(zip(all_bands, band_names), start=1):
        print(f"Writing band {idx}/{len(all_bands)}: {name}")
        band = out_ds.GetRasterBand(idx)
        band.WriteArray(data)
        band.SetDescription(name)
        band.FlushCache()
    
    # Close dataset
    out_ds = None
    
    print(f"Successfully created: {output_file}")
    print(f"Header file: {output_file.rsplit('.', 1)[0]}.hdr")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python netcdf_to_envi.py <input.nc> <output.bin>")
        sys.exit(1)
    
    nc_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(nc_file):
        print(f"Error: Input file not found: {nc_file}")
        sys.exit(1)
    
    netcdf_to_envi(nc_file, output_file)


