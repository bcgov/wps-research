#!/usr/bin/env python3
"""
Convert netCDF file with multiple subdatasets to a single ENVI format file.
Handles non-georeferenced data (like VIIRS swath data).
"""

from osgeo import gdal, osr
import numpy as np
import sys
import os

gdal.UseExceptions()

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
    
    info = {
        'name': subdataset_name,
        'width': ds.RasterXSize,
        'height': ds.RasterYSize,
        'bands': ds.RasterCount,
        'geotransform': ds.GetGeoTransform(),
        'projection': ds.GetProjection()
    }
    
    ds = None
    return info

def find_largest_dimensions(subdatasets):
    """Find the subdataset with the largest dimensions (finest resolution)."""
    largest = None
    max_pixels = 0
    
    for sd_name, sd_desc in subdatasets:
        info = get_dataset_info(sd_name)
        if info is None:
            continue
        
        pixels = info['width'] * info['height']
        if pixels > max_pixels:
            max_pixels = pixels
            largest = info
    
    return largest

def resample_array(data, target_height, target_width):
    """Resample array using nearest neighbor to target dimensions."""
    from scipy.ndimage import zoom
    
    if data.shape == (target_height, target_width):
        return data
    
    zoom_factors = (target_height / data.shape[0], target_width / data.shape[1])
    return zoom(data, zoom_factors, order=1)  # order=1 for bilinear

def netcdf_to_envi(nc_file, output_file, use_geolocation=False):
    """
    Convert netCDF with multiple subdatasets to single ENVI file.
    
    Parameters:
    -----------
    nc_file : str
        Path to input netCDF file
    output_file : str
        Path to output ENVI .bin file (will also create .hdr)
    use_geolocation : bool
        If True, try to find and use geolocation arrays for georeferencing
    """
    print(f"Processing: {nc_file}")
    
    # Get all subdatasets
    subdatasets = get_subdatasets(nc_file)
    print(f"Found {len(subdatasets)} subdatasets")
    
    if not subdatasets:
        print("No subdatasets found!")
        return
    
    # Print available subdatasets
    print("\nAvailable subdatasets:")
    for idx, (sd_name, sd_desc) in enumerate(subdatasets, 1):
        print(f"  {idx}. {sd_desc}")
    
    # Find largest dimensions (finest resolution)
    reference = find_largest_dimensions(subdatasets)
    if reference is None:
        print("Could not determine reference dimensions!")
        return
    
    print(f"\nReference dimensions: {reference['width']} x {reference['height']}")
    target_width = reference['width']
    target_height = reference['height']
    
    # Check if scipy is available for resampling
    try:
        import scipy.ndimage
        has_scipy = True
    except ImportError:
        print("Warning: scipy not available, will skip resampling of different sized bands")
        has_scipy = False
    
    # Prepare to collect all bands
    all_bands = []
    band_names = []
    
    # Process each subdataset
    for sd_name, sd_desc in subdatasets:
        print(f"\nProcessing: {sd_desc}")
        
        try:
            ds = gdal.Open(sd_name)
            if ds is None:
                print(f"  Skipping (could not open)")
                continue
            
            width = ds.RasterXSize
            height = ds.RasterYSize
            
            # Extract bands
            for band_idx in range(1, ds.RasterCount + 1):
                band = ds.GetRasterBand(band_idx)
                data = band.ReadAsArray()
                
                # Resample if needed
                if (height != target_height or width != target_width):
                    if has_scipy:
                        print(f"  Resampling from {width}x{height} to {target_width}x{target_height}")
                        data = resample_array(data, target_height, target_width)
                    else:
                        print(f"  Skipping (size mismatch: {width}x{height})")
                        continue
                
                all_bands.append(data.astype(np.float32))
                
                # Create band name from subdataset description
                var_name = sd_name.split(':')[-1].replace('/', '_')
                if ds.RasterCount > 1:
                    band_names.append(f"{var_name}_band{band_idx}")
                else:
                    band_names.append(var_name)
            
            ds = None
            
        except Exception as e:
            print(f"  Error processing: {e}")
            continue
    
    print(f"\nTotal bands collected: {len(all_bands)}")
    
    if not all_bands:
        print("No bands to write!")
        return
    
    # Create output ENVI file
    driver = gdal.GetDriverByName('ENVI')
    
    out_ds = driver.Create(
        output_file,
        target_width,
        target_height,
        len(all_bands),
        gdal.GDT_Float32
    )
    
    # Set basic geotransform (pixel coordinates if no georeferencing)
    if reference['geotransform'] and reference['geotransform'] != (0, 1, 0, 0, 0, 1):
        out_ds.SetGeoTransform(reference['geotransform'])
    else:
        # Default pixel-based geotransform
        out_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    
    # Set projection if available
    if reference['projection']:
        out_ds.SetProjection(reference['projection'])
    
    # Write bands
    for idx, (data, name) in enumerate(zip(all_bands, band_names), start=1):
        print(f"Writing band {idx}/{len(all_bands)}: {name}")
        band = out_ds.GetRasterBand(idx)
        band.WriteArray(data)
        band.SetDescription(name)
        
        # Set nodata value if there are fill values
        nodata = np.nan
        band.SetNoDataValue(nodata)
        
        band.FlushCache()
    
    # Close dataset
    out_ds = None
    
    print(f"\nSuccessfully created: {output_file}")
    hdr_file = output_file.rsplit('.', 1)[0] + '.hdr'
    print(f"Header file: {hdr_file}")
    
    # Try to find and report geolocation info
    print("\nChecking for geolocation data...")
    for sd_name, sd_desc in subdatasets:
        if 'longitude' in sd_desc.lower() or 'latitude' in sd_desc.lower():
            print(f"  Found: {sd_desc}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python netcdf_to_envi.py <input.nc> <output.bin>")
        sys.exit(1)
    
    nc_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(nc_file):
        print(f"Error: Input file not found: {nc_file}")
        sys.exit(1)
    
    netcdf_to_envi(nc_file, output_file)
