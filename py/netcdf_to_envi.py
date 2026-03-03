#!/usr/bin/env python3
"""
Convert netCDF file with multiple subdatasets to a single ENVI format file.
Handles non-georeferenced data (like VIIRS swath data) with optional geolocation.
"""

from osgeo import gdal, osr
import numpy as np
import sys
import os
import argparse
import re

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

def get_geolocation_arrays(geo_file):
    """Extract latitude and longitude arrays from geolocation file."""
    print(f"\nReading geolocation from: {geo_file}")
    
    subdatasets = get_subdatasets(geo_file)
    
    lat_array = None
    lon_array = None
    
    for sd_name, sd_desc in subdatasets:
        if 'latitude' in sd_desc.lower():
            print(f"  Found latitude: {sd_desc}")
            ds = gdal.Open(sd_name)
            if ds:
                lat_array = ds.GetRasterBand(1).ReadAsArray()
                ds = None
        elif 'longitude' in sd_desc.lower():
            print(f"  Found longitude: {sd_desc}")
            ds = gdal.Open(sd_name)
            if ds:
                lon_array = ds.GetRasterBand(1).ReadAsArray()
                ds = None
    
    if lat_array is not None and lon_array is not None:
        print(f"  Geolocation arrays shape: {lat_array.shape}")
        return lat_array, lon_array
    else:
        print("  Warning: Could not find both lat/lon arrays")
        return None, None

def create_geotransform_from_corners(lat_array, lon_array):
    """Create approximate geotransform from lat/lon corner points."""
    # Get corner coordinates
    ul_lat = lat_array[0, 0]
    ul_lon = lon_array[0, 0]
    ur_lat = lat_array[0, -1]
    ur_lon = lon_array[0, -1]
    ll_lat = lat_array[-1, 0]
    ll_lon = lon_array[-1, 0]
    lr_lat = lat_array[-1, -1]
    lr_lon = lon_array[-1, -1]
    
    height, width = lat_array.shape
    
    # Calculate pixel size (approximate, assuming regular grid)
    pixel_width = (ur_lon - ul_lon) / width
    pixel_height = (ll_lat - ul_lat) / height  # Will be negative
    
    # Upper-left corner coordinates
    geotransform = (ul_lon, pixel_width, 0, ul_lat, 0, pixel_height)
    
    # Report bounding box
    min_lon = min(ul_lon, ll_lon)
    max_lon = max(ur_lon, lr_lon)
    min_lat = min(ll_lat, lr_lat)
    max_lat = max(ul_lat, ur_lat)
    
    print(f"\n  Bounding box: ({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})")
    print(f"  Pixel size: {abs(pixel_width):.6f}° x {abs(pixel_height):.6f}°")
    
    return geotransform

def netcdf_to_envi(nc_file, output_file, geo_file=None, bands_only=False, pattern=None):
    """
    Convert netCDF with multiple subdatasets to single ENVI file.
    
    Parameters:
    -----------
    nc_file : str
        Path to input netCDF file
    output_file : str
        Path to output ENVI .bin file (will also create .hdr)
    geo_file : str, optional
        Path to geolocation netCDF file (e.g., VNP03IMG)
    bands_only : bool
        If True, exclude quality_flags and uncert_index bands
    pattern : str
        Regex pattern to filter subdatasets
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
    
    # Get geolocation if provided
    lat_array = None
    lon_array = None
    geotransform = None
    projection = None
    
    if geo_file and os.path.exists(geo_file):
        lat_array, lon_array = get_geolocation_arrays(geo_file)
        
        if lat_array is not None and lon_array is not None:
            # Resample geolocation to match target dimensions if needed
            if lat_array.shape != (target_height, target_width):
                print(f"  Resampling geolocation from {lat_array.shape} to ({target_height}, {target_width})")
                try:
                    lat_array = resample_array(lat_array, target_height, target_width)
                    lon_array = resample_array(lon_array, target_height, target_width)
                except ImportError:
                    print("  Warning: scipy not available for geolocation resampling")
                    lat_array = None
                    lon_array = None
            
            if lat_array is not None:
                geotransform = create_geotransform_from_corners(lat_array, lon_array)
                
                # Create WGS84 projection
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                projection = srs.ExportToWkt()
    
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
        # Apply filters
        if bands_only and ('quality_flags' in sd_desc or 'uncert_index' in sd_desc):
            print(f"\nSkipping: {sd_desc} (quality/uncertainty band)")
            continue
        
        if pattern and not re.search(pattern, sd_desc):
            print(f"\nSkipping: {sd_desc} (doesn't match pattern '{pattern}')")
            continue
        
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
    
    # Set geotransform
    if geotransform:
        out_ds.SetGeoTransform(geotransform)
        print(f"\nSet geotransform: {geotransform}")
    else:
        # Default pixel-based geotransform
        out_ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    
    # Set projection
    if projection:
        out_ds.SetProjection(projection)
        print(f"Set projection: WGS84")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert netCDF to ENVI format with optional geolocation'
    )
    parser.add_argument('input', help='Input netCDF file')
    parser.add_argument('output', help='Output ENVI .bin file')
    parser.add_argument('--geo', type=str, help='Geolocation file (e.g., VNP03IMG*.nc)')
    parser.add_argument('--bands-only', action='store_true',
                       help='Extract only main bands (exclude quality_flags and uncert_index)')
    parser.add_argument('--pattern', type=str,
                       help='Only include subdatasets matching this regex pattern')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.geo and not os.path.exists(args.geo):
        print(f"Error: Geolocation file not found: {args.geo}")
        sys.exit(1)
    
    netcdf_to_envi(
        args.input, 
        args.output, 
        geo_file=args.geo,
        bands_only=args.bands_only,
        pattern=args.pattern
    )


