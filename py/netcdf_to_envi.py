#!/usr/bin/env python3
"""
Convert VIIRS netCDF file to ENVI format with automatic geolocation.
Automatically downloads matching VNP03IMG geolocation file if needed.
"""

from osgeo import gdal, osr
import numpy as np
import sys
import os
import argparse
import re
import requests
from urllib.parse import urlparse
import glob

gdal.UseExceptions()

def parse_viirs_filename(filename):
    """
    Parse VIIRS filename to extract metadata.
    Format: VNP02IMG.A2025245.0000.002.2025245091212.nc
    Returns: dict with product, date, time, collection, processing_time
    """
    basename = os.path.basename(filename)
    parts = basename.split('.')
    
    if len(parts) < 5:
        return None
    
    return {
        'product': parts[0],  # VNP02IMG
        'date': parts[1],     # A2025245
        'time': parts[2],     # 0000
        'collection': parts[3],  # 002
        'processing': parts[4].replace('.nc', '')  # 2025245091212
    }

def find_local_geolocation_file(img_file):
    """
    Look for matching VNP03IMG file in the same directory.
    """
    metadata = parse_viirs_filename(img_file)
    if not metadata:
        return None
    
    directory = os.path.dirname(img_file) or '.'
    
    # Build search pattern for matching geolocation file
    pattern = f"VNP03IMG.{metadata['date']}.{metadata['time']}.{metadata['collection']}.*.nc"
    search_path = os.path.join(directory, pattern)
    
    matches = glob.glob(search_path)
    
    if matches:
        print(f"Found local geolocation file: {matches[0]}")
        return matches[0]
    
    return None

def download_geolocation_file(img_file):
    """
    Download matching VNP03IMG geolocation file from LAADS DAAC.
    Requires EARTHDATA_TOKEN environment variable.
    """
    metadata = parse_viirs_filename(img_file)
    if not metadata:
        print("Error: Could not parse VIIRS filename")
        return None
    
    # Check for authentication token
    token = os.environ.get('EARTHDATA_TOKEN')
    if not token:
        print("\nError: EARTHDATA_TOKEN environment variable not set")
        print("To download geolocation files, you need a NASA Earthdata token:")
        print("1. Register at https://urs.earthdata.nasa.gov/")
        print("2. Generate an app token at https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#tokens")
        print("3. Set environment variable: export EARTHDATA_TOKEN='your_token_here'")
        return None
    
    # Build URL for geolocation file
    # Extract year and day of year from date field (A2025245)
    year = metadata['date'][1:5]
    doy = metadata['date'][5:8]
    
    # LAADS DAAC URL structure
    base_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData"
    collection = "5000"  # VIIRS collection number
    product = "VNP03IMG"
    
    # Construct directory URL
    dir_url = f"{base_url}/{collection}/{product}/{year}/{doy}/"
    
    print(f"\nSearching for geolocation file at LAADS DAAC...")
    print(f"URL: {dir_url}")
    
    # List files in directory
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.get(dir_url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML to find matching file
        # Look for file matching our date/time/collection
        pattern = f"VNP03IMG.{metadata['date']}.{metadata['time']}.{metadata['collection']}"
        
        matching_files = []
        for line in response.text.split('\n'):
            if pattern in line and '.nc' in line:
                # Extract filename from HTML
                match = re.search(r'href="([^"]*\.nc)"', line)
                if match:
                    matching_files.append(match.group(1))
        
        if not matching_files:
            print(f"Error: No matching geolocation file found")
            return None
        
        geo_filename = matching_files[0]
        geo_url = dir_url + geo_filename
        
        print(f"Found: {geo_filename}")
        print(f"Downloading from: {geo_url}")
        
        # Download file
        output_path = os.path.join(os.path.dirname(img_file) or '.', geo_filename)
        
        response = requests.get(geo_url, headers=headers, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\nDownloaded to: {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

def get_geolocation_file(img_file):
    """
    Get geolocation file - first check locally, then download if needed.
    """
    # First, try to find local file
    geo_file = find_local_geolocation_file(img_file)
    
    if geo_file:
        return geo_file
    
    print("\nNo local geolocation file found.")
    
    # Ask user if they want to download
    response = input("Download geolocation file from NASA LAADS DAAC? (y/n): ")
    
    if response.lower() == 'y':
        return download_geolocation_file(img_file)
    else:
        print("\nYou can manually download the matching VNP03IMG file from:")
        print("https://ladsweb.modaps.eosdis.nasa.gov/search/")
        return None

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
    """Resample array using bilinear interpolation to target dimensions."""
    from scipy.ndimage import zoom
    
    if data.shape == (target_height, target_width):
        return data
    
    zoom_factors = (target_height / data.shape[0], target_width / data.shape[1])
    return zoom(data, zoom_factors, order=1)  # order=1 for bilinear

def get_geolocation_arrays(geo_file):
    """Extract latitude and longitude arrays from geolocation file."""
    print(f"\nReading geolocation from: {os.path.basename(geo_file)}")
    
    subdatasets = get_subdatasets(geo_file)
    
    lat_array = None
    lon_array = None
    
    for sd_name, sd_desc in subdatasets:
        if 'latitude' in sd_desc.lower() and 'latitude' == sd_name.split(':')[-1].lower():
            print(f"  Found latitude: {sd_desc}")
            ds = gdal.Open(sd_name)
            if ds:
                lat_array = ds.GetRasterBand(1).ReadAsArray()
                ds = None
        elif 'longitude' in sd_desc.lower() and 'longitude' == sd_name.split(':')[-1].lower():
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

def netcdf_to_envi(nc_file, output_file, geo_file, bands_only=False, pattern=None):
    """
    Convert netCDF with multiple subdatasets to single ENVI file.
    
    Parameters:
    -----------
    nc_file : str
        Path to input netCDF file
    output_file : str
        Path to output ENVI .bin file (will also create .hdr)
    geo_file : str
        Path to geolocation netCDF file (VNP03IMG) - REQUIRED
    bands_only : bool
        If True, exclude quality_flags and uncert_index bands
    pattern : str
        Regex pattern to filter subdatasets
    """
    print(f"Processing: {nc_file}")
    
    if not geo_file:
        print("\nError: Geolocation file is required!")
        print("Geolocation provides the lat/lon coordinates for proper georeferencing.")
        return False
    
    # Get all subdatasets
    subdatasets = get_subdatasets(nc_file)
    print(f"Found {len(subdatasets)} subdatasets")
    
    if not subdatasets:
        print("No subdatasets found!")
        return False
    
    # Print available subdatasets
    print("\nAvailable subdatasets:")
    for idx, (sd_name, sd_desc) in enumerate(subdatasets, 1):
        print(f"  {idx}. {sd_desc}")
    
    # Find largest dimensions (finest resolution)
    reference = find_largest_dimensions(subdatasets)
    if reference is None:
        print("Could not determine reference dimensions!")
        return False
    
    print(f"\nReference dimensions: {reference['width']} x {reference['height']}")
    target_width = reference['width']
    target_height = reference['height']
    
    # Get geolocation arrays
    lat_array, lon_array = get_geolocation_arrays(geo_file)
    
    if lat_array is None or lon_array is None:
        print("\nError: Could not read geolocation arrays!")
        return False
    
    # Resample geolocation to match target dimensions if needed
    if lat_array.shape != (target_height, target_width):
        print(f"Resampling geolocation from {lat_array.shape} to ({target_height}, {target_width})")
        try:
            lat_array = resample_array(lat_array, target_height, target_width)
            lon_array = resample_array(lon_array, target_height, target_width)
        except ImportError:
            print("Error: scipy is required for geolocation resampling")
            print("Install with: pip install scipy")
            return False
    
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
        return False
    
    # Create output ENVI file
    driver = gdal.GetDriverByName('ENVI')
    
    out_ds = driver.Create(
        output_file,
        target_width,
        target_height,
        len(all_bands),
        gdal.GDT_Float32
    )
    
    # Set geotransform and projection
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    print(f"\nGeotransform: {geotransform}")
    print(f"Projection: WGS84 (EPSG:4326)")
    
    # Write bands
    for idx, (data, name) in enumerate(zip(all_bands, band_names), start=1):
        print(f"Writing band {idx}/{len(all_bands)}: {name}")
        band = out_ds.GetRasterBand(idx)
        band.WriteArray(data)
        band.SetDescription(name)
        
        # Set nodata value
        band.SetNoDataValue(np.nan)
        
        band.FlushCache()
    
    # Close dataset
    out_ds = None
    
    print(f"\n✓ Successfully created: {output_file}")
    hdr_file = output_file.rsplit('.', 1)[0] + '.hdr'
    print(f"✓ Header file: {hdr_file}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert VIIRS netCDF to georeferenced ENVI format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find local geolocation file
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin
  
  # Specify geolocation file
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin --geo VNP03IMG.nc
  
  # Only imagery bands (exclude quality flags)
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin --bands-only
  
  # Only specific bands
  python3 netcdf_to_envi.py VNP02IMG.nc output.bin --pattern "I0[1-3]"

Note: Geolocation file (VNP03IMG) is required for proper georeferencing.
      The script will search locally first, then offer to download if needed.
        """
    )
    parser.add_argument('input', help='Input VIIRS netCDF file (VNP02IMG)')
    parser.add_argument('output', help='Output ENVI .bin file')
    parser.add_argument('--geo', type=str, 
                       help='Geolocation file (VNP03IMG*.nc) - auto-detected if not specified')
    parser.add_argument('--bands-only', action='store_true',
                       help='Extract only imagery bands (exclude quality_flags and uncert_index)')
    parser.add_argument('--pattern', type=str,
                       help='Only include subdatasets matching this regex pattern')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Get geolocation file
    geo_file = args.geo
    
    if not geo_file:
        geo_file = get_geolocation_file(args.input)
    
    if not geo_file or not os.path.exists(geo_file):
        print("\nError: Geolocation file is required but not found.")
        print("Please download the matching VNP03IMG file or provide it with --geo")
        sys.exit(1)
    
    # Require scipy
    try:
        import scipy
    except ImportError:
        print("\nError: scipy is required for this script")
        print("Install with: pip install scipy")
        sys.exit(1)
    
    success = netcdf_to_envi(
        args.input, 
        args.output, 
        geo_file=geo_file,
        bands_only=args.bands_only,
        pattern=args.pattern
    )
    
    sys.exit(0 if success else 1)

