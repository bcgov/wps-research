#!/usr/bin/env python3
"""
Spatial Autocorrelation Calculator

Calculates spatial autocorrelation (Moran's I) for:
1. Each band individually (spatial autocorrelation within a band)
2. All pairwise combinations of bands (cross-correlation between bands)

Output is written in 32-bit float ENVI format (.bin and .hdr files).

Supports reading via:
- GDAL (if available) - best for geospatial formats
- tifffile - for TIFF/GeoTIFF files
- imageio - for standard image formats

Band names are extracted from:
- GDAL band descriptions
- TIFF page names/descriptions
- ENVI header files
"""

import numpy as np
import sys
import os
from itertools import combinations
from typing import List, Tuple, Optional, Dict, Any
import argparse
import re


# Try to import GDAL, fall back to tifffile
GDAL_AVAILABLE = False
try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    pass


def read_envi_header(hdr_path: str) -> Dict[str, Any]:
    """
    Parse an ENVI header file.
    
    Args:
        hdr_path: Path to the .hdr file
        
    Returns:
        Dictionary of header parameters
    """
    header = {}
    
    if not os.path.exists(hdr_path):
        return header
    
    with open(hdr_path, 'r') as f:
        content = f.read()
    
    # Handle multi-line values in curly braces
    # First, join lines that are part of multi-line values
    lines = []
    current_line = ""
    brace_depth = 0
    
    for char in content:
        current_line += char
        if char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
        elif char == '\n' and brace_depth == 0:
            lines.append(current_line.strip())
            current_line = ""
    
    if current_line.strip():
        lines.append(current_line.strip())
    
    for line in lines:
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Remove curly braces and parse lists
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1].strip()
                # Check if it's a list of values
                if ',' in value:
                    # Split by comma, handling potential newlines
                    items = [item.strip() for item in value.split(',')]
                    items = [item for item in items if item]  # Remove empty items
                    value = items
            
            header[key] = value
    
    return header


def read_image_gdal(filepath: str) -> Tuple[np.ndarray, Any, List[str], Dict[str, Any]]:
    """
    Read image using GDAL.
    
    Returns:
        Tuple of (image data, dataset, band names, metadata dict)
    """
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f"Could not open file: {filepath}")
    
    # Read all bands
    bands = []
    band_names = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        data = band.ReadAsArray().astype(np.float64)
        bands.append(data)
        
        name = band.GetDescription()
        if not name:
            name = f"Band_{i}"
        band_names.append(name)
    
    # Extract metadata
    metadata = {
        'geotransform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'nodata': ds.GetRasterBand(1).GetNoDataValue(),
    }
    
    return np.array(bands), ds, band_names, metadata


def read_image_tifffile(filepath: str) -> Tuple[np.ndarray, None, List[str], Dict[str, Any]]:
    """
    Read image using tifffile.
    
    Returns:
        Tuple of (image data, None, band names, metadata dict)
    """
    import tifffile
    
    with tifffile.TiffFile(filepath) as tif:
        # Read image data
        data = tif.asarray()
        
        # Handle different array shapes
        if data.ndim == 2:
            # Single band - add band dimension
            data = data[np.newaxis, :, :]
        elif data.ndim == 3:
            # Check if bands are last dimension (common in image formats)
            if data.shape[2] <= 4 and data.shape[0] > 4 and data.shape[1] > 4:
                # Shape is (rows, cols, bands) - transpose to (bands, rows, cols)
                data = np.transpose(data, (2, 0, 1))
        
        n_bands = data.shape[0]
        
        # Try to get band names from page descriptions
        # Filter out JSON-like metadata that isn't a real band name
        band_names = []
        for i, page in enumerate(tif.pages):
            if i >= n_bands:
                break
            name = None
            if hasattr(page, 'description') and page.description:
                desc = page.description
                # Skip if it looks like JSON metadata
                if not (desc.startswith('{') or desc.startswith('[') or '\"axes\"' in desc):
                    name = desc
            if not name:
                name = f"Band_{i+1}"
            band_names.append(name)
        
        # Pad if needed
        while len(band_names) < n_bands:
            band_names.append(f"Band_{len(band_names)+1}")
        
        # Extract GeoTIFF metadata if available
        metadata = {
            'geotransform': None,
            'projection': None,
            'nodata': None,
        }
        
        # Try to get geotiff tags
        if tif.pages[0].tags:
            tags = tif.pages[0].tags
            
            # ModelPixelScale and ModelTiePoint for geotransform
            if 'ModelPixelScaleTag' in tags and 'ModelTiepointTag' in tags:
                scale = tags['ModelPixelScaleTag'].value
                tiepoint = tags['ModelTiepointTag'].value
                if len(scale) >= 2 and len(tiepoint) >= 6:
                    # GeoTIFF convention: tiepoint is (i, j, k, x, y, z)
                    # Geotransform: (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
                    metadata['geotransform'] = (
                        tiepoint[3] - tiepoint[0] * scale[0],
                        scale[0],
                        0,
                        tiepoint[4] + tiepoint[1] * scale[1],
                        0,
                        -scale[1]
                    )
            
            # GeoKeyDirectoryTag for projection info
            if 'GeoKeyDirectoryTag' in tags:
                metadata['geo_keys'] = tags['GeoKeyDirectoryTag'].value
            
            # NoData value
            if 'GDAL_NODATA' in tags:
                try:
                    metadata['nodata'] = float(tags['GDAL_NODATA'].value)
                except:
                    pass
    
    return data.astype(np.float64), None, band_names, metadata


def read_image_envi(filepath: str) -> Tuple[np.ndarray, None, List[str], Dict[str, Any]]:
    """
    Read ENVI format image (.bin/.hdr or .img/.hdr).
    
    Returns:
        Tuple of (image data, None, band names, metadata dict)
    """
    # Find header file
    base = os.path.splitext(filepath)[0]
    hdr_candidates = [base + '.hdr', filepath + '.hdr']
    hdr_path = None
    for candidate in hdr_candidates:
        if os.path.exists(candidate):
            hdr_path = candidate
            break
    
    if not hdr_path:
        raise ValueError(f"Could not find ENVI header file for: {filepath}")
    
    header = read_envi_header(hdr_path)
    
    # Get required parameters
    samples = int(header.get('samples', 0))
    lines = int(header.get('lines', 0))
    bands = int(header.get('bands', 1))
    data_type = int(header.get('data type', 4))
    interleave = header.get('interleave', 'bsq').lower()
    byte_order = int(header.get('byte order', 0))
    header_offset = int(header.get('header offset', 0))
    
    # Map ENVI data types to numpy
    dtype_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        6: np.complex64,
        9: np.complex128,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    
    dtype = dtype_map.get(data_type, np.float32)
    
    # Handle byte order
    if byte_order == 1:  # Big endian
        dtype = np.dtype(dtype).newbyteorder('>')
    
    # Read binary data
    bin_path = filepath
    if filepath.endswith('.hdr'):
        bin_path = base + '.bin'
        if not os.path.exists(bin_path):
            bin_path = base + '.img'
            if not os.path.exists(bin_path):
                bin_path = base + '.dat'
    
    with open(bin_path, 'rb') as f:
        f.seek(header_offset)
        data = np.fromfile(f, dtype=dtype)
    
    # Reshape based on interleave
    if interleave == 'bsq':
        data = data.reshape((bands, lines, samples))
    elif interleave == 'bil':
        data = data.reshape((lines, bands, samples))
        data = np.transpose(data, (1, 0, 2))
    elif interleave == 'bip':
        data = data.reshape((lines, samples, bands))
        data = np.transpose(data, (2, 0, 1))
    
    # Get band names
    band_names = header.get('band names', [])
    if isinstance(band_names, str):
        band_names = [band_names]
    
    # Ensure we have enough band names
    while len(band_names) < bands:
        band_names.append(f"Band_{len(band_names)+1}")
    
    # Build metadata
    metadata = {
        'geotransform': None,
        'projection': None,
        'nodata': None,
        'map_info': header.get('map info'),
        'coordinate_system_string': header.get('coordinate system string'),
    }
    
    # Parse map info if present
    map_info = header.get('map info')
    if map_info:
        if isinstance(map_info, str):
            # Parse map info string
            parts = [p.strip() for p in map_info.split(',')]
            if len(parts) >= 7:
                try:
                    proj_name = parts[0]
                    ref_x = float(parts[1])
                    ref_y = float(parts[2])
                    easting = float(parts[3])
                    northing = float(parts[4])
                    pixel_x = float(parts[5])
                    pixel_y = float(parts[6])
                    
                    # Convert to GDAL geotransform
                    # (x_origin, pixel_width, 0, y_origin, 0, -pixel_height)
                    x_origin = easting - (ref_x - 1) * pixel_x
                    y_origin = northing + (ref_y - 1) * pixel_y
                    metadata['geotransform'] = (x_origin, pixel_x, 0, y_origin, 0, -pixel_y)
                except (ValueError, IndexError):
                    pass
    
    return data.astype(np.float64), None, band_names, metadata


def read_image(filepath: str) -> Tuple[np.ndarray, Any, List[str], Dict[str, Any]]:
    """
    Read a multi-band image using the best available method.
    
    Args:
        filepath: Path to the input image
        
    Returns:
        Tuple of (image data as numpy array, optional dataset, band names, metadata)
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    # Check for ENVI format
    if ext in ['.bin', '.img', '.dat', '.hdr'] or os.path.exists(filepath + '.hdr'):
        try:
            return read_image_envi(filepath)
        except Exception as e:
            print(f"ENVI read failed: {e}")
    
    # Try GDAL first if available
    if GDAL_AVAILABLE:
        try:
            return read_image_gdal(filepath)
        except Exception as e:
            print(f"GDAL read failed: {e}")
    
    # Fall back to tifffile for TIFF files
    if ext in ['.tif', '.tiff', '.gtiff']:
        try:
            return read_image_tifffile(filepath)
        except Exception as e:
            print(f"tifffile read failed: {e}")
    
    # Try imageio as last resort
    try:
        import imageio.v3 as iio
        data = iio.imread(filepath)
        
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3 and data.shape[2] <= 4:
            data = np.transpose(data, (2, 0, 1))
        
        n_bands = data.shape[0]
        band_names = [f"Band_{i+1}" for i in range(n_bands)]
        metadata = {'geotransform': None, 'projection': None, 'nodata': None}
        
        return data.astype(np.float64), None, band_names, metadata
    except Exception as e:
        raise ValueError(f"Could not read image file: {filepath}. Error: {e}")


def calculate_spatial_lag(data: np.ndarray, valid_mask: np.ndarray, window_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the spatial lag (mean of neighbors) for each pixel using vectorized operations.
    Uses Queen contiguity (all neighbors within window, excluding center).
    
    Args:
        data: 2D numpy array
        valid_mask: 2D boolean array indicating valid pixels
        window_size: Size of the neighborhood window (must be odd, default 3 for 8-connected)
        
    Returns:
        Tuple of (spatial lag array, neighbor count array)
    """
    from scipy.ndimage import convolve
    
    # Validate window size
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1  # Make odd
    
    # Create a copy with invalid values set to 0 for summation
    data_masked = np.where(valid_mask, data, 0.0)
    
    # Create kernel: all 1s except center is 0
    kernel = np.ones((window_size, window_size), dtype=np.float64)
    center = window_size // 2
    kernel[center, center] = 0
    
    # Sum of neighbor values
    neighbor_sum = convolve(data_masked, kernel, mode='constant', cval=0.0)
    
    # Count of valid neighbors
    neighbor_count = convolve(valid_mask.astype(np.float64), kernel, mode='constant', cval=0.0)
    
    # Calculate mean (spatial lag) - avoid division by zero
    spatial_lag = np.where(neighbor_count > 0, neighbor_sum / neighbor_count, np.nan)
    
    return spatial_lag, neighbor_count


def calculate_local_morans_i(data: np.ndarray, nodata: Optional[float] = None, window_size: int = 3) -> np.ndarray:
    """
    Calculate Local Moran's I for a single band using a queen contiguity weight matrix.
    Uses vectorized operations for efficiency.
    
    Args:
        data: 2D numpy array of band data
        nodata: NoData value to exclude from calculations
        window_size: Size of the neighborhood window (must be odd, default 3)
        
    Returns:
        2D numpy array of Local Moran's I values
    """
    from scipy.ndimage import convolve
    
    # Create mask for valid data
    if nodata is not None:
        valid_mask = ~np.isclose(data, nodata) & ~np.isnan(data)
    else:
        valid_mask = ~np.isnan(data)
    
    # Calculate global mean and variance of valid data
    valid_data = data[valid_mask]
    if len(valid_data) == 0:
        return np.full(data.shape, np.nan, dtype=np.float32)
    
    global_mean = np.mean(valid_data)
    global_var = np.var(valid_data)
    
    if global_var == 0:
        return np.full(data.shape, np.nan, dtype=np.float32)
    
    # Calculate deviation from mean
    z = data - global_mean
    z_masked = np.where(valid_mask, z, 0.0)
    
    # Calculate spatial lag of z
    spatial_lag, neighbor_count = calculate_spatial_lag(z_masked, valid_mask, window_size)
    
    # Local Moran's I = (zi / variance) * spatial_lag(z)
    # With row-standardized weights, spatial_lag is already the mean of neighbors
    result = (z / global_var) * spatial_lag
    
    # Set invalid pixels to NaN
    result = np.where(valid_mask & (neighbor_count > 0), result, np.nan)
    
    return result.astype(np.float32)


def calculate_bivariate_morans_i(data1: np.ndarray, data2: np.ndarray, 
                                  nodata: Optional[float] = None, window_size: int = 3) -> np.ndarray:
    """
    Calculate Bivariate Local Moran's I between two bands.
    Uses vectorized operations for efficiency.
    
    This measures the spatial correlation between a variable at location i
    and the spatial lag of a different variable at neighboring locations.
    
    Args:
        data1: 2D numpy array of first band data
        data2: 2D numpy array of second band data
        nodata: NoData value to exclude from calculations
        window_size: Size of the neighborhood window (must be odd, default 3)
        
    Returns:
        2D numpy array of Bivariate Local Moran's I values
    """
    # Create mask for valid data (must be valid in both bands)
    if nodata is not None:
        valid_mask = (~np.isclose(data1, nodata) & ~np.isclose(data2, nodata) & 
                      ~np.isnan(data1) & ~np.isnan(data2))
    else:
        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
    
    # Calculate statistics for valid data
    valid_data1 = data1[valid_mask]
    valid_data2 = data2[valid_mask]
    
    if len(valid_data1) == 0:
        return np.full(data1.shape, np.nan, dtype=np.float32)
    
    mean1 = np.mean(valid_data1)
    mean2 = np.mean(valid_data2)
    std1 = np.std(valid_data1)
    std2 = np.std(valid_data2)
    
    if std1 == 0 or std2 == 0:
        return np.full(data1.shape, np.nan, dtype=np.float32)
    
    # Standardize the data
    z1 = (data1 - mean1) / std1
    z2 = (data2 - mean2) / std2
    
    # Mask for calculations
    z2_masked = np.where(valid_mask, z2, 0.0)
    
    # Calculate spatial lag of z2
    spatial_lag_z2, neighbor_count = calculate_spatial_lag(z2_masked, valid_mask, window_size)
    
    # Bivariate Local Moran's I = z1i * spatial_lag(z2)
    result = z1 * spatial_lag_z2
    
    # Set invalid pixels to NaN
    result = np.where(valid_mask & (neighbor_count > 0), result, np.nan)
    
    return result.astype(np.float32)


def format_map_info(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Format map information as ENVI map info string.
    
    Args:
        metadata: Dictionary containing geotransform and projection info
        
    Returns:
        ENVI-formatted map info string or None
    """
    # Check if we already have a map_info string
    if metadata.get('map_info'):
        map_info = metadata['map_info']
        if isinstance(map_info, str):
            # Clean up and return
            if not map_info.startswith('{'):
                map_info = '{ ' + map_info + ' }'
            return map_info
    
    gt = metadata.get('geotransform')
    
    if gt is None or gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        return None
    
    # Extract geotransform components
    x_origin = gt[0]
    pixel_width = gt[1]
    y_origin = gt[3]
    pixel_height = abs(gt[5])  # Make positive
    
    # Try to determine projection name
    proj_name = "Arbitrary"
    zone_info = ""
    datum_info = ""
    units_info = ""
    
    proj = metadata.get('projection')
    if proj and GDAL_AVAILABLE:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        
        if srs.IsGeographic():
            proj_name = "Geographic Lat/Lon"
            units_info = ", units=Degrees"
        elif srs.IsProjected():
            proj_cs_name = srs.GetAttrValue("PROJCS", 0) or ""
            if "UTM" in proj_cs_name.upper():
                proj_name = "UTM"
                zone = abs(srs.GetUTMZone())
                hemisphere = "North" if srs.GetUTMZone() > 0 else "South"
                zone_info = f", {zone}, {hemisphere}"
            else:
                proj_name = proj_cs_name if proj_cs_name else "Projected"
            
            units = srs.GetAttrValue("UNIT", 0)
            if units:
                units_info = f", units={units}"
        
        datum = srs.GetAttrValue("DATUM", 0)
        if datum:
            datum = datum.replace("D_", "").replace("_", " ")
            datum_info = f", {datum}"
    
    # ENVI map info format
    map_info = (f"{{ {proj_name}, 1.0000, 1.0000, {x_origin:.6f}, {y_origin:.6f}, "
                f"{pixel_width:.6f}, {pixel_height:.6f}{zone_info}{datum_info}{units_info} }}")
    
    return map_info


def get_coordinate_system_string(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Extract coordinate system string from metadata.
    
    Args:
        metadata: Dictionary containing projection info
        
    Returns:
        WKT coordinate system string or None
    """
    # Check if we already have a coordinate system string
    if metadata.get('coordinate_system_string'):
        cs = metadata['coordinate_system_string']
        if isinstance(cs, str):
            return cs.replace('\n', ' ')
    
    proj = metadata.get('projection')
    if proj:
        return proj.replace('\n', ' ')
    return None


def write_envi_output(output_path: str, data: np.ndarray, band_names: List[str],
                      metadata: Dict[str, Any]):
    """
    Write output in ENVI format with proper header.
    
    Args:
        output_path: Path for output files (without extension)
        data: 3D numpy array (bands, rows, cols)
        band_names: List of band name descriptions
        metadata: Dictionary containing geospatial information
    """
    n_bands, rows, cols = data.shape
    
    # Ensure data is float32
    data = data.astype(np.float32)
    
    # Write binary file (BSQ interleave)
    bin_path = output_path + '.bin'
    data.tofile(bin_path)
    
    # Create header file
    hdr_path = output_path + '.hdr'
    
    # Get map information
    map_info = format_map_info(metadata)
    coord_sys = get_coordinate_system_string(metadata)
    
    # Format band names for ENVI header
    # ENVI expects band names in curly braces, comma-separated
    band_names_str = "{ " + ",\n ".join(band_names) + " }"
    
    with open(hdr_path, 'w') as f:
        f.write("ENVI\n")
        f.write(f"description = {{ Spatial Autocorrelation Results }}\n")
        f.write(f"samples = {cols}\n")
        f.write(f"lines = {rows}\n")
        f.write(f"bands = {n_bands}\n")
        f.write("header offset = 0\n")
        f.write("file type = ENVI Standard\n")
        f.write("data type = 4\n")  # 4 = 32-bit float (IEEE standard)
        f.write("interleave = bsq\n")
        f.write("byte order = 0\n")  # Little-endian (Intel)
        f.write(f"band names = {band_names_str}\n")
        
        if map_info:
            f.write(f"map info = {map_info}\n")
        
        if coord_sys:
            f.write(f"coordinate system string = {{{coord_sys}}}\n")
    
    print(f"Output written to:")
    print(f"  Binary: {bin_path}")
    print(f"  Header: {hdr_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate spatial autocorrelation for multi-band images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python spatial_autocorrelation.py input.tif output
    python spatial_autocorrelation.py input.img output --nodata -9999
    python spatial_autocorrelation.py input.tif output --window 5
    
Output:
    Creates output.bin (binary data) and output.hdr (ENVI header)
    
    Output bands include:
    - Local Moran's I for each input band
    - Bivariate Moran's I for all pairwise band combinations
    
    Band names in the header describe which input bands were used.
        """
    )
    parser.add_argument('input', help='Input multi-band image file')
    parser.add_argument('output', help='Output file path (extension will be replaced with .bin/.hdr)')
    parser.add_argument('--nodata', type=float, default=None,
                        help='NoData value to exclude from calculations')
    parser.add_argument('--window', '-w', type=int, default=3,
                        help='Window size for neighborhood (must be odd, default=3 for 8-connected neighbors)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Clean up output path - remove any existing extension
    output_base = args.output
    # Common raster extensions to strip
    extensions_to_strip = ['.bin', '.hdr', '.tif', '.tiff', '.img', '.dat', '.bsq', '.bil', '.bip']
    for ext in extensions_to_strip:
        if output_base.lower().endswith(ext):
            output_base = output_base[:-len(ext)]
            break
    
    # Validate and adjust window size
    window_size = args.window
    if window_size < 3:
        print(f"Warning: Window size {window_size} too small, using 3")
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
        print(f"Warning: Window size must be odd, using {window_size}")
    
    print(f"Reading input image: {args.input}")
    if GDAL_AVAILABLE:
        print("Using GDAL for image reading")
    else:
        print("GDAL not available, using tifffile/imageio")
    
    # Read input image
    image_data, ds, input_band_names, metadata = read_image(args.input)
    
    n_bands = image_data.shape[0]
    print(f"Found {n_bands} bands:")
    for i, name in enumerate(input_band_names):
        print(f"  {i+1}: {name}")
    
    # Get nodata value from metadata if not specified
    nodata = args.nodata
    if nodata is None:
        nodata = metadata.get('nodata')
        if nodata is not None:
            print(f"Using NoData value from input: {nodata}")
    
    print(f"Window size: {window_size}x{window_size} ({window_size*window_size - 1} neighbors)")
    
    # Calculate autocorrelations
    output_bands = []
    output_band_names = []
    
    # 1. Single-band autocorrelations
    print("\nCalculating single-band spatial autocorrelations...")
    for i in range(n_bands):
        print(f"  Processing band {i+1}/{n_bands}: {input_band_names[i]}")
        result = calculate_local_morans_i(image_data[i], nodata, window_size)
        output_bands.append(result)
        output_band_names.append(f"LocalMoransI_{input_band_names[i]}")
    
    # 2. Pairwise cross-correlations
    if n_bands > 1:
        print("\nCalculating pairwise bivariate spatial autocorrelations...")
        pairs = list(combinations(range(n_bands), 2))
        for idx, (i, j) in enumerate(pairs):
            print(f"  Processing pair {idx+1}/{len(pairs)}: {input_band_names[i]} x {input_band_names[j]}")
            result = calculate_bivariate_morans_i(image_data[i], image_data[j], nodata, window_size)
            output_bands.append(result)
            output_band_names.append(f"BivariateMoransI_{input_band_names[i]}_x_{input_band_names[j]}")
    
    # Stack all output bands
    output_data = np.array(output_bands)
    
    print(f"\nTotal output bands: {len(output_band_names)}")
    print("Band descriptions:")
    for i, name in enumerate(output_band_names):
        print(f"  {i+1}: {name}")
    
    # Write output
    print(f"\nWriting output...")
    write_envi_output(output_base, output_data, output_band_names, metadata)
    
    # Clean up
    ds = None
    
    print("\nDone!")


if __name__ == "__main__":
    main()

