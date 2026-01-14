#!/usr/bin/env python3
"""
20260114 Spatial Autocorrelation Calculator

Calculates spatial autocorrelation (Moran's I) for:
1. Each band individually (spatial autocorrelation within a band)
2. All pairwise combinations of bands (cross-correlation between bands)

Output is written in 32-bit float ENVI format (.bin and .hdr files).
"""

import numpy as np
from osgeo import gdal
import sys
import os
from itertools import combinations
from typing import List, Tuple, Optional
import argparse


def read_image(filepath: str) -> Tuple[np.ndarray, gdal.Dataset]:
    """
    Read a multi-band image using GDAL.
    
    Args:
        filepath: Path to the input image
        
    Returns:
        Tuple of (image data as numpy array, GDAL dataset)
    """
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f"Could not open file: {filepath}")
    
    # Read all bands into a 3D array (bands, rows, cols)
    bands = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        data = band.ReadAsArray().astype(np.float64)
        bands.append(data)
    
    return np.array(bands), ds


def get_band_names(ds: gdal.Dataset) -> List[str]:
    """
    Extract band names from GDAL dataset.
    
    Args:
        ds: GDAL dataset
        
    Returns:
        List of band names
    """
    band_names = []
    for i in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(i)
        name = band.GetDescription()
        if not name:
            name = f"Band_{i}"
        band_names.append(name)
    return band_names


def calculate_local_morans_i(data: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """
    Calculate Local Moran's I for a single band using a queen contiguity weight matrix.
    
    Args:
        data: 2D numpy array of band data
        nodata: NoData value to exclude from calculations
        
    Returns:
        2D numpy array of Local Moran's I values
    """
    rows, cols = data.shape
    result = np.full((rows, cols), np.nan, dtype=np.float32)
    
    # Create mask for valid data
    if nodata is not None:
        valid_mask = ~np.isclose(data, nodata)
    else:
        valid_mask = ~np.isnan(data)
    
    # Calculate global mean and standard deviation of valid data
    valid_data = data[valid_mask]
    if len(valid_data) == 0:
        return result
    
    global_mean = np.mean(valid_data)
    global_var = np.var(valid_data)
    
    if global_var == 0:
        return result
    
    # Standardize the data
    z = (data - global_mean)
    
    # Queen contiguity offsets (8-connected neighbors)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1),  (1, 0), (1, 1)]
    
    # Calculate Local Moran's I for each pixel
    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r, c]:
                continue
            
            zi = z[r, c]
            neighbor_sum = 0.0
            neighbor_count = 0
            
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and valid_mask[nr, nc]:
                    neighbor_sum += z[nr, nc]
                    neighbor_count += 1
            
            if neighbor_count > 0:
                # Local Moran's I = (zi / variance) * sum(wij * zj)
                # With row-standardized weights (wij = 1/n for each neighbor)
                avg_neighbor_z = neighbor_sum / neighbor_count
                result[r, c] = (zi / global_var) * avg_neighbor_z
    
    return result


def calculate_bivariate_morans_i(data1: np.ndarray, data2: np.ndarray, 
                                  nodata: Optional[float] = None) -> np.ndarray:
    """
    Calculate Bivariate Local Moran's I between two bands.
    
    This measures the spatial correlation between a variable at location i
    and the spatial lag of a different variable at neighboring locations.
    
    Args:
        data1: 2D numpy array of first band data
        data2: 2D numpy array of second band data
        nodata: NoData value to exclude from calculations
        
    Returns:
        2D numpy array of Bivariate Local Moran's I values
    """
    rows, cols = data1.shape
    result = np.full((rows, cols), np.nan, dtype=np.float32)
    
    # Create mask for valid data (must be valid in both bands)
    if nodata is not None:
        valid_mask = ~np.isclose(data1, nodata) & ~np.isclose(data2, nodata)
    else:
        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
    
    # Calculate statistics for valid data
    valid_data1 = data1[valid_mask]
    valid_data2 = data2[valid_mask]
    
    if len(valid_data1) == 0:
        return result
    
    mean1 = np.mean(valid_data1)
    mean2 = np.mean(valid_data2)
    std1 = np.std(valid_data1)
    std2 = np.std(valid_data2)
    
    if std1 == 0 or std2 == 0:
        return result
    
    # Standardize the data
    z1 = (data1 - mean1) / std1
    z2 = (data2 - mean2) / std2
    
    # Queen contiguity offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1),  (1, 0), (1, 1)]
    
    # Calculate Bivariate Local Moran's I
    for r in range(rows):
        for c in range(cols):
            if not valid_mask[r, c]:
                continue
            
            z1i = z1[r, c]
            neighbor_sum = 0.0
            neighbor_count = 0
            
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and valid_mask[nr, nc]:
                    neighbor_sum += z2[nr, nc]
                    neighbor_count += 1
            
            if neighbor_count > 0:
                # Bivariate Local Moran's I = zi * sum(wij * zj_other)
                avg_neighbor_z2 = neighbor_sum / neighbor_count
                result[r, c] = z1i * avg_neighbor_z2
    
    return result


def get_map_info_string(ds: gdal.Dataset) -> Optional[str]:
    """
    Extract map information from GDAL dataset and format as ENVI map info string.
    
    Args:
        ds: GDAL dataset
        
    Returns:
        ENVI-formatted map info string or None
    """
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    
    if gt is None or gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        return None
    
    # Extract geotransform components
    x_origin = gt[0]
    pixel_width = gt[1]
    y_origin = gt[3]
    pixel_height = abs(gt[5])  # Make positive
    
    # Try to get projection name
    proj_name = "Arbitrary"
    if proj:
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        
        if srs.IsGeographic():
            proj_name = "Geographic Lat/Lon"
        elif srs.IsProjected():
            proj_name = srs.GetAttrValue("PROJCS", 0) or "UTM"
            if "UTM" in proj_name.upper():
                zone = srs.GetUTMZone()
                if zone != 0:
                    hemisphere = "North" if zone > 0 else "South"
                    proj_name = f"UTM"
    
    # ENVI map info format:
    # {projection name, reference pixel x, reference pixel y, 
    #  reference easting, reference northing, pixel size x, pixel size y,
    #  [projection zone], [North or South], [datum], [units]}
    
    map_info = f"{{  {proj_name}, 1.0000, 1.0000, {x_origin:.6f}, {y_origin:.6f}, {pixel_width:.6f}, {pixel_height:.6f}"
    
    if proj:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        if srs.IsProjected() and "UTM" in (srs.GetAttrValue("PROJCS", 0) or "").upper():
            zone = abs(srs.GetUTMZone())
            hemisphere = "North" if srs.GetUTMZone() > 0 else "South"
            map_info += f", {zone}, {hemisphere}"
        
        datum = srs.GetAttrValue("DATUM", 0)
        if datum:
            # Clean up datum name
            datum = datum.replace("D_", "").replace("_", " ")
            map_info += f", {datum}"
        
        units = srs.GetAttrValue("UNIT", 0)
        if units:
            map_info += f", units={units}"
    
    map_info += " }"
    
    return map_info


def get_coordinate_system_string(ds: gdal.Dataset) -> Optional[str]:
    """
    Extract coordinate system string from GDAL dataset.
    
    Args:
        ds: GDAL dataset
        
    Returns:
        WKT coordinate system string or None
    """
    proj = ds.GetProjection()
    if proj:
        return proj.replace('\n', ' ')
    return None


def write_envi_output(output_path: str, data: np.ndarray, band_names: List[str],
                      ds_template: gdal.Dataset):
    """
    Write output in ENVI format with proper header.
    
    Args:
        output_path: Path for output files (without extension)
        data: 3D numpy array (bands, rows, cols)
        band_names: List of band name descriptions
        ds_template: GDAL dataset to copy geospatial info from
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
    map_info = get_map_info_string(ds_template)
    coord_sys = get_coordinate_system_string(ds_template)
    
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
        """
    )
    parser.add_argument('input', help='Input multi-band image file')
    parser.add_argument('output', help='Output file path (without extension)')
    parser.add_argument('--nodata', type=float, default=None,
                        help='NoData value to exclude from calculations')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print(f"Reading input image: {args.input}")
    
    # Read input image
    image_data, ds = read_image(args.input)
    input_band_names = get_band_names(ds)
    
    n_bands = image_data.shape[0]
    print(f"Found {n_bands} bands:")
    for i, name in enumerate(input_band_names):
        print(f"  {i+1}: {name}")
    
    # Get nodata value from dataset if not specified
    nodata = args.nodata
    if nodata is None:
        band1 = ds.GetRasterBand(1)
        nodata = band1.GetNoDataValue()
        if nodata is not None:
            print(f"Using NoData value from input: {nodata}")
    
    # Calculate autocorrelations
    output_bands = []
    output_band_names = []
    
    # 1. Single-band autocorrelations
    print("\nCalculating single-band spatial autocorrelations...")
    for i in range(n_bands):
        print(f"  Processing band {i+1}/{n_bands}: {input_band_names[i]}")
        result = calculate_local_morans_i(image_data[i], nodata)
        output_bands.append(result)
        output_band_names.append(f"LocalMoransI_{input_band_names[i]}")
    
    # 2. Pairwise cross-correlations
    if n_bands > 1:
        print("\nCalculating pairwise bivariate spatial autocorrelations...")
        pairs = list(combinations(range(n_bands), 2))
        for idx, (i, j) in enumerate(pairs):
            print(f"  Processing pair {idx+1}/{len(pairs)}: {input_band_names[i]} x {input_band_names[j]}")
            result = calculate_bivariate_morans_i(image_data[i], image_data[j], nodata)
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
    write_envi_output(args.output, output_data, output_band_names, ds)
    
    # Clean up
    ds = None
    
    print("\nDone!")


if __name__ == "__main__":
    main()


