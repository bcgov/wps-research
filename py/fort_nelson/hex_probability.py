
"""
Standalone script: Aggregate classification probabilities on hexagonal grid
Uses GDAL directly - no QGIS layer dependencies for reading rasters

Finds all *_classification.bin files in current directory and aggregates
probabilities onto a hexagonal grid within the AOI.

Usage: python3 hex_probability_standalone.py
"""

import os
import glob
import math
import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

# ============ PARAMETERS ============
HEX_SPACING = 500  # meters
AOI_SHP = "/data/fort_nelson/circle_100km.shp"
OUTPUT_SHP = "/data/fort_nelson/hex_probability_grid.shp"
CLASSIFICATION_PATTERN = "*_classification.bin"
# ====================================

def get_aoi_geometry_and_crs(shp_path):
    """Load AOI polygon and its CRS"""
    ds = ogr.Open(shp_path, 0)
    if ds is None:
        raise RuntimeError(f"Could not open AOI shapefile: {shp_path}")
    
    layer = ds.GetLayer(0)
    srs = layer.GetSpatialRef()
    
    feat = layer.GetNextFeature()
    geom = feat.GetGeometryRef().Clone()
    
    extent = layer.GetExtent()  # (minX, maxX, minY, maxY)
    
    return geom, srs, extent

def find_classification_files(pattern):
    """Find all classification files matching pattern"""
    files = glob.glob(pattern)
    print(f"Found {len(files)} classification files:")
    for f in files:
        print(f"  - {f}")
    return files

def load_classification_raster(path):
    """Load a classification raster and return dataset, geotransform, and CRS"""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"  WARNING: Could not open {path}")
        return None, None, None
    
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    
    print(f"  Loaded: {os.path.basename(path)}")
    print(f"    Size: {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"    Origin: ({gt[0]:.2f}, {gt[3]:.2f})")
    print(f"    Pixel size: ({gt[1]:.2f}, {gt[5]:.2f})")
    
    return ds, gt, srs

def generate_hex_centers(extent, spacing):
    """Generate hexagon center points covering the extent"""
    minX, maxX, minY, maxY = extent
    
    h_spacing = spacing
    v_spacing = spacing * math.sqrt(3) / 2
    
    # Add buffer
    minX -= spacing
    minY -= spacing
    maxX += spacing
    maxY += spacing
    
    centers = []
    row = 0
    y = minY
    while y <= maxY:
        x_offset = (h_spacing / 2) if (row % 2 == 1) else 0
        x = minX + x_offset
        while x <= maxX:
            centers.append((x, y))
            x += h_spacing
        y += v_spacing
        row += 1
    
    return centers

def create_hexagon_wkt(cx, cy, spacing):
    """Create a hexagon polygon WKT around center point"""
    r = spacing / math.sqrt(3)
    
    points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        points.append(f"{px} {py}")
    points.append(points[0])  # Close
    
    return f"POLYGON(({', '.join(points)}))"

def point_in_polygon(px, py, geom):
    """Check if point is inside polygon geometry"""
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(px, py)
    return geom.Contains(pt)

def sample_raster_at_point(ds, gt, px, py):
    """Sample raster value at geographic point. Returns None for nodata/NaN/out of bounds."""
    # Convert geographic to pixel coordinates
    inv_gt = gdal.InvGeoTransform(gt)
    if inv_gt is None:
        return None
    
    col = int(inv_gt[0] + inv_gt[1] * px + inv_gt[2] * py)
    row = int(inv_gt[3] + inv_gt[4] * px + inv_gt[5] * py)
    
    if col < 0 or row < 0 or col >= ds.RasterXSize or row >= ds.RasterYSize:
        return None
    
    band = ds.GetRasterBand(1)
    val = band.ReadAsArray(col, row, 1, 1)
    
    if val is None:
        return None
    
    val = float(val[0, 0])
    
    if np.isnan(val):
        return None
    
    return val

def get_sample_points(cx, cy, spacing):
    """Get sample points within a hexagon (center + two rings)"""
    points = [(cx, cy)]
    
    # Inner ring
    r_inner = spacing / (3 * math.sqrt(3))
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        points.append((cx + r_inner * math.cos(angle), cy + r_inner * math.sin(angle)))
    
    # Outer ring
    r_outer = 2 * spacing / (3 * math.sqrt(3))
    for i in range(6):
        angle = i * math.pi / 3
        points.append((cx + r_outer * math.cos(angle), cy + r_outer * math.sin(angle)))
    
    return points

def create_coordinate_transform(src_srs, dst_srs):
    """Create coordinate transform between two SRS"""
    if src_srs is None or dst_srs is None:
        return None
    if src_srs.IsSame(dst_srs):
        return None
    return osr.CoordinateTransformation(src_srs, dst_srs)

def transform_point(x, y, transform):
    """Transform a point, return (x, y)"""
    if transform is None:
        return x, y
    pt = transform.TransformPoint(x, y)
    return pt[0], pt[1]

# ============ MAIN ============

print("=" * 70)
print("Hexagonal Probability Aggregation (Standalone GDAL)")
print("=" * 70)

# Load AOI
print(f"\nLoading AOI: {AOI_SHP}")
if not os.path.exists(AOI_SHP):
    print(f"ERROR: AOI file not found: {AOI_SHP}")
    exit(1)

aoi_geom, aoi_srs, aoi_extent = get_aoi_geometry_and_crs(AOI_SHP)
print(f"  CRS: {aoi_srs.GetAttrValue('PROJCS') or aoi_srs.GetAttrValue('GEOGCS')}")
print(f"  Extent: {aoi_extent}")

# Find classification files
print(f"\nSearching for: {CLASSIFICATION_PATTERN}")
class_files = find_classification_files(CLASSIFICATION_PATTERN)

if len(class_files) == 0:
    print("ERROR: No classification files found!")
    exit(1)

# Load all classification rasters
print("\nLoading classification rasters...")
rasters = []  # List of (ds, gt, srs, transform_to_aoi)

for f in class_files:
    ds, gt, srs = load_classification_raster(f)
    if ds is not None:
        # Create transform from raster CRS to AOI CRS
        transform = create_coordinate_transform(aoi_srs, srs)  # AOI coords -> raster coords
        rasters.append({
            'path': f,
            'ds': ds,
            'gt': gt,
            'srs': srs,
            'transform': transform,  # Transform AOI points to raster CRS
            'extent': (gt[0], gt[0] + gt[1] * ds.RasterXSize,
                      gt[3] + gt[5] * ds.RasterYSize, gt[3])  # minX, maxX, minY, maxY
        })

print(f"\nLoaded {len(rasters)} rasters successfully")

# Generate hex grid in AOI CRS
print(f"\nGenerating hex grid with {HEX_SPACING}m spacing...")
hex_centers = generate_hex_centers(aoi_extent, HEX_SPACING)
print(f"  Generated {len(hex_centers)} hex centers")

# Filter to AOI
print("  Filtering to AOI polygon...")
hex_centers = [(x, y) for x, y in hex_centers if point_in_polygon(x, y, aoi_geom)]
print(f"  {len(hex_centers)} centers within AOI")

# Create output shapefile
print(f"\nCreating output shapefile: {OUTPUT_SHP}")
driver = ogr.GetDriverByName("ESRI Shapefile")

if os.path.exists(OUTPUT_SHP):
    driver.DeleteDataSource(OUTPUT_SHP)

out_ds = driver.CreateDataSource(OUTPUT_SHP)
out_layer = out_ds.CreateLayer("hex_probabilities", aoi_srs, ogr.wkbPolygon)

out_layer.CreateField(ogr.FieldDefn("prob_mean", ogr.OFTReal))
out_layer.CreateField(ogr.FieldDefn("n_samples", ogr.OFTInteger))
out_layer.CreateField(ogr.FieldDefn("n_rasters", ogr.OFTInteger))

src_field = ogr.FieldDefn("src_image", ogr.OFTString)
src_field.SetWidth(200)
out_layer.CreateField(src_field)

# Process hexagons
print("\nProcessing hexagons...")
n_with_data = 0
n_nodata = 0
n_partial = 0

for i, (cx, cy) in enumerate(hex_centers):
    if (i + 1) % 1000 == 0:
        print(f"  Processing {i+1}/{len(hex_centers)}... ({n_with_data} with data)")
    
    # Get sample points in AOI CRS
    sample_points_aoi = get_sample_points(cx, cy, HEX_SPACING)
    
    # Collect values from all rasters
    all_values = []
    sources = set()
    
    for raster in rasters:
        ds = raster['ds']
        gt = raster['gt']
        transform = raster['transform']
        
        for px, py in sample_points_aoi:
            # Transform point from AOI CRS to raster CRS
            rx, ry = transform_point(px, py, transform)
            
            val = sample_raster_at_point(ds, gt, rx, ry)
            if val is not None:
                all_values.append(val)
                sources.add(os.path.basename(raster['path']))
    
    # Create feature
    feat = ogr.Feature(out_layer.GetLayerDefn())
    hex_wkt = create_hexagon_wkt(cx, cy, HEX_SPACING)
    feat.SetGeometry(ogr.CreateGeometryFromWkt(hex_wkt))
    
    if len(all_values) == 0:
        # All nodata
        feat.SetField("prob_mean", None)  # NULL
        feat.SetField("n_samples", 0)
        feat.SetField("n_rasters", 0)
        feat.SetField("src_image", "NODATA")
        n_nodata += 1
    else:
        mean_val = sum(all_values) / len(all_values)
        feat.SetField("prob_mean", mean_val)
        feat.SetField("n_samples", len(all_values))
        feat.SetField("n_rasters", len(sources))
        feat.SetField("src_image", "; ".join(sorted(sources)[:3]))  # First 3 sources
        n_with_data += 1
        
        if len(all_values) < 13:
            n_partial += 1
    
    out_layer.CreateFeature(feat)

# Cleanup
out_ds = None

print(f"\n--- Summary ---")
print(f"Total hexagons: {len(hex_centers)}")
print(f"With valid data: {n_with_data}")
print(f"  Partial coverage: {n_partial}")
print(f"Nodata: {n_nodata}")
print(f"\nOutput saved to: {OUTPUT_SHP}")
print("=" * 70)
print("Done!")
print("=" * 70)

# Debug: print first raster's actual data range
if len(rasters) > 0:
    print("\n--- Debug: First raster stats ---")
    r = rasters[0]
    band = r['ds'].GetRasterBand(1)
    stats = band.GetStatistics(True, True)
    print(f"  File: {os.path.basename(r['path'])}")
    print(f"  Min: {stats[0]:.4f}, Max: {stats[1]:.4f}")
    print(f"  Mean: {stats[2]:.4f}, StdDev: {stats[3]:.4f}")
    
    # Sample center of image
    cx = r['ds'].RasterXSize // 2
    cy = r['ds'].RasterYSize // 2
    val = band.ReadAsArray(cx, cy, 1, 1)
    print(f"  Center pixel [{cx},{cy}]: {val[0,0]}")
    
    # Print raster extent in its native CRS
    gt = r['gt']
    print(f"  Raster extent (native CRS):")
    print(f"    X: {r['extent'][0]:.2f} to {r['extent'][1]:.2f}")
    print(f"    Y: {r['extent'][2]:.2f} to {r['extent'][3]:.2f}")
    print(f"  AOI extent:")
    print(f"    X: {aoi_extent[0]:.2f} to {aoi_extent[1]:.2f}")
    print(f"    Y: {aoi_extent[2]:.2f} to {aoi_extent[3]:.2f}")



