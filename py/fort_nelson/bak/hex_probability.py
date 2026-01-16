"""
Standalone script: Aggregate classification probabilities on hexagonal grid
Uses GDAL directly - no QGIS layer dependencies for reading rasters

Finds all *_classification.bin files in current directory and aggregates
probabilities onto a hexagonal grid within the AOI.

For overlapping coverage, uses the LATEST year (extracted from filename).

Usage: python3 hex_probability_standalone.py
"""

import os
import glob
import math
import re
import time
import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

# ============ PARAMETERS ============
HEX_SPACING = 500  # meters
AOI_SHP = "/data/fort_nelson/circle_100km.shp"
OUTPUT_SHP = "/data/fort_nelson/hex_probability_grid.shp"
CLASSIFICATION_PATTERN = "*_classification.bin"
# ====================================

def extract_year_from_filename(filename):
    """
    Extract year from filename. Returns year as int, or 0 if not found.
    
    Patterns handled:
    - _2025_ -> 2025
    - _202508_ -> 2025 (YYYYMM format, treated as that year)
    - _20250815_ -> 2025 (YYYYMMDD format)
    """
    basename = os.path.basename(filename)
    
    # Try YYYYMM or YYYYMMDD pattern first (more specific)
    match = re.search(r'_(\d{4})(\d{2,4})_', basename)
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return year
    
    # Try simple _YYYY_ pattern
    match = re.search(r'_(\d{4})_', basename)
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return year
    
    # Try YYYY anywhere in filename
    match = re.search(r'(\d{4})', basename)
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2100:
            return year
    
    return 0

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
    """Find all classification files matching pattern, sorted by year (newest first)"""
    files = glob.glob(pattern)
    
    # Sort by year (newest first)
    files_with_year = [(f, extract_year_from_filename(f)) for f in files]
    files_with_year.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(files)} classification files (sorted by year, newest first):")
    current_year = None
    for f, year in files_with_year:
        if year != current_year:
            current_year = year
            print(f"\n  Year {year}:")
        print(f"    {os.path.basename(f)}")
    
    return files_with_year  # Return tuples of (filename, year)

def load_classification_raster(path):
    """Load a classification raster and return dataset, geotransform, and CRS"""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"  WARNING: Could not open {path}")
        return None, None, None
    
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    
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

def print_progress(current, total, start_time, n_with_data, n_nodata):
    """Print progress with percentage and ETA"""
    pct = 100.0 * current / total
    elapsed = time.time() - start_time
    
    if current > 0 and elapsed > 0:
        rate = current / elapsed
        remaining = (total - current) / rate
        eta_min = int(remaining // 60)
        eta_sec = int(remaining % 60)
        
        print(f"\r  Processing: {current}/{total} ({pct:5.1f}%) | "
              f"Data: {n_with_data} | NoData: {n_nodata} | "
              f"ETA: {eta_min}m {eta_sec:02d}s   ", end="", flush=True)
    else:
        print(f"\r  Processing: {current}/{total} ({pct:5.1f}%)   ", end="", flush=True)

# ============ MAIN ============

print("=" * 70)
print("Hexagonal Probability Aggregation (Standalone GDAL)")
print("  - Uses LATEST year for overlapping coverage")
print("=" * 70)

# Load AOI
print(f"\nLoading AOI: {AOI_SHP}")
if not os.path.exists(AOI_SHP):
    print(f"ERROR: AOI file not found: {AOI_SHP}")
    exit(1)

aoi_geom, aoi_srs, aoi_extent = get_aoi_geometry_and_crs(AOI_SHP)
print(f"  CRS: {aoi_srs.GetAttrValue('PROJCS') or aoi_srs.GetAttrValue('GEOGCS')}")
print(f"  Extent: X({aoi_extent[0]:.1f} to {aoi_extent[1]:.1f}), Y({aoi_extent[2]:.1f} to {aoi_extent[3]:.1f})")

# Find classification files (sorted by year, newest first)
print(f"\nSearching for: {CLASSIFICATION_PATTERN}")
class_files_with_years = find_classification_files(CLASSIFICATION_PATTERN)

if len(class_files_with_years) == 0:
    print("ERROR: No classification files found!")
    exit(1)

# Load all classification rasters
print("\n" + "-" * 70)
print("Loading classification rasters...")
rasters = []  # List of dicts, sorted by year (newest first)

for filepath, year in class_files_with_years:
    ds, gt, srs = load_classification_raster(filepath)
    if ds is not None:
        # Create transform from AOI CRS to raster CRS
        transform = create_coordinate_transform(aoi_srs, srs)
        
        rasters.append({
            'path': filepath,
            'year': year,
            'ds': ds,
            'gt': gt,
            'srs': srs,
            'transform': transform,
        })

print(f"\nLoaded {len(rasters)} rasters successfully")

# Show year distribution
years = {}
for r in rasters:
    y = r['year']
    years[y] = years.get(y, 0) + 1
print("Year distribution:", ", ".join([f"{y}: {c} files" for y, c in sorted(years.items(), reverse=True)]))

# Generate hex grid in AOI CRS
print(f"\n" + "-" * 70)
print(f"Generating hex grid with {HEX_SPACING}m spacing...")
hex_centers = generate_hex_centers(aoi_extent, HEX_SPACING)
print(f"  Generated {len(hex_centers)} hex centers")

# Filter to AOI
print("  Filtering to AOI polygon...")
hex_centers = [(x, y) for x, y in hex_centers if point_in_polygon(x, y, aoi_geom)]
print(f"  {len(hex_centers)} centers within AOI")

# Create output shapefile
print(f"\n" + "-" * 70)
print(f"Creating output: {OUTPUT_SHP}")
driver = ogr.GetDriverByName("ESRI Shapefile")

if os.path.exists(OUTPUT_SHP):
    driver.DeleteDataSource(OUTPUT_SHP)

os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)

out_ds = driver.CreateDataSource(OUTPUT_SHP)
out_layer = out_ds.CreateLayer("hex_probabilities", aoi_srs, ogr.wkbPolygon)

out_layer.CreateField(ogr.FieldDefn("prob_mean", ogr.OFTReal))
out_layer.CreateField(ogr.FieldDefn("n_samples", ogr.OFTInteger))
out_layer.CreateField(ogr.FieldDefn("year", ogr.OFTInteger))

src_field = ogr.FieldDefn("src_image", ogr.OFTString)
src_field.SetWidth(200)
out_layer.CreateField(src_field)

# Process hexagons
print("\n" + "-" * 70)
print("Processing hexagons (latest year wins for overlapping coverage)...")

n_with_data = 0
n_nodata = 0
n_partial = 0
start_time = time.time()

for i, (cx, cy) in enumerate(hex_centers):
    if (i + 1) % 100 == 0 or i == len(hex_centers) - 1:
        print_progress(i + 1, len(hex_centers), start_time, n_with_data, n_nodata)
    
    # Get sample points in AOI CRS
    sample_points_aoi = get_sample_points(cx, cy, HEX_SPACING)
    
    # For each sample point, find value from LATEST year that has data
    values = []
    best_year = 0
    best_source = None
    
    for px, py in sample_points_aoi:
        # Try rasters in order (already sorted by year, newest first)
        for raster in rasters:
            ds = raster['ds']
            gt = raster['gt']
            transform = raster['transform']
            
            # Transform point from AOI CRS to raster CRS
            rx, ry = transform_point(px, py, transform)
            
            val = sample_raster_at_point(ds, gt, rx, ry)
            if val is not None:
                values.append(val)
                if raster['year'] > best_year:
                    best_year = raster['year']
                    best_source = os.path.basename(raster['path'])
                break  # Found data for this point, don't check older years
    
    # Create feature
    feat = ogr.Feature(out_layer.GetLayerDefn())
    hex_wkt = create_hexagon_wkt(cx, cy, HEX_SPACING)
    feat.SetGeometry(ogr.CreateGeometryFromWkt(hex_wkt))
    
    if len(values) == 0:
        # All nodata
        feat.SetFieldNull("prob_mean")
        feat.SetField("n_samples", 0)
        feat.SetField("year", 0)
        feat.SetField("src_image", "NODATA")
        n_nodata += 1
    else:
        mean_val = sum(values) / len(values)
        feat.SetField("prob_mean", mean_val)
        feat.SetField("n_samples", len(values))
        feat.SetField("year", best_year)
        feat.SetField("src_image", best_source if best_source else "UNKNOWN")
        n_with_data += 1
        
        if len(values) < 13:
            n_partial += 1
    
    out_layer.CreateFeature(feat)

print()  # Newline after progress

# Cleanup
out_ds = None

elapsed = time.time() - start_time

# Calculate hex area
hex_radius = HEX_SPACING / math.sqrt(3)
hex_area_m2 = (3 * math.sqrt(3) / 2) * hex_radius ** 2
hex_area_km2 = hex_area_m2 / 1e6
hex_area_ha = hex_area_m2 / 1e4

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Total hexagons:     {len(hex_centers)}")
print(f"With valid data:    {n_with_data} ({100*n_with_data/len(hex_centers):.1f}%)")
print(f"  Partial coverage: {n_partial}")
print(f"Nodata:             {n_nodata} ({100*n_nodata/len(hex_centers):.1f}%)")
print(f"Processing time:    {elapsed:.1f} seconds")
print(f"\nOutput saved to: {OUTPUT_SHP}")

# Area statistics
print("\n" + "-" * 70)
print("Area Statistics")
print("-" * 70)
print(f"Hex cell area: {hex_area_km2:.4f} km² ({hex_area_ha:.2f} ha)")

total_data_area_km2 = n_with_data * hex_area_km2
total_data_area_ha = n_with_data * hex_area_ha
print(f"\nTotal area with probability data:")
print(f"  {total_data_area_km2:.2f} km²")
print(f"  {total_data_area_ha:.2f} ha")

# Count hexagons in each probability class and calculate weighted sum
class_counts = [0] * 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
weighted_sum = 0.0

# Re-read the shapefile to get probability values
out_ds_read = ogr.Open(OUTPUT_SHP, 0)
out_layer_read = out_ds_read.GetLayer(0)

for feat in out_layer_read:
    prob = feat.GetField("prob_mean")
    if prob is not None:
        weighted_sum += prob
        # Determine class (0-9)
        class_idx = min(int(prob * 10), 9)
        class_counts[class_idx] += 1

out_ds_read = None

print("\nArea by probability class:")
print(f"  {'Class':<12} {'Hexagons':>10} {'km²':>12} {'ha':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")

for i in range(10):
    lower = i / 10
    upper = (i + 1) / 10
    count = class_counts[i]
    area_km2 = count * hex_area_km2
    area_ha = count * hex_area_ha
    print(f"  {lower:.1f} - {upper:.1f}      {count:>10} {area_km2:>12.2f} {area_ha:>12.2f}")

print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
print(f"  {'TOTAL':<12} {n_with_data:>10} {total_data_area_km2:>12.2f} {total_data_area_ha:>12.2f}")

# Equivalent area at 100% detection
equiv_area_km2 = weighted_sum * hex_area_km2
equiv_area_ha = weighted_sum * hex_area_ha

print("\n" + "-" * 70)
print("Equivalent Area at 100% Detection Probability")
print("-" * 70)
print(f"Sum of (area × probability) across all cells:")
print(f"  {equiv_area_km2:.2f} km²")
print(f"  {equiv_area_ha:.2f} ha")
print(f"\nThis represents the equivalent area if all detections")
print(f"were concentrated at 100% probability.")

# Create QML style file for automatic coloring in QGIS
qml_path = OUTPUT_SHP.replace('.shp', '.qml')
print(f"\nCreating style file: {qml_path}")

qml_content = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.0" styleCategories="Symbology">
  <renderer-v2 type="graduatedSymbol" attr="prob_mean" graduatedMethod="GraduatedColor">
    <ranges>
      <range lower="0.0" upper="0.1" symbol="0" label="0.0 - 0.1"/>
      <range lower="0.1" upper="0.2" symbol="1" label="0.1 - 0.2"/>
      <range lower="0.2" upper="0.3" symbol="2" label="0.2 - 0.3"/>
      <range lower="0.3" upper="0.4" symbol="3" label="0.3 - 0.4"/>
      <range lower="0.4" upper="0.5" symbol="4" label="0.4 - 0.5"/>
      <range lower="0.5" upper="0.6" symbol="5" label="0.5 - 0.6"/>
      <range lower="0.6" upper="0.7" symbol="6" label="0.6 - 0.7"/>
      <range lower="0.7" upper="0.8" symbol="7" label="0.7 - 0.8"/>
      <range lower="0.8" upper="0.9" symbol="8" label="0.8 - 0.9"/>
      <range lower="0.9" upper="1.0" symbol="9" label="0.9 - 1.0"/>
    </ranges>
    <symbols>
      <symbol type="fill" name="0"><layer class="SimpleFill">
        <prop k="color" v="255,255,255,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="1"><layer class="SimpleFill">
        <prop k="color" v="229,240,255,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="2"><layer class="SimpleFill">
        <prop k="color" v="203,225,255,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="3"><layer class="SimpleFill">
        <prop k="color" v="170,200,255,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="4"><layer class="SimpleFill">
        <prop k="color" v="135,170,240,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="5"><layer class="SimpleFill">
        <prop k="color" v="100,140,220,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="6"><layer class="SimpleFill">
        <prop k="color" v="70,110,200,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="7"><layer class="SimpleFill">
        <prop k="color" v="45,80,175,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="8"><layer class="SimpleFill">
        <prop k="color" v="25,55,145,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
      <symbol type="fill" name="9"><layer class="SimpleFill">
        <prop k="color" v="0,30,120,180"/><prop k="outline_color" v="128,128,128,100"/><prop k="outline_width" v="0.1"/>
      </layer></symbol>
    </symbols>
  </renderer-v2>
</qgis>
'''

with open(qml_path, 'w') as f:
    f.write(qml_content)

print("Style file created - QGIS will auto-apply white-to-blue gradient")
print("=" * 70)

# Debug: print first raster's info
if len(rasters) > 0:
    print("\n--- Debug: Sample raster info ---")
    r = rasters[0]
    band = r['ds'].GetRasterBand(1)
    stats = band.GetStatistics(True, True)
    print(f"  File: {os.path.basename(r['path'])}")
    print(f"  Year: {r['year']}")
    print(f"  Size: {r['ds'].RasterXSize} x {r['ds'].RasterYSize}")
    print(f"  Stats - Min: {stats[0]:.4f}, Max: {stats[1]:.4f}, Mean: {stats[2]:.4f}")
    
    gt = r['gt']
    ext_minx = gt[0]
    ext_maxx = gt[0] + gt[1] * r['ds'].RasterXSize
    ext_maxy = gt[3]
    ext_miny = gt[3] + gt[5] * r['ds'].RasterYSize
    print(f"  Raster extent: X({ext_minx:.1f} to {ext_maxx:.1f}), Y({ext_miny:.1f} to {ext_maxy:.1f})")
    print(f"  AOI extent:    X({aoi_extent[0]:.1f} to {aoi_extent[1]:.1f}), Y({aoi_extent[2]:.1f} to {aoi_extent[3]:.1f})")

