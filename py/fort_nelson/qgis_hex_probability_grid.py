"""
QGIS Script: Aggregate classification probabilities on hexagonal grid from top-most images

This script:
1. Creates a hexagonal grid (500m spacing) over the AOI
2. For each location, finds the top-most visible raster
3. Loads the corresponding _classification.bin file
4. Sums probability values within each hexagon (NaN for fully nodata hexes)

Run in QGIS Python console.
"""

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsField, QgsMapLayer, QgsRasterLayer, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsVectorFileWriter, QgsWkbTypes, QgsFillSymbol,
    QgsGraduatedSymbolRenderer, QgsRendererRange
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from osgeo import gdal
import numpy as np
import os
import math

# ============ PARAMETERS ============
HEX_SPACING = 500  # meters
AOI_LAYER_NAME = "circle_100km"  # Name of your AOI vector layer in QGIS
OUTPUT_SHP = "/data/fort_nelson/hex_probability_grid.shp"
CLASSIFICATION_SUFFIX = "_classification.bin"
# ====================================

def get_aoi_layer():
    """Find the AOI layer by name"""
    for layer in QgsProject.instance().mapLayers().values():
        if layer.name() == AOI_LAYER_NAME and layer.type() == QgsMapLayer.VectorLayer:
            return layer
    # Also try partial match
    for layer in QgsProject.instance().mapLayers().values():
        if AOI_LAYER_NAME.lower() in layer.name().lower() and layer.type() == QgsMapLayer.VectorLayer:
            return layer
    return None

def get_visible_rasters():
    """Get list of visible raster layers in top-to-bottom order"""
    rasters = []
    for lyr in QgsProject.instance().layerTreeRoot().layerOrder():
        if lyr.type() == QgsMapLayer.RasterLayer:
            node = QgsProject.instance().layerTreeRoot().findLayer(lyr.id())
            if node and node.isVisible():
                rasters.append(lyr)
    return rasters

def get_top_raster_at_point(pt, rasters):
    """Find the top-most raster that contains this point"""
    for lyr in rasters:
        if lyr.extent().contains(pt):
            return lyr
    return None

def get_classification_path(raster_layer):
    """Get the classification file path for a raster"""
    src = raster_layer.source()
    base, ext = os.path.splitext(src)
    class_path = base + CLASSIFICATION_SUFFIX
    
    if os.path.exists(class_path):
        return class_path
    
    # Try looking in same directory with .tif removed
    dirname = os.path.dirname(src)
    basename = os.path.basename(base)
    alt_path = os.path.join(dirname, basename + CLASSIFICATION_SUFFIX)
    if os.path.exists(alt_path):
        return alt_path
    
    return None

def generate_hex_grid(extent, spacing, crs):
    """Generate hexagon center points covering the extent"""
    # Hex geometry: pointy-top hexagons
    h_spacing = spacing
    v_spacing = spacing * math.sqrt(3) / 2
    
    xmin, ymin, xmax, ymax = extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()
    
    # Add buffer
    xmin -= spacing
    ymin -= spacing
    xmax += spacing
    ymax += spacing
    
    centers = []
    row = 0
    y = ymin
    while y <= ymax:
        x_offset = (h_spacing / 2) if (row % 2 == 1) else 0
        x = xmin + x_offset
        while x <= xmax:
            centers.append(QgsPointXY(x, y))
            x += h_spacing
        y += v_spacing
        row += 1
    
    return centers

def create_hexagon_geometry(center, spacing):
    """Create a hexagon polygon geometry around a center point"""
    r = spacing / math.sqrt(3)
    
    points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        px = center.x() + r * math.cos(angle)
        py = center.y() + r * math.sin(angle)
        points.append(QgsPointXY(px, py))
    points.append(points[0])
    
    return QgsGeometry.fromPolygonXY([points])

def sample_classification_at_point(class_ds, geo_transform, pt):
    """Sample the classification raster at a geographic point. Returns None for nodata/NaN."""
    inv_gt = gdal.InvGeoTransform(geo_transform)
    if inv_gt is None:
        return None
    
    px = int(inv_gt[0] + inv_gt[1] * pt.x() + inv_gt[2] * pt.y())
    py = int(inv_gt[3] + inv_gt[4] * pt.x() + inv_gt[5] * pt.y())
    
    if px < 0 or py < 0 or px >= class_ds.RasterXSize or py >= class_ds.RasterYSize:
        return None
    
    band = class_ds.GetRasterBand(1)
    val = band.ReadAsArray(px, py, 1, 1)
    
    if val is None:
        return None
    
    val = float(val[0, 0])
    
    # Check for NaN (nodata in classification output)
    if np.isnan(val):
        return None
    
    return val

def aggregate_hex_probabilities(hex_center, spacing, class_ds, geo_transform):
    """
    Sample multiple points within the hexagon and aggregate probabilities.
    Returns (mean_prob, n_valid_samples).
    If all samples are nodata, returns (None, 0).
    """
    # Sample at center and 6 surrounding points, plus 6 more at intermediate radius
    sample_points = [hex_center]
    
    # Inner ring (1/3 radius)
    r_inner = spacing / (3 * math.sqrt(3))
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        px = hex_center.x() + r_inner * math.cos(angle)
        py = hex_center.y() + r_inner * math.sin(angle)
        sample_points.append(QgsPointXY(px, py))
    
    # Outer ring (2/3 radius)
    r_outer = 2 * spacing / (3 * math.sqrt(3))
    for i in range(6):
        angle = i * math.pi / 3  # Offset from inner ring
        px = hex_center.x() + r_outer * math.cos(angle)
        py = hex_center.y() + r_outer * math.sin(angle)
        sample_points.append(QgsPointXY(px, py))
    
    values = []
    for pt in sample_points:
        val = sample_classification_at_point(class_ds, geo_transform, pt)
        if val is not None:
            values.append(val)
    
    if len(values) == 0:
        return None, 0  # All nodata -> hex gets NaN
    
    return sum(values) / len(values), len(values)

# ============ MAIN SCRIPT ============

print("=" * 60)
print("Hexagonal Probability Aggregation")
print("=" * 60)

# Get AOI
aoi_layer = get_aoi_layer()
if aoi_layer is None:
    print(f"ERROR: Could not find AOI layer named '{AOI_LAYER_NAME}'")
    print("Available layers:", [l.name() for l in QgsProject.instance().mapLayers().values()])
else:
    print(f"AOI layer: {aoi_layer.name()}")
    print(f"AOI CRS: {aoi_layer.crs().authid()}")
    
    aoi_extent = aoi_layer.extent()
    aoi_geom = None
    for feat in aoi_layer.getFeatures():
        aoi_geom = feat.geometry()
        break
    
    print(f"AOI extent: {aoi_extent.toString()}")
    
    # Get visible rasters
    rasters = get_visible_rasters()
    print(f"\nFound {len(rasters)} visible raster layers (top to bottom):")
    for i, r in enumerate(rasters):
        class_path = get_classification_path(r)
        has_class = "✓" if class_path else "✗"
        print(f"  {i+1}. {r.name()} [{has_class} classification]")
    
    # Generate hex grid
    print(f"\nGenerating hexagonal grid with {HEX_SPACING}m spacing...")
    hex_centers = generate_hex_grid(aoi_extent, HEX_SPACING, aoi_layer.crs())
    print(f"Generated {len(hex_centers)} hex centers")
    
    # Filter to AOI
    if aoi_geom:
        hex_centers = [c for c in hex_centers if aoi_geom.contains(QgsGeometry.fromPointXY(c))]
        print(f"Filtered to {len(hex_centers)} centers within AOI")
    
    # Create output layer
    out_layer = QgsVectorLayer(f"Polygon?crs={aoi_layer.crs().authid()}", "Hex_Probabilities", "memory")
    provider = out_layer.dataProvider()
    provider.addAttributes([
        QgsField("prob_mean", QVariant.Double),
        QgsField("n_samples", QVariant.Int),
        QgsField("src_image", QVariant.String, len=200)
    ])
    out_layer.updateFields()
    
    # Cache for loaded classification datasets
    class_cache = {}  # raster_source -> (gdal_dataset, geo_transform)
    
    # Process each hexagon
    features = []
    n_processed = 0
    n_with_data = 0
    n_partial = 0
    n_nodata = 0
    
    print("\nProcessing hexagons...")
    
    for i, center in enumerate(hex_centers):
        if (i + 1) % 500 == 0:
            print(f"  Processing hex {i+1}/{len(hex_centers)}... ({n_with_data} with data, {n_nodata} nodata)")
        
        # Find top raster at this location
        top_raster = get_top_raster_at_point(center, rasters)
        
        hex_geom = create_hexagon_geometry(center, HEX_SPACING)
        feat = QgsFeature(out_layer.fields())
        feat.setGeometry(hex_geom)
        
        if top_raster is None:
            # No raster coverage at all
            feat["prob_mean"] = None  # Will be NULL/NaN
            feat["n_samples"] = 0
            feat["src_image"] = "NO_COVERAGE"
            features.append(feat)
            n_nodata += 1
            continue
        
        # Get classification file
        class_path = get_classification_path(top_raster)
        if class_path is None:
            feat["prob_mean"] = None
            feat["n_samples"] = 0
            feat["src_image"] = os.path.basename(top_raster.source()) + " (no classification)"
            features.append(feat)
            n_nodata += 1
            continue
        
        # Load classification dataset (with caching)
        if class_path not in class_cache:
            print(f"  Loading: {os.path.basename(class_path)}")
            ds = gdal.Open(class_path, gdal.GA_ReadOnly)
            if ds is None:
                print(f"  WARNING: Could not open {class_path}")
                continue
            gt = ds.GetGeoTransform()
            class_cache[class_path] = (ds, gt)
        
        class_ds, geo_transform = class_cache[class_path]
        
        # Aggregate probabilities
        prob_mean, n_samples = aggregate_hex_probabilities(
            center, HEX_SPACING, class_ds, geo_transform
        )
        
        n_processed += 1
        
        if prob_mean is None:
            # All samples were nodata
            feat["prob_mean"] = None
            feat["n_samples"] = 0
            feat["src_image"] = os.path.basename(top_raster.source()) + " (nodata)"
            n_nodata += 1
        else:
            feat["prob_mean"] = prob_mean
            feat["n_samples"] = n_samples
            feat["src_image"] = os.path.basename(top_raster.source())
            n_with_data += 1
            if n_samples < 13:  # Less than full sample count
                n_partial += 1
        
        features.append(feat)
    
    # Add features to layer
    provider.addFeatures(features)
    out_layer.updateExtents()
    
    print(f"\n--- Summary ---")
    print(f"Total hexagons: {len(features)}")
    print(f"With valid data: {n_with_data}")
    print(f"  Partial coverage: {n_partial}")
    print(f"Nodata (NaN): {n_nodata}")
    
    # Save to shapefile
    print(f"\nSaving to {OUTPUT_SHP}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_SHP), exist_ok=True)
    
    error = QgsVectorFileWriter.writeAsVectorFormat(
        out_layer, OUTPUT_SHP, "utf-8", aoi_layer.crs(), "ESRI Shapefile"
    )
    if error[0] == QgsVectorFileWriter.NoError:
        print("Shapefile saved successfully")
    else:
        print(f"Error saving shapefile: {error}")
    
    # Add to map
    saved_layer = QgsVectorLayer(OUTPUT_SHP, "Hex_Probabilities", "ogr")
    if saved_layer.isValid():
        QgsProject.instance().addMapLayer(saved_layer)
        print("Layer added to map")
        
        # Apply graduated symbology (yellow -> orange -> red)
        ranges = []
        color_stops = [
            (0.0, 0.1, QColor(255, 255, 200, 180)),   # Very low - pale yellow
            (0.1, 0.2, QColor(255, 240, 150, 180)),   
            (0.2, 0.3, QColor(255, 220, 100, 180)),   
            (0.3, 0.4, QColor(255, 200, 50, 180)),    
            (0.4, 0.5, QColor(255, 170, 30, 180)),    # Medium - orange
            (0.5, 0.6, QColor(255, 140, 20, 180)),    
            (0.6, 0.7, QColor(255, 100, 10, 180)),    
            (0.7, 0.8, QColor(255, 60, 0, 180)),      
            (0.8, 0.9, QColor(230, 30, 0, 180)),      
            (0.9, 1.0, QColor(200, 0, 0, 180)),       # High - dark red
        ]
        
        for lower, upper, color in color_stops:
            sym = QgsFillSymbol.createSimple({
                'color': f'{color.red()},{color.green()},{color.blue()},{color.alpha()}',
                'outline_color': '80,80,80,150',
                'outline_width': '0.1'
            })
            ranges.append(QgsRendererRange(lower, upper, sym, f'{lower:.1f} - {upper:.1f}'))
        
        renderer = QgsGraduatedSymbolRenderer('prob_mean', ranges)
        saved_layer.setRenderer(renderer)
        saved_layer.triggerRepaint()
        
        print("Applied graduated symbology (yellow=low, red=high probability)")
    
    # Cleanup GDAL datasets
    class_cache.clear()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
