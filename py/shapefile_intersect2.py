#!/usr/bin/env python3
'''20260127: shapefile_intersect.py: Find polygons that spatially intersect between two shapefiles.Creates output shapefiles containing only the intersecting polygons from each input.

Usage: python shapefile_intersect.py <shapefile1> <shapefile2>

Output files are named <input>_intersect.shp and are in the CRS of the second shapefile.'''

import sys
import os
import multiprocessing as mp
from joblib import Parallel, delayed
from osgeo import ogr, osr

# Enable OGR exceptions
ogr.UseExceptions()

single_thread = False  # Set to True to force single-threaded execution


def parfor(my_function, my_inputs, n_thread=min(32, int(mp.cpu_count()))):
    print("PARFOR", n_thread)
    if n_thread == 1 or single_thread:
        return [my_function(my_inputs[i]) for i in range(len(my_inputs))]
    else:
        n_thread = mp.cpu_count() if n_thread is None else n_thread
        if my_inputs is None or type(my_inputs) == list and len(my_inputs) == 0:
            return []
        return Parallel(n_jobs=n_thread)(delayed(my_function)(input) for input in my_inputs)


def load_shapefile(shapefile_path):
    """Load a shapefile and return the datasource, layer, and spatial reference."""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    datasource = driver.Open(shapefile_path, 0)  # 0 = read-only
    if datasource is None:
        raise RuntimeError(f"Could not open shapefile: {shapefile_path}")
    
    layer = datasource.GetLayer()
    srs = layer.GetSpatialRef()
    
    return datasource, layer, srs


def get_geometries_with_fids(layer, transform=None):
    """
    Extract all geometries from a layer with their FIDs.
    Optionally transform them to a new CRS.
    Returns list of (fid, geometry_wkb) tuples.
    """
    geometries = []
    layer.ResetReading()
    
    for feature in layer:
        fid = feature.GetFID()
        geom = feature.GetGeometryRef()
        
        if geom is None:
            continue
        
        # Clone the geometry so we can transform it without affecting the original
        geom_clone = geom.Clone()
        
        if transform is not None:
            geom_clone.Transform(transform)
        
        # Prepare geometry for faster spatial operations
        geom_clone = geom_clone.Buffer(0)  # Fix any invalid geometries
        
        # Store as WKB for thread safety (OGR geometry objects aren't thread-safe)
        geometries.append((fid, geom_clone.ExportToWkb()))
    
    layer.ResetReading()
    return geometries


def check_intersection(args):
    """
    Check if a geometry from set1 intersects with any geometry in set2.
    Args is a tuple: (fid1, wkb1, list of (fid2, wkb2) tuples)
    Returns (fid1, True/False, list of intersecting fid2s)
    """
    fid1, wkb1, geoms2_wkb = args
    
    geom1 = ogr.CreateGeometryFromWkb(wkb1)
    
    if geom1 is None or geom1.IsEmpty():
        return (fid1, False, [])
    
    intersecting_fids = []
    
    for fid2, wkb2 in geoms2_wkb:
        geom2 = ogr.CreateGeometryFromWkb(wkb2)
        
        if geom2 is None or geom2.IsEmpty():
            continue
        
        # Use proper computational geometry intersection test
        # First do a quick bounding box check, then precise intersection
        if geom1.Intersects(geom2):
            intersecting_fids.append(fid2)
    
    has_intersection = len(intersecting_fids) > 0
    return (fid1, has_intersection, intersecting_fids)


def create_output_shapefile(input_path, output_path, layer, fids_to_keep, target_srs, transform=None):
    """
    Create a new shapefile containing only features with FIDs in fids_to_keep.
    Output is in target_srs coordinate system.
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    
    # Remove output file if it exists
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    # Create output datasource
    out_datasource = driver.CreateDataSource(output_path)
    if out_datasource is None:
        raise RuntimeError(f"Could not create output shapefile: {output_path}")
    
    # Get the geometry type from the input layer
    geom_type = layer.GetGeomType()
    
    # Create output layer with target SRS
    out_layer = out_datasource.CreateLayer(
        os.path.splitext(os.path.basename(output_path))[0],
        target_srs,
        geom_type
    )
    
    # Copy field definitions from input layer
    layer_defn = layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)
    
    out_layer_defn = out_layer.GetLayerDefn()
    
    # Copy features with matching FIDs
    layer.ResetReading()
    fids_set = set(fids_to_keep)
    
    for feature in layer:
        fid = feature.GetFID()
        if fid in fids_set:
            # Create new feature
            out_feature = ogr.Feature(out_layer_defn)
            
            # Copy attributes
            for i in range(out_layer_defn.GetFieldCount()):
                out_feature.SetField(i, feature.GetField(i))
            
            # Copy and optionally transform geometry
            geom = feature.GetGeometryRef()
            if geom is not None:
                geom_clone = geom.Clone()
                if transform is not None:
                    geom_clone.Transform(transform)
                out_feature.SetGeometry(geom_clone)
            
            out_layer.CreateFeature(out_feature)
            out_feature = None
    
    layer.ResetReading()
    out_datasource = None
    
    print(f"Created: {output_path} with {len(fids_to_keep)} features")


def generate_output_path(input_path):
    """Generate output path by adding _intersect before .shp extension."""
    base, ext = os.path.splitext(input_path)
    return f"{base}_intersect{ext}"


def main():
    if len(sys.argv) != 3:
        print("Usage: python shapefile_intersect.py <shapefile1> <shapefile2>")
        print("\nFinds polygons that intersect between two shapefiles.")
        print("Output files are named <input>_intersect.shp")
        print("All output is in the CRS of the second shapefile.")
        sys.exit(1)
    
    shapefile1_path = sys.argv[1]
    shapefile2_path = sys.argv[2]
    
    # Verify input files exist
    for path in [shapefile1_path, shapefile2_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    print(f"Loading shapefile 1: {shapefile1_path}")
    ds1, layer1, srs1 = load_shapefile(shapefile1_path)
    
    print(f"Loading shapefile 2: {shapefile2_path}")
    ds2, layer2, srs2 = load_shapefile(shapefile2_path)
    
    # Target CRS is from shapefile 2
    target_srs = srs2.Clone() if srs2 else None
    
    # Create coordinate transformation from shapefile1 CRS to shapefile2 CRS
    transform1_to_2 = None
    if srs1 and srs2:
        if not srs1.IsSame(srs2):
            print(f"Transforming shapefile 1 from {srs1.GetName()} to {srs2.GetName()}")
            transform1_to_2 = osr.CoordinateTransformation(srs1, srs2)
        else:
            print("Both shapefiles are in the same CRS")
    else:
        print("Warning: One or both shapefiles missing CRS information")
    
    # Extract geometries (shapefile1 transformed to shapefile2's CRS)
    print("Extracting geometries from shapefile 1...")
    geoms1 = get_geometries_with_fids(layer1, transform1_to_2)
    print(f"  Found {len(geoms1)} geometries")
    
    print("Extracting geometries from shapefile 2...")
    geoms2 = get_geometries_with_fids(layer2)
    print(f"  Found {len(geoms2)} geometries")
    
    if len(geoms1) == 0 or len(geoms2) == 0:
        print("Error: One or both shapefiles have no valid geometries")
        sys.exit(1)
    
    # Prepare inputs for parallel processing
    # Each task checks one geometry from set1 against all geometries in set2
    print("\nChecking intersections (shapefile 1 vs shapefile 2)...")
    inputs_1vs2 = [(fid1, wkb1, geoms2) for fid1, wkb1 in geoms1]
    results_1vs2 = parfor(check_intersection, inputs_1vs2)
    
    # Collect FIDs of intersecting features
    fids1_intersecting = set()
    fids2_intersecting = set()
    
    for fid1, has_intersection, intersecting_fids2 in results_1vs2:
        if has_intersection:
            fids1_intersecting.add(fid1)
            fids2_intersecting.update(intersecting_fids2)
    
    print(f"\nResults:")
    print(f"  Shapefile 1: {len(fids1_intersecting)} of {len(geoms1)} polygons intersect")
    print(f"  Shapefile 2: {len(fids2_intersecting)} of {len(geoms2)} polygons intersect")
    
    # Generate output paths
    output1_path = generate_output_path(shapefile1_path)
    output2_path = generate_output_path(shapefile2_path)
    
    # Create output shapefiles
    print(f"\nCreating output shapefiles...")
    
    if len(fids1_intersecting) > 0:
        create_output_shapefile(
            shapefile1_path, output1_path, layer1, 
            fids1_intersecting, target_srs, transform1_to_2
        )
    else:
        print(f"No intersecting features in shapefile 1, skipping output")
    
    if len(fids2_intersecting) > 0:
        create_output_shapefile(
            shapefile2_path, output2_path, layer2,
            fids2_intersecting, target_srs, None
        )
    else:
        print(f"No intersecting features in shapefile 2, skipping output")
    
    # Cleanup
    ds1 = None
    ds2 = None
    
    print("\nDone!")


if __name__ == "__main__":
    main()


