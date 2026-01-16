'''20260115 convert all .tif and .bin raster files in present directory, into a KMZ file
'''

#!/usr/bin/env python3
"""
Convert GeoTIFF (.tif) and ENVI (.bin) raster files to KMZ for Google Earth.
"""
import os
import sys
import zipfile
import tempfile
import shutil
from osgeo import gdal, osr

# Enable GDAL exceptions
gdal.UseExceptions()

KML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Raster Overlays</name>
{overlays}
  </Document>
</kml>
"""

OVERLAY_TEMPLATE = """    <GroundOverlay>
      <name>{name}</name>
      <Icon>
        <href>files/{png_name}</href>
      </Icon>
      <LatLonBox>
        <north>{north}</north>
        <south>{south}</south>
        <east>{east}</east>
        <west>{west}</west>
      </LatLonBox>
    </GroundOverlay>"""


def get_wgs84_bounds(dataset):
    """
    Get the bounding box of a raster in WGS84 (EPSG:4326) coordinates.
    Returns: (west, south, east, north)
    """
    # Get the geotransform
    gt = dataset.GetGeoTransform()

    # Get raster dimensions
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # Calculate corner coordinates in the source CRS
    # Geotransform: [top_left_x, pixel_width, rotation, top_left_y, rotation, pixel_height]
    # Upper left
    ul_x = gt[0]
    ul_y = gt[3]

    # Upper right
    ur_x = gt[0] + cols * gt[1]
    ur_y = gt[3] + cols * gt[4]

    # Lower left
    ll_x = gt[0] + rows * gt[2]
    ll_y = gt[3] + rows * gt[5]

    # Lower right
    lr_x = gt[0] + cols * gt[1] + rows * gt[2]
    lr_y = gt[3] + cols * gt[4] + rows * gt[5]

    # Get source spatial reference
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(dataset.GetProjection())

    # Create WGS84 spatial reference
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)

    # Create coordinate transformation
    transform = osr.CoordinateTransformation(src_srs, wgs84_srs)

    # Transform all corners
    corners = [
        transform.TransformPoint(ul_x, ul_y),
        transform.TransformPoint(ur_x, ur_y),
        transform.TransformPoint(ll_x, ll_y),
        transform.TransformPoint(lr_x, lr_y)
    ]

    # Extract lons and lats
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]

    # Return bounds: west, south, east, north
    west = min(lons)
    east = max(lons)
    south = min(lats)
    north = max(lats)

    return west, south, east, north


def convert_to_png(input_path, output_path):
    """
    Convert a raster to PNG with proper georeferencing for KMZ.
    """
    print(f"  Converting to PNG...")

    # Open the source dataset
    src_ds = gdal.Open(input_path)
    if src_ds is None:
        raise RuntimeError(f"Cannot open {input_path}")

    # Get WGS84 bounds first (for KML)
    west, south, east, north = get_wgs84_bounds(src_ds)

    # Create a temporary VRT to reproject to WGS84
    vrt_options = gdal.WarpOptions(
        format='VRT',
        dstSRS='EPSG:4326',
        outputBounds=(west, south, east, north),
        resampleAlg='bilinear'
    )

    temp_vrt = f"/tmp/{os.path.basename(input_path)}.vrt"
    gdal.Warp(temp_vrt, src_ds, options=vrt_options)

    # Open the VRT
    vrt_ds = gdal.Open(temp_vrt)

    # Translate to PNG with proper scaling
    translate_options = gdal.TranslateOptions(
        format='PNG',
        outputType=gdal.GDT_Byte,
        scaleParams=[[0, 255]]  # Auto-scale to byte range
    )

    gdal.Translate(output_path, vrt_ds, options=translate_options)

    # Clean up
    vrt_ds = None
    src_ds = None
    if os.path.exists(temp_vrt):
        os.remove(temp_vrt)

    return west, south, east, north


def find_rasters(directory="."):
    """Find all .tif and .bin raster files in the directory."""
    rasters = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.tif', '.tiff', '.bin')):
            filepath = os.path.join(directory, filename)

            # For .bin files, check if there's a corresponding .hdr file (ENVI format)
            if filename.lower().endswith('.bin'):
                hdr_file = filepath.replace('.bin', '.hdr')
                if not os.path.exists(hdr_file):
                    print(f"[WARN] Skipping {filename}: no .hdr file found")
                    continue

            rasters.append(filepath)

    return rasters


def create_kmz(rasters, output_kmz="rasters.kmz"):
    """
    Create a KMZ file from a list of raster files.
    """
    if not rasters:
        print("[ERROR] No raster files found!")
        return

    print(f"\nFound {len(rasters)} raster file(s)")
    print(f"Creating {output_kmz}...\n")

    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    files_dir = os.path.join(temp_dir, 'files')
    os.makedirs(files_dir)

    overlays = []

    try:
        for raster_path in rasters:
            raster_name = os.path.basename(raster_path)
            base_name = os.path.splitext(raster_name)[0]
            png_name = f"{base_name}.png"

            print(f"Processing: {raster_name}")

            try:
                # Convert to PNG and get bounds
                png_path = os.path.join(files_dir, png_name)
                west, south, east, north = convert_to_png(raster_path, png_path)

                print(f"  Bounds: W={west:.6f}, S={south:.6f}, E={east:.6f}, N={north:.6f}")

                # Add overlay to KML
                overlays.append(
                    OVERLAY_TEMPLATE.format(
                        name=base_name,
                        png_name=png_name,
                        north=north,
                        south=south,
                        east=east,
                        west=west
                    )
                )

                print(f"  ✓ Success\n")

            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                continue

        if not overlays:
            print("[ERROR] No rasters were successfully processed!")
            return

        # Create KML content
        kml_content = KML_TEMPLATE.format(overlays="\n".join(overlays))

        # Write doc.kml
        kml_path = os.path.join(temp_dir, 'doc.kml')
        with open(kml_path, 'w') as f:
            f.write(kml_content)

        # Create KMZ (which is just a ZIP file)
        with zipfile.ZipFile(output_kmz, 'w', zipfile.ZIP_DEFLATED) as kmz:
            # Add doc.kml at root
            kmz.write(kml_path, 'doc.kml')

            # Add all PNG files in files/ directory
            for png_file in os.listdir(files_dir):
                png_path = os.path.join(files_dir, png_file)
                kmz.write(png_path, f'files/{png_file}')

        print(f"[SUCCESS] Created {output_kmz} with {len(overlays)} overlay(s)")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Raster to KMZ Converter")
    print("=" * 60)

    # Find all rasters in current directory
    rasters = find_rasters()

    if not rasters:
        print("\n[ERROR] No .tif or .bin raster files found in current directory!")
        print("Make sure .bin files have corresponding .hdr files.")
        sys.exit(1)

    # Create KMZ
    create_kmz(rasters)


if __name__ == "__main__":
    main()
