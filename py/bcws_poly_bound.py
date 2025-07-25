'''20250704 bcws_poly_bound.py: create bounding boxes for each polygon
'''
import os
import sys
import math
import fiona
from shapely.geometry import shape
from osgeo import gdal, osr
from pyproj import Transformer
gdal.DontUseExceptions()

def get_expanded_bbox(geom, src_crs_wkt, tgt_crs_wkt, size_ha):
    minx, miny, maxx, maxy = geom.bounds
    transformer = Transformer.from_crs(src_crs_wkt, tgt_crs_wkt, always_xy=True)
    minx_t, miny_t = transformer.transform(minx, miny)
    maxx_t, maxy_t = transformer.transform(maxx, maxy)

    fire_length_m = math.sqrt(size_ha * 10000)  # ha → m² → m
    half_len = fire_length_m / 2.0

    return minx_t - half_len, miny_t - half_len, maxx_t + half_len, maxy_t + half_len

def clip_raster_by_bbox(raster_path, bbox, output_path):
    minx, miny, maxx, maxy = bbox
    if os.path.exists(output_path):
        print(f"+r {output_path}")
        return

    gdal.Warp(
        output_path,
        raster_path,
        outputBounds=(minx, miny, maxx, maxy),
        format="ENVI",  # Output as .bin + .hdr
        dstNodata=0
    )

def main():
    fire_number_use = None

    if len(sys.argv) > 1:
        fire_number_use = sys.argv[1]

    # Only include .bin raster files
    raster_files = [f for f in os.listdir('.') if f.lower().endswith('.bin')]

    if not raster_files:
        print("No .bin raster files found in current directory.")
        return

    # Open shapefile
    required_fields = {'FIRE_NUM', 'FIRE_SZ_HA'}
    with fiona.open('prot_current_fire_polys.shp') as features:

        first_props = features[0]['properties'].keys()
        missing = required_fields - set(first_props)
        if missing:
            print(f"Error: Missing required fields in shapefile: {', '.join(missing)}")
            exit(1)

        vector_crs_wkt = features.crs_wkt

        for f in features:
            props = f['properties']
            FIRE_ID = props.get('FIRE_NUM')
            SIZE_HA = props.get('FIRE_SZ_HA')
            #if not FIRE_ID == 'G90216':
            #    continue
            
            if not FIRE_ID or SIZE_HA <= 0:
                continue

            if fire_number_use is not None and FIRE_ID != fire_number_use:
                continue

            geom = shape(f['geometry'])
            print(f"\nFIRE_ID: {FIRE_ID}, SIZE_HA: {SIZE_HA}")

            for raster_path in raster_files:
                # Open raster and get CRS
                raster_ds = gdal.Open(raster_path)
                if raster_ds is None:
                    print(f"Failed to open raster: {raster_path}")
                    continue

                raster_wkt = raster_ds.GetProjection()

                try:
                    expanded_bbox = get_expanded_bbox(geom, vector_crs_wkt, raster_wkt, SIZE_HA)
                except Exception as e:
                    print(f"Error transforming bbox for FIRE_ID {FIRE_ID}: {e}")
                    continue

                # Output directory and filename
                output_dir = f"fire_{FIRE_ID}"
                os.makedirs(output_dir, exist_ok=True)

                raster_base, _ = os.path.splitext(os.path.basename(raster_path))
                output_file = f"{raster_base}_{FIRE_ID}.bin"
                output_path = os.path.join(output_dir, output_file)

                clip_raster_by_bbox(raster_path, expanded_bbox, output_path)
                print(f"+w {output_path}")

if __name__ == "__main__":
    main()

