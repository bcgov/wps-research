'''20250704 bcws_poly_bound.py: create bounding boxes for each polygon
'''
import sys
import os
import math
import fiona
from shapely.geometry import shape
from osgeo import gdal, osr
from pyproj import Transformer

def get_expanded_bbox(geom, src_crs_wkt, tgt_crs_wkt, size_ha):
    # Get bounding box
    minx, miny, maxx, maxy = geom.bounds

    # Transform coordinates
    transformer = Transformer.from_crs(src_crs_wkt, tgt_crs_wkt, always_xy=True)
    minx_t, miny_t = transformer.transform(minx, miny)
    maxx_t, maxy_t = transformer.transform(maxx, maxy)

    # Expand bbox
    fire_length_m = math.sqrt(size_ha * 10000)  # ha → m² → m length
    half_len = fire_length_m / 2.0

    expanded_minx = minx_t - half_len
    expanded_miny = miny_t - half_len
    expanded_maxx = maxx_t + half_len
    expanded_maxy = maxy_t + half_len

    return expanded_minx, expanded_miny, expanded_maxx, expanded_maxy

def clip_raster_by_bbox(raster_path, bbox, output_path):
    minx, miny, maxx, maxy = bbox

    # Use GDAL Warp to clip
    gdal.Warp(
        output_path,
        raster_path,
        outputBounds=(minx, miny, maxx, maxy),
        format="ENVI",  # Output as .bin with ENVI header
        dstNodata=0
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <raster_file>")
        sys.exit(1)

    raster_path = sys.argv[1]

    # Open raster and get CRS
    raster_ds = gdal.Open(raster_path)
    raster_wkt = raster_ds.GetProjection()
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_wkt)

    raster_base, _ = os.path.splitext(raster_path)

    # Open shapefile
    with fiona.open('prot_current_fire_polys.shp') as features:
        vector_crs_wkt = features.crs_wkt

        for f in features:
            props = f['properties']
            FIRE_ID = props.get('FIRE_NUM')
            SIZE_HA = props.get('FIRE_SZ_HA', 0)
            FIRE_ID = str(FIRE_ID)
            print([FIRE_ID, SIZE_HA]) # , props])
            if FIRE_ID != "G90216":
                continue

            if not FIRE_ID or SIZE_HA <= 0:
                continue

            geom = shape(f['geometry'])

            # Get expanded bbox in raster CRS
            expanded_bbox = get_expanded_bbox(geom, vector_crs_wkt, raster_wkt, SIZE_HA)

            # Prepare output filename
            output_file = f"{raster_base}_{FIRE_ID}.bin"

            print(f"Clipping raster to expanded bbox for FIRE_ID={FIRE_ID}")
            clip_raster_by_bbox(raster_path, expanded_bbox, output_file)
            print(f"Saved to: {output_file}\n")

if __name__ == "__main__":
    main()

