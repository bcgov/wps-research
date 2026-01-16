'''20260115 convert all .tif and .bin raster files in present directory, into a KMZ file
'''

#!/usr/bin/env python3

import os
import zipfile
from osgeo import gdal, osr

KML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
{name_blocks}
</Document>
</kml>
"""

OVERLAY_TEMPLATE = """
<GroundOverlay>
  <name>{name}</name>
  <Icon>
    <href>{filename}</href>
  </Icon>
  <LatLonBox>
    <north>{north}</north>
    <south>{south}</south>
    <east>{east}</east>
    <west>{west}</west>
  </LatLonBox>
</GroundOverlay>
"""


def raster_bounds_to_wgs84(ds):
    gt = ds.GetGeoTransform()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    corners = [
        (0, 0),
        (cols, 0),
        (cols, rows),
        (0, rows)
    ]

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    lons, lats = [], []

    for px, py in corners:
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        lon, lat, _ = transform.TransformPoint(x, y)
        lons.append(lon)
        lats.append(lat)

    return max(lats), min(lats), max(lons), min(lons)


def main():
    rasters = [f for f in os.listdir(".")
               if f.lower().endswith((".tif", ".bin"))]

    if not rasters:
        raise RuntimeError("No .tif or .bin rasters found")

    overlays = []

    for raster in rasters:
        print(f"[INFO] Processing {raster}")
        ds = gdal.Open(raster)
        if ds is None:
            print(f"[WARN] Skipping unreadable raster: {raster}")
            continue

        north, south, east, west = raster_bounds_to_wgs84(ds)

        overlays.append(
            OVERLAY_TEMPLATE.format(
                name=os.path.splitext(raster)[0],
                filename=raster,
                north=north,
                south=south,
                east=east,
                west=west
            )
        )

    kml_content = KML_TEMPLATE.format(
        name_blocks="\n".join(overlays)
    )

    kmz_name = "rasters.kmz"

    with zipfile.ZipFile(kmz_name, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml_content)
        for raster in rasters:
            z.write(raster)

    print(f"[SUCCESS] Created {kmz_name} with {len(overlays)} rasters")


if __name__ == "__main__":
    main()



