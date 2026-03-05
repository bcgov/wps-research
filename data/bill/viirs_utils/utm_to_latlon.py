#!/usr/bin/env python3
"""
Convert a UTM bounding box to lat/lon and print the LAADS DAAC URL regions parameter.
Usage: python utm_bbox_to_url.py
"""

from pyproj import Transformer

# ── Edit these values ────────────────────────────────────────────────
min_easting  = 699960
min_northing = 5790240
max_easting  = 809760
max_northing = 5900040
utm_zone     = 9
northern     = True
# ─────────────────────────────────────────────────────────────────────

hemi = "north" if northern else "south"
crs  = f"+proj=utm +zone={utm_zone} +{hemi} +datum=WGS84"
t    = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

min_lon, min_lat = t.transform(min_easting,  min_northing)
max_lon, max_lat = t.transform(max_easting,  max_northing)

print(f"W (min_lon): {min_lon:.6f}")
print(f"S (min_lat): {min_lat:.6f}")
print(f"E (max_lon): {max_lon:.6f}")
print(f"N (max_lat): {max_lat:.6f}")
print()
print("URL-encoded regions parameter:")

encoded = f"%5BBBOX%5DN{max_lat:.6f}%20S{min_lat:.6f}%20E{max_lon:.6f}%20W{min_lon:.6f}"

print(f"  regions={encoded}")