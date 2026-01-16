'''20260114 draw 100km radius circle around Fort Nelson, British Columbia (in QGIS)
Fixed UTM zone: Fort Nelson is at ~122°W, which is UTM zone 10N (EPSG:32610)
'''
from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsVectorLayer,
    QgsField,
    QgsFeature,
    QgsVectorFileWriter,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtCore import QVariant

# ============ PARAMETERS ============
BUFFER_DISTANCE = 100000  # meters (100 km)
OUTPUT_PATH = "/data/fort_nelson/circle_100km.shp"
# ====================================

# Get Fort Nelson coordinates
try:
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="qgis_fort_nelson_locator")
    location = geolocator.geocode("Fort Nelson, British Columbia, Canada")
    if location:
        center_lat = location.latitude
        center_lon = location.longitude
        print(f"Fort Nelson coordinates (from geocoder): {center_lat:.4f}, {center_lon:.4f}")
    else:
        raise ValueError("Geocoder returned no results")
except Exception as e:
    # Fallback to known coordinates
    print(f"Geocoder failed ({e}), using fallback coordinates")
    center_lat = 58.8050
    center_lon = -122.6972
    print(f"Fort Nelson coordinates (fallback): {center_lat:.4f}, {center_lon:.4f}")

# Determine correct UTM zone based on longitude
# UTM zones are 6 degrees wide, zone 1 starts at 180°W
utm_zone = int((center_lon + 180) / 6) + 1
hemisphere = "N" if center_lat >= 0 else "S"
epsg_code = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone

print(f"Using UTM zone {utm_zone}{hemisphere} (EPSG:{epsg_code})")

# Create CRS objects
crs_wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
crs_utm = QgsCoordinateReferenceSystem(f"EPSG:{epsg_code}")

# Transform point from WGS84 to UTM
transform = QgsCoordinateTransform(crs_wgs84, crs_utm, QgsProject.instance())
point_wgs84 = QgsPointXY(center_lon, center_lat)
point_utm = transform.transform(point_wgs84)

print(f"UTM coordinates: {point_utm.x():.1f} E, {point_utm.y():.1f} N")

# Create point geometry and buffer
point_geom = QgsGeometry.fromPointXY(point_utm)
buffer_geom = point_geom.buffer(BUFFER_DISTANCE, segments=64)

# Create buffer layer
buffer_layer = QgsVectorLayer(f"Polygon?crs=EPSG:{epsg_code}", "circle_100km", "memory")
buffer_provider = buffer_layer.dataProvider()

# Add some metadata fields
buffer_provider.addAttributes([
    QgsField("name", QVariant.String, len=50),
    QgsField("radius_m", QVariant.Int),
    QgsField("center_lat", QVariant.Double),
    QgsField("center_lon", QVariant.Double)
])
buffer_layer.updateFields()

# Add feature
buffer_feat = QgsFeature(buffer_layer.fields())
buffer_feat.setGeometry(buffer_geom)
buffer_feat["name"] = "Fort Nelson 100km AOI"
buffer_feat["radius_m"] = BUFFER_DISTANCE
buffer_feat["center_lat"] = center_lat
buffer_feat["center_lon"] = center_lon
buffer_provider.addFeature(buffer_feat)
buffer_layer.updateExtents()

# Export to shapefile
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

error = QgsVectorFileWriter.writeAsVectorFormat(
    buffer_layer, OUTPUT_PATH, "UTF-8", buffer_layer.crs(), "ESRI Shapefile"
)

if error[0] == QgsVectorFileWriter.NoError:
    print(f"Shapefile saved: {OUTPUT_PATH}")
else:
    print(f"Error saving shapefile: {error}")

# Add to QGIS project
QgsProject.instance().addMapLayer(buffer_layer)

print(f"✅ {BUFFER_DISTANCE/1000:.0f} km circle created around Fort Nelson")
print(f"   CRS: EPSG:{epsg_code} (UTM zone {utm_zone}{hemisphere})")



