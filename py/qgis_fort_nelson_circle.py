'''20251110 draw 100km radius circle around Fort Nelson, British Columbia ( in QGIS ) 
'''

from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsVectorLayer,
    QgsField,
    QgsFeature,
    QgsVectorFileWriter
)
from qgis.PyQt.QtCore import QVariant

# --- PARAMETERS ---
# Choose coordinates for the center point (lon, lat)
from geopy.geocoders import Nominatim

# Initialize geocoder
geolocator = Nominatim(user_agent="qgis_fort_nelson_locator")

# Fetch coordinates for Fort Nelson, BC
location = geolocator.geocode("Fort Nelson, British Columbia, Canada")

if location:
    center_lat = location.latitude
    center_lon = location.longitude
    print(f"Fort Nelson coordinates: {center_lat}, {center_lon}")
else:
    raise ValueError("Could not find Fort Nelson coordinates")


buffer_distance = 100000                    # meters (100 km)
output_path = "circle_100km.shp"    # Output shapefile

# --- CREATE A POINT IN A PROJECTED CRS ---
# Use an appropriate projected CRS (UTM zone for your area)
crs = QgsCoordinateReferenceSystem("EPSG:32630")  # UTM zone 30N (covers London)

# Create a temporary point layer
layer = QgsVectorLayer("Point?crs=EPSG:32630", "center_point", "memory")
prov = layer.dataProvider()

# Add a feature (the center point)
point = QgsPointXY(center_lon, center_lat)

# If your point is in geographic coordinates (WGS84), reproject it:
from qgis.core import QgsCoordinateTransform, QgsCoordinateReferenceSystem
transform = QgsCoordinateTransform(QgsCoordinateReferenceSystem("EPSG:4326"), crs, QgsProject.instance())
point_projected = transform.transform(point)

feat = QgsFeature()
feat.setGeometry(QgsGeometry.fromPointXY(point_projected))
prov.addFeature(feat)
layer.updateExtents()

# --- CREATE BUFFER (CIRCLE) ---
buffer_geom = feat.geometry().buffer(buffer_distance, segments=64)

# --- SAVE AS NEW POLYGON SHAPEFILE ---
buffer_layer = QgsVectorLayer("Polygon?crs=EPSG:32630", "circle_100km", "memory")
buffer_provider = buffer_layer.dataProvider()
buffer_feat = QgsFeature()
buffer_feat.setGeometry(buffer_geom)
buffer_provider.addFeature(buffer_feat)
buffer_layer.updateExtents()

# Export to shapefile
QgsVectorFileWriter.writeAsVectorFormat(buffer_layer, output_path, "UTF-8", buffer_layer.crs(), "ESRI Shapefile")

# Add to QGIS project
QgsProject.instance().addMapLayer(buffer_layer)

print("✅ 100 km circle created and added to map:", output_path)

