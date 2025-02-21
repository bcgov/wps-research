'''20250221 apply color table in csv to raster

highlight a ftl map (ftl.bin) in QGIS to apply this)

in the QGIS python console:

exec(open('qgis_raster_csv_colortable.py').read())

'''

# Specify the path to the color table CSV
color_table_csv = 'bc_fbp_fuel_type_lookup_table.csv'

import os
import sys
from qgis.core import QgsRasterLayer, QgsColorRampShader, QgsRasterShader, QgsColorRampShader
from PyQt5.QtGui import QColor
import csv

# Check if the CSV file exists
if not os.path.exists(color_table_csv):
    print("Error: color table CSV not found")
    sys.exit(1)

# Define the raster layer
raster_layer = iface.activeLayer()  # Get the active raster layer

# Check if the active layer is a raster
if not isinstance(raster_layer, QgsRasterLayer):
    raise Exception("Please select a raster layer")

# Check if the raster layer is valid
if raster_layer is None or not raster_layer.type() == QgsMapLayer.RasterLayer:
    print("No raster layer selected or active layer is not a raster.")
else:
    print("Raster layer selected:", raster_layer.name())

# Initialize an empty color table dictionary
color_table = {}

# Open the CSV file and read the color table
with open(color_table_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Strip whitespace from the column names (in case there are any extra spaces)
    reader.fieldnames = [field.strip() for field in reader.fieldnames]
    
    # Check if 'grid_value', 'r', 'g', 'b' columns exist
    if 'grid_value' not in reader.fieldnames or 'r' not in reader.fieldnames or 'g' not in reader.fieldnames or 'b' not in reader.fieldnames:
        raise Exception(f"Error: CSV does not contain the expected columns 'grid_value', 'r', 'g', 'b'. Found columns: {reader.fieldnames}")
    
    # Read the rows and populate the color table
    for row in reader:
        try:
            # Ensure 'grid_value' is properly converted to float and 'r', 'g', 'b' to integers
            grid_value = float(row['grid_value'])
            r = int(row['r'])
            g = int(row['g'])
            b = int(row['b'])
            color_table[grid_value] = (r, g, b)
        except ValueError as e:
            print(f"Error processing row {row}: {e}")
            continue

# Check the contents of the color table
print("Color Table:", color_table)

# Create a color ramp shader for the raster
shader = QgsRasterShader()
ramp_shader = QgsColorRampShader()

# Set the shader type to interpolated (continuous gradient)
ramp_shader.setColorRampType(QgsColorRampShader.Interpolated)

# Create a list of color stops
color_stops = []

# Add color stops from the CSV color table
for grid_value, (r, g, b) in color_table.items():
    color = QColor(r, g, b)  # Create a QColor object
    # Create QgsColorRampShader.ColorRampItem
    color_stop = QgsColorRampShader.ColorRampItem(grid_value, color)
    color_stops.append(color_stop)  # Add the ColorRampItem to the list

# Set the color ramp item list to the shader
ramp_shader.setColorRampItemList(color_stops)

# Apply the shader to the raster layer using `setColorRampShader`
shader.setRasterShaderFunction(ramp_shader)
renderer = QgsSingleBandPseudoColorRenderer(raster_layer.dataProvider(), 1, shader)

raster_layer.setRenderer(renderer)

# Inform the user that the script has finished
print("Color table applied to the raster layer.")

