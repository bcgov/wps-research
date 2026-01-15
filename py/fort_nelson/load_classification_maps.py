'''20260109 within QGIS:
    load all classification maps into a display group'''

'''20260109 within QGIS:
    load all classification maps into a display group, newest on top'''
import os
import re
from qgis.core import (
    QgsProject,
    QgsRasterLayer
)

folder = "/ram/parker"
suffix = "_classification.bin"
group_name = "classifications"

def extract_date(filename):
    """Extract date from filename. Returns YYYYMMDD string for sorting.
    
    Handles formats:
    - YYYYMMDD (e.g., 20250823)
    - YYYYMM (e.g., 202508)
    - YYYY (e.g., 2024)
    """
    # Look for 8-digit date (YYYYMMDD)
    match = re.search(r'_(\d{8})_', filename)
    if match:
        return match.group(1)
    
    # Look for 6-digit date (YYYYMM)
    match = re.search(r'_(\d{6})_', filename)
    if match:
        return match.group(1) + "00"  # Pad to 8 digits for sorting
    
    # Look for 4-digit year (YYYY)
    match = re.search(r'_(\d{4})_', filename)
    if match:
        return match.group(1) + "0000"  # Pad to 8 digits for sorting
    
    return "00000000"  # Default for files without dates

project = QgsProject.instance()
root = project.layerTreeRoot()

# Get or create the group
group = root.findGroup(group_name)
if group is None:
    group = root.addGroup(group_name)

# Get matching files and sort by date (oldest first, so newest ends up on top)
files = [f for f in os.listdir(folder) if f.endswith(suffix)]
files_sorted = sorted(files, key=extract_date)  # oldest first

# Load in order (oldest first = added first = ends up at bottom)
for fname in files_sorted:
    path = os.path.join(folder, fname)
    layer = QgsRasterLayer(path, fname, "gdal")
    if layer.isValid():
        project.addMapLayer(layer, False)  # don't add to root
        group.insertLayer(0, layer)  # insert at top of group
    else:
        print(f"Failed to load: {path}")

print("Classification layers loaded (newest on top).")

