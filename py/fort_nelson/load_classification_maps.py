'''20260109 within QGIS:
    load all classification maps into a display group'''

import os
from qgis.core import (
    QgsProject,
    QgsRasterLayer
)

folder = "/ram/parker"
suffix = "_classification.bin"
group_name = "classifications"

project = QgsProject.instance()
root = project.layerTreeRoot()

# Get or create the group
group = root.findGroup(group_name)
if group is None:
    group = root.addGroup(group_name)

# Load matching files
for fname in sorted(os.listdir(folder)):
    if fname.endswith(suffix):
        path = os.path.join(folder, fname)

        layer = QgsRasterLayer(path, fname, "gdal")
        if layer.isValid():
            project.addMapLayer(layer, False)  # don't add to root
            group.addLayer(layer)
        else:
            print(f"Failed to load: {path}")

print("Classification layers loaded.")


