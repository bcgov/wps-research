'''20251203 load tif files on ramdisk, into QGIS.  ''' 

import os
import re
from qgis.core import QgsRasterLayer, QgsProject

# Folder to search
base_dir = "/ram/parker/"

# Regex patterns
fire_pattern = re.compile(r"pl_(g\d{5})", re.IGNORECASE)
year_pattern = re.compile(r"_(20\d{2})")   # captures 2023, 2024, 2025, ...

project = QgsProject.instance()
root = project.layerTreeRoot()


def get_or_create_group(parent, name):
    """Return an existing group or create a new one."""
    group = parent.findGroup(name)
    if group is None:
        group = parent.addGroup(name)
    return group


# ----------------------------------------------------------
# STEP 1 — Collect all .tif files (full paths)
# ----------------------------------------------------------
all_tifs = []

for root_dir, dirs, files in os.walk(base_dir):
    for f in files:
        if f.lower().endswith(".tif"):
            all_tifs.append(os.path.join(root_dir, f))


# ----------------------------------------------------------
# STEP 2 — Extract metadata (year + fire number)
#          and sort by year DESCENDING (newest first)
# ----------------------------------------------------------
def parse_metadata(path):
    fname = os.path.basename(path)

    # Fire number (first g##### after pl_)
    fire_match = fire_pattern.search(fname)
    fire_number = fire_match.group(1).upper() if fire_match else None

    # Year (first 4-digit year pattern)
    year_match = year_pattern.search(fname)
    year = int(year_match.group(1)) if year_match else None

    return year, fire_number


# Build list with parsed info
items = []
for tif in all_tifs:
    year, fire = parse_metadata(tif)
    if year and fire:
        items.append((year, fire, tif))
    else:
        print("Skipping (missing metadata):", tif)

# Sort so NEWEST IMAGERY is on top
# (Year descending)
items.sort(key=lambda x: x[0], reverse=True)


# ----------------------------------------------------------
# STEP 3 — Add layers to QGIS in sorted order
# ----------------------------------------------------------
for year, fire_number, fpath in items:

    # Create hierarchical groups
    year_group = get_or_create_group(root, str(year))
    fire_group = get_or_create_group(year_group, fire_number)

    # Load raster
    layer_name = os.path.basename(fpath)
    rlayer = QgsRasterLayer(fpath, layer_name)

    if not rlayer.isValid():
        print("Invalid raster:", fpath)
        continue

    # Add to project without putting in root
    project.addMapLayer(rlayer, False)

    # Place into tree group
    fire_group.addLayer(rlayer)

    print(f"Added: {fpath}  →  {year}/{fire_number}")

print("\n=== Done. Images sorted newest → oldest. ===")

