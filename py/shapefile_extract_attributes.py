'''20260202 extract attribute table from shapefile'''
#!/usr/bin/env python3

import sys
import csv
from osgeo import ogr

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_attrs.py <input_vector>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_csv = input_file.rsplit(".", 1)[0] + ".csv"

    # Open dataset
    ds = ogr.Open(input_file)
    if ds is None:
        raise RuntimeError(f"Could not open {input_file}")

    layer = ds.GetLayer(0)
    layer_defn = layer.GetLayerDefn()

    # Get field names
    field_names = [
        layer_defn.GetFieldDefn(i).GetName()
        for i in range(layer_defn.GetFieldCount())
    ]

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)

        for feature in layer:
            row = [feature.GetField(name) for name in field_names]
            writer.writerow(row)

    print(f"Attribute table written to: {output_csv}")

if __name__ == "__main__":
    main()

