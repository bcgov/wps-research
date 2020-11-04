import os
import sys
args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

if len(args) < 3:
    err("envi_update_band_names.py [.hdr file with band names to use] " +
        "[.hdr file with band names to overwrite]")




