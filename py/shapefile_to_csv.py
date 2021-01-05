import os
import sys
args = sys.argv

if len(args) < 2:
    print("Error: shapefile_to_csv [input .shp file]")
    sys.exit(1)

fn = args[1]

ofn = args[1][:-4] + '.csv'

cmd = ['ogr2ogr',
       '-f "CSV"',
       ofn,
       fn,
       # "-lco RFC7946=YES"] # "-preserve_fid"]
       "-lco GEOMETRY=AS_XY"]
cmd = ' '.join(cmd)
print(cmd)
a = os.system(cmd)


