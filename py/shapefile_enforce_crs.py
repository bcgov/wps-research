# convert shapefile to specified CRS, where a CRS is indicated in EPSG format
import os
import sys
args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

print(args)
fn = args[1]
try:
    if fn[-4:] != '.shp':
        err("shapefile input req'd")
except Exception:
    err("please check input file")

cmd = "gdalsrsinfo -v " + fn
data = os.popen(cmd).read().strip()

if len(data.split("Validate Succeeds")) < 2:
    err("command:\n\t" + cmd + "\ndid not give expected result. Result:\n\n" + data) 

lines = data.split('\n')
print(data)

