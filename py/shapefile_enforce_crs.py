# convert shapefile to specified CRS, where a CRS is indicated in EPSG format
import os
import sys
args = sys.argv

def err(m):
    print("Error: " + m); sys.exit(1)

print(args)

if len(args) < 2:
    err("python3 shapefile_enforce_crs.py [input shapefile] [optional argument: destination crs EPSG number] # default EPSG 32609")


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

src_EPSG = None
for line in lines:
    line = line.strip()
    w = line.split('ID["EPSG')
    if len(w) > 1:
        w = w[1].strip('"').strip(',').strip(']').strip(']').strip()
        src_EPSG = int(w)

