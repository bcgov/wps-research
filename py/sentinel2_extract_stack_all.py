# extract Sentinel2, resample to 10m, 
# prefix bandnames with dates..
# .. stack everything!

# extract Sentinel-2 data from zip..
#.. if not already extracted (check for .SAFE folder)
import os
import sys
args = sys.argv
sep = os.path.sep
exists = os.path.exists
pd = sep.join(__file__.split(sep)[:-1]) + sep

print(pd)

def err(m):
    print("Error: " + m); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0: err("failed to run: " + str(c))

extract = pd + "extract_sentinel2.py"

zips = os.popen("ls -1 *.zip").readlines()

for z in zips:
    z = z.strip()
    safe = z[:-4] + ".SAFE" # print(safe)
    a = os.system("ls -1 " + safe + os.path.sep + "*.bin")
    
    if not os.path.exists(safe):
        cmd = "python3 " + extract + " " + z
        print(cmd)
        a = os.system(cmd)


