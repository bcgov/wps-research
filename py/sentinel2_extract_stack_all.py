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
    bins = [x.strip() for x in os.popen("ls -1 " + safe + os.path.sep + "*m_EPSG*.bin").readlines()] # don't pull the TCI true colour image. Already covered in 10m
    print(safe)

    if len(bins) != 3:
        err("unexpected number of bin files (expected 3): " + str(bins))

    m10, m20, m60 = bins
    print("  10m:", m10)
    print("  20m:", m20)
    print("  60m:", m60)

    #for b in bins:
    #    print('  ' + b) # print('  ' + b.split(sep)[-1])
    if not os.path.exists(safe):
        cmd = "python3 " + extract + " " + z
        print(cmd)
        a = os.system(cmd)

    
    # names for files resampled to 10m
    m20r = m20[:-4] + '_10m.bin'
    print("   " + m20r)
    m60r = m60[:-4] + '_10m.bin'
    print("   " + m60r)

    cmd = ['python3 ' + pd + 'raster_project_onto.py',
           m20,
           m10,
           m20r]

    cmd = ' '.join(cmd)
    print(cmd)
