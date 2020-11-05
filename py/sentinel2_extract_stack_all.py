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

extract = pd + "extract_sentinel2.py" # command to extract a zip
zips = os.popen("ls -1 *.zip").readlines() # list the zip files
raster_files = []

for z in zips:
    z = z.strip()
    safe = z[:-4] + ".SAFE" # print(safe)
    bins = [x.strip() for x in os.popen("ls -1 " + safe + os.path.sep + "*m_EPSG_*.bin").readlines()] # don't pull the TCI true colour image. Already covered in 10m
    print(safe)

    if len(bins) != 3:
        err("unexpected number of bin files (expected 3): " + str('\n'.join(bins)))

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
    m20r, m60r = m20[:-4] + '_10m.bin', m60[:-4] + '_10m.bin'
    
    def resample(src, ref, dst): # resample src onto ref, w output file dst
        cmd = ['python3 ' + pd + 'raster_project_onto.py',
           src, # source image
           ref, # project onto
           dst] # result image
 
        if not exists(dst):
            run(' '.join(cmd))

    resample(m20, m10, m20r) # resample the 20m 
    resample(m60, m10, m60r) # resample the 60m

    # now do the stacking..
    print(m10)
     
    sfn = safe + sep + m10.split(sep)[-1].replace("_10m", "")[:-4] + '_10m.bin'  # stacked file name..
    print(sfn)

    cmd = ['cat',
            m10,
            m20r,
            m60r,
            '>', 
            sfn] # cat bands together, don't forget to "cat" the header files after..

    if not exists(sfn):
        run(' '.join(cmd))

    # add a date prefix
    dp = '"' + safe.split('T')[0].strip().split('_')[-1].strip()
    dp10 = dp + ' 10m: "'
    dp20 = dp + ' 20m: "'
    dp60 = dp + ' 60m: "'

    # now "cat" the header files together
    shn = sfn[:-4] + '.hdr' # header file name for stack

    cmd = ['python3', # envi_header_cat.py is almost like a reverse-polish notation. Have to put the "first thing" on the back..
           pd + 'envi_header_cat.py',
           m20r[:-4] + '.hdr',
           m10[:-4] + '.hdr', 
           shn,
           dp20,
           dp10]
    
    cmd = ' '.join(cmd)
    run(cmd)

    cmd = ['python3',
            pd + 'envi_header_cat.py',
            m60r[:-4] + '.hdr',
            shn[:-4] + '.hdr',
            shn,
            dp60]

    cmd = ' '.join(cmd)
    run(cmd)
    raster_files.append(sfn)

# cat the bin files together, combining headers
cmd = ['python3', pd + 'raster_stack.py']
cmd = cmd + raster_files + ['raster.bin']
a = os.system(' '.join(cmd))
