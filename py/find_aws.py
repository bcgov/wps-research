import os
import sys
from misc import err, run, exists, sep, band_names, read_hdr, args, datestamp

if len(args) < 3:
    err("find_aws.py [date yyyymmdd] [path to folder for that date]") # should only have one parameter but I'm tired

latest, date = None, args[1]
if len(args) > 1:
    latest = '../L2_' + date 

if len(args) > 2:
    latest = args[2]

if not exists('../hyperlink') or not exists('../fpf'):
    err('expected to be run from active/$FIRE_NUMBER/yyyymmdd')

# get the fire number
fire_number = os.getcwd().strip().split(sep)[-2]
print("FIRE_NUMBER", fire_number)

# get the tiles
tiles = open("/home/" + os.popen("whoami").read().strip() + sep + "GitHub/wps-research/py/.select/" + fire_number).read().strip().split()
print("TILES", tiles)

# get the latest AWS folder. Assume "active" folder is one level up!

'''
if False:
    lines = [x.strip() for x in os.popen("ls -1 ../ | grep L2_").readlines()]
    lines.sort(reverse=True)  # decreasing order, AKA most recent first
    for line in lines:
        print(line)

    if latest is None:
        latest = lines[0] # most recent date of AWS retrieval 
    print("LATEST", latest)
'''
# latest = 'L2_' + datestamp()

to_merge = []
for tile in tiles:
    print(tile)
    cmd = "ls -1 " + latest + sep + "*" + tile + "*.bin"
    print(cmd)
    for line in [x.strip() for x in os.popen("ls -1 " + latest + sep + "*" + tile + "*.bin").readlines()]:
        if len(line.split('swir')) > 1:
            err("please remove _swir_ files")
        to_merge += [line]
    cmd = "ls -1 " + latest + "*" + tile + "*.bin"
    for line in [x.strip() for x in os.popen(cmd).readlines()]:
        to_merge += [line]        
        run('cp -v ' + line + ' .')
print(to_merge)

if len(to_merge) < 1:
    err("no data found, please check data are retrieved, unzipped, unpacked, converted to the appropriate format, and that this tile is imaged on the provided date")

# ../L2_20230520/S2B_MSIL2A_20230520T190919_N0509_R056_T10UEC_20230520T214840.hdr
first = to_merge[0][:-4] + '.hdr'
ts = first.split(sep)[-1].split("_")[2].split('T')[1][0:4]

out_file = ts + ".bin"
out_hdr = ts + ".hdr"

if not exists(out_file):
    run("merge2.py")
    # run("gdal_merge.py -of ENVI -ot Float32 -n nan " + " ".join(to_merge) + " -o " + out_file)
    run("fh " + out_hdr)

run('envi_header_copy_bandnames.py ' + first + ' ' + out_hdr)
#samples, lines, bands = read_hdr(out_hdr)

#bn = band_names(first)


#run(' '.join(['envi_header_modify.py', out_hdr, str(lines), str(samples), str(bands)] + bn))
# merge the appropriate tiles if the merge file isn't already available

