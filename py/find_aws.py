import os
import sys
from misc import err, run, exists, sep, band_names, read_hdr, args, datestamp

if len(args) < 3:
    err("find_aws.py [date yyyymmdd] [path to folder for that date] # [optional arg: fire number] # optional arg: skip merge.") # should only have one parameter but I'm tired

fire_number = None
latest, date = None, args[1]
if len(args) > 1:
    latest = '../L2_' + date 

if len(args) > 2:
    latest = args[2]

# get the fire number
if len(args) > 3:
    fire_number = args[3]
else:
    if not exists('../hyperlink') or not exists('../fpf'):
        err('expected to be run from active/$FIRE_NUMBER/yyyymmdd')
    fire_number = os.getcwd().strip().split(sep)[-2]
print("FIRE_NUMBER", fire_number)

# skip merging?
skip_merge = len(args) > 4 

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
        run('cp -v ' + line[:-4] + '.* .')
print(to_merge)


if not skip_merge:
    run('rm -rf tmp_sub* merge* resample')

    if len(to_merge) < 1:
        err("no data found, please check data are retrieved, unzipped, unpacked, converted to the appropriate format, and that this tile is imaged on the provided date")

    run("merge2.py")
