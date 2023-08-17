'''20230817: run this from fire subfolder in "active" directory

20230808: need to update to look for previous dates.'''
import os
import sys
from misc import err, run, exists, sep, band_names, read_hdr, args, datestamp


active = '/media/' + os.popen('whoami').read().strip() + '/disk4/active/'
#if len(args) < 3:
#    err("find_aws.py [date yyyymmdd] [path to folder for that date] # [optional arg: fire number] # optional arg: skip merge.") # should only have one parameter but I'm tired

fire_number = os.path.abspath(os.getcwd().strip()).split(os.path.sep)[-1]
print(fire_number)

if len(fire_number) != 6 or not os.path.exists(os.getcwd() + os.path.sep + 'fpf') or not os.path.exists(os.getcwd() + os.path.sep + 'hyperlink'):
    err("please run from active/fire_number")

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
    cmd = "find " + active + sep + ' -name "S2*' + tile + '*.bin"'
    lines = [x.strip() for x in os.popen(cmd).readlines()]

    to_sort = []
    for line in lines:
        a = os.path.abspath(line)
        w = a.split(sep)
        if w[-2][0] == 'L' and w[-2][1] == '1':
            ww = w[-1].split('_')
            ts = ww[2][0:8]
            to_sort.append([ts, w[-1], a])

            
    to_sort.sort(reverse=False)
    for t in to_sort:
        if len(t[0]) != 8:
            err('unexpected ts length')
        print(t[0], t[1])

        if not exists(t[0]):
            os.mkdir(t[0])

        dest = t[0] + sep + t[1]
        if not exists(dest):
            run('ln -s ' + t[2] + ' ' + dest)
        dest = dest[:-4] + '.hdr'
        if not exists(dest):
            run('ln -s ' + t[2][:-4] + '.hdr ' + dest[:-4] + '.hdr')

'''
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
'''
