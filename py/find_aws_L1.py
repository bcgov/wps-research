'''20230521 need to merge this script with find_aws.py
'''
import os
import sys
from misc import err, run, exists, sep, band_names, read_hdr, args

latest = None
if len(args) > 1:
	latest = args[1]	

if not exists('hyperlink') or not exists('fpf'):
	err('expected to be run from active/$FIRE_NUMBER')

# get the fire number
fire_number = os.getcwd().strip().split(sep)[-1]
print("FIRE_NUMBER", fire_number)

# get the tiles
tiles = open("/home/" + os.popen("whoami").read().strip() + sep + "GitHub/wps-research/py/.select/" + fire_number).read().strip().split()
print("TILES", tiles)

# get the latest AWS folder. Assume "active" folder is one level up!

lines = [x.strip() for x in os.popen("ls -1 ../ | grep L1_").readlines()]
lines.sort(reverse=True)  # decreasing order, AKA most recent first
for line in lines:
	print(line)

if latest is None:
	latest = lines[0] # most recent date of AWS retrieval 
print("LATEST", latest)

to_merge = []
for tile in tiles:
	print(tile)
	for line in [x.strip() for x in os.popen("ls -1 ../" + latest + sep + "*" + tile + "*.zip").readlines()]:
		if len(line.split('swir')) > 1:
			err("please remove _swir_ files")

		to_merge += [line]
print(to_merge)

if len(to_merge) < 1:
	err("no data found, please check data are retrieved, unzipped, unpacked, converted to the appropriate format, and that this tile is imaged on the provided date")

for f in to_merge:
	run("cp -v " + f + " .")

if False:
	# ../L2_20230520/S2B_MSIL2A_20230520T190919_N0509_R056_T10UEC_20230520T214840.hdr
	first = to_merge[0][:-4] + '.hdr'
	ts = first.split(sep)[-1].split("_")[2].split('T')[1][0:4]

	out_file = latest + "_" + ts + ".bin"
	out_hdr = latest + "_" + ts + ".hdr"

	if not exists(out_file):
		run("gdal_merge.py -of ENVI -ot Float32 -n nan " + " ".join(to_merge) + " -o " + out_file)
		run("fh " + out_hdr)

	run('envi_header_copy_bandnames.py ' + first + ' ' + out_hdr)
