import os
import sys
from misc import err, run, exists, sep, band_names, read_hdr

if not exists('hyperlink') or not exists('fpf'):
	err('expected to be run from active/$FIRE_NUMBER')

# get the fire number
fire_number = os.getcwd().strip().split(sep)[-1]
print("FIRE_NUMBER", fire_number)

# get the tiles
tiles = open("/home/" + os.popen("whoami").read().strip() + sep + "GitHub/wps-research/py/.select/" + fire_number).read().strip().split()
print("TILES", tiles)

# get the latest AWS folder. Assume "active" folder is one level up!

lines = [x.strip() for x in os.popen("ls -1 ../ | grep L2_").readlines()]
lines.sort(reverse=True)  # decreasing order, AKA most recent first
for line in lines:
	print(line)

latest = lines[0] # most recent date of AWS retrieval 

to_merge = []
for tile in tiles:
	print(tile)
	for line in [x.strip() for x in os.popen("ls -1 ../" + latest + sep + "*" + tile + "*.bin").readlines()]:
		if len(line.split('swir')) > 1:
			err("please remove _swir_ files")

		to_merge += [line]
print(to_merge)

out_file = latest + ".bin"
out_hdr = latest + ".hdr"

if not exists(out_file):
	run("gdal_merge.py -of ENVI -ot Float32 -n nan " + " ".join(to_merge) + " -o " + out_file)
	run("fh " + out_hdr)

first = to_merge[0][:-4] + '.hdr'

run('envi_header_copy_bandnames.py ' + first + ' ' + out_hdr)
#samples, lines, bands = read_hdr(out_hdr)

#bn = band_names(first)


#run(' '.join(['envi_header_modify.py', out_hdr, str(lines), str(samples), str(bands)] + bn))
# merge the appropriate tiles if the merge file isn't already available

